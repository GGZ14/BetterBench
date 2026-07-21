"""BetterBench command line: run · report · compare · ab."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import CORPUS_VERSION, RESULTS_SCHEMA, __version__
from .config import Config
from .corpus import load_corpus
from .env import content_hash, fingerprint
from .metrics import paired_compare
from .report import render_ab_markdown, render_markdown
from .runner import concurrency_sweep, paired_ab, prefill_sweep, single_stream


def _load(cfg_path, overrides: dict) -> Config:
    cfg = Config.load(cfg_path)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def cmd_run(args):
    cfg = _load(args.config, {
        "runs_per_category": args.runs, "warmup": args.warmup,
        "greedy": args.greedy or None, "run_concurrency": not args.no_concurrency,
        "seed": args.seed,
    })
    corpus = load_corpus(args.corpus, args.categories)
    if not corpus:
        if args.corpus:
            sys.exit(f"no prompts found in --corpus {args.corpus!r} "
                     f"(need *.jsonl files; check --categories {args.categories})")
        from .corpus import corpus_search_paths
        searched = "\n  ".join(str(p) for p in corpus_search_paths())
        sys.exit("no prompts found. Looked in:\n  " + searched +
                 "\nIf you installed with `pip install -e .`, this is a known "
                 "path issue — update to the latest version, or pass "
                 "--corpus /path/to/corpus/v1 explicitly.")
    print(f"BetterBench {__version__} · corpus v{CORPUS_VERSION} · "
          f"{sum(len(v) for v in corpus.values())} prompts in {len(corpus)} categories")

    results = {
        "schema": RESULTS_SCHEMA,
        "betterbench_version": __version__,
        "corpus_version": CORPUS_VERSION,
        "config": cfg.as_dict(),
        "env": fingerprint(args.endpoint, args.model,
                           {"categories": list(corpus.keys())}),
    }
    results["env"]["corpus_hash"] = content_hash(
        {c: [p.id for p in ps] for c, ps in corpus.items()})

    results["single_stream"] = asyncio.run(
        single_stream(args.endpoint, args.model, corpus, cfg))
    if cfg.run_prefill and not args.no_prefill:
        results["prefill"] = asyncio.run(
            prefill_sweep(args.endpoint, args.model, cfg))
    if cfg.run_concurrency and not args.no_concurrency:
        results["concurrency"] = asyncio.run(
            concurrency_sweep(args.endpoint, args.model, corpus, cfg))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    print(render_markdown(results))


def cmd_report(args):
    results = json.loads(Path(args.results).read_text())
    md = render_markdown(results)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


def cmd_ab(args):
    cfg = _load(args.config, {
        "greedy": True if not args.sampled else None,   # A/B defaults to greedy
        "target_mde_pct": args.mde, "ab_max_pairs": args.max_pairs, "seed": args.seed,
    })
    corpus = load_corpus(args.corpus, args.categories)
    ab = asyncio.run(paired_ab(args.endpoint_a, args.endpoint_b, args.model, corpus, cfg))
    if args.out:
        Path(args.out).write_text(json.dumps(ab, indent=2))
    print(render_ab_markdown(ab))


def cmd_compare(args):
    """Offline paired compare of two results.json (per-category decode-tps).
    Note: only valid if both were collected on the same warm box / interleaved —
    for a rigorous comparison use `ab`."""
    A = json.loads(Path(args.a).read_text())
    B = json.loads(Path(args.b).read_text())
    print("# BetterBench compare (offline, per-category decode t/s)\n")
    print("| category | A med | B med | Δ% | 95% CI | verdict |")
    print("|---|--:|--:|--:|---|---|")
    for cat in sorted(set(A.get("single_stream", {})) & set(B.get("single_stream", {}))):
        a = [r["decode_tps"] for r in A["single_stream"][cat] if r.get("decode_tps")]
        b = [r["decode_tps"] for r in B["single_stream"][cat] if r.get("decode_tps")]
        n = min(len(a), len(b))
        if n < 2:
            continue
        pr = paired_compare(a[:n], b[:n], cat, higher_is_better=True)
        import numpy as np
        print(f"| {cat} | {np.median(a):.1f} | {np.median(b):.1f} | {pr.pct_diff:+.2f}% | "
              f"[{pr.ci_low_pct:+.1f}%,{pr.ci_high_pct:+.1f}%] | "
              f"{'SIG' if pr.significant else 'noise'} |")
    print("\n*Cross-file compares are unpaired in time; prefer `betterbench ab` "
          "for interleaved, drift-cancelled comparisons.*")


def main(argv=None):
    p = argparse.ArgumentParser(prog="betterbench",
                                description="Real-world LLM inference benchmark.")
    p.add_argument("--version", action="version", version=f"BetterBench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark one endpoint")
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base, e.g. http://host:8080/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--config"); r.add_argument("--corpus")
    r.add_argument("--categories", nargs="*")
    r.add_argument("--runs", type=int); r.add_argument("--warmup", type=int)
    r.add_argument("--seed", type=int)
    r.add_argument("--greedy", action="store_true", help="temperature=0 (reproducibility)")
    r.add_argument("--no-concurrency", action="store_true")
    r.add_argument("--no-prefill", action="store_true", help="skip the prompt-processing sweep")
    r.add_argument("--out", default="results/run.json")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="render a results.json as markdown")
    rep.add_argument("results"); rep.add_argument("--out")
    rep.set_defaults(func=cmd_report)

    ab = sub.add_parser("ab", help="interleaved paired A/B between two endpoints")
    ab.add_argument("--endpoint-a", required=True); ab.add_argument("--endpoint-b", required=True)
    ab.add_argument("--model", required=True)
    ab.add_argument("--config"); ab.add_argument("--corpus"); ab.add_argument("--categories", nargs="*")
    ab.add_argument("--mde", type=float, default=1.0, help="target minimum detectable effect %%")
    ab.add_argument("--max-pairs", type=int, default=200)
    ab.add_argument("--seed", type=int)
    ab.add_argument("--sampled", action="store_true", help="keep sampling (default greedy)")
    ab.add_argument("--out")
    ab.set_defaults(func=cmd_ab)

    cmp = sub.add_parser("compare", help="offline compare two results.json")
    cmp.add_argument("a"); cmp.add_argument("b")
    cmp.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    Path(getattr(args, "out", "results/x") or "results/x").parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
