"""BetterBench command line: run · report · compare · ab."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import CORPUS_VERSION, RESULTS_SCHEMA, __version__
from .client import get_model_context
from .config import Config
from .corpus import load_corpus
from .env import content_hash, fingerprint
from .metrics import paired_compare
from .html_report import render_html
from .report import render_ab_markdown, render_markdown, sample_gate
from .runner import concurrency_sweep, paired_ab, prefill_sweep, single_stream


# `--quick` preset: a short smoke run, not a publishable measurement.
QUICK_PASSES = 5
QUICK_WARMUP = 1


def _kv(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key.strip(), val.strip()


def _positive_int(value: str) -> int:
    if not value.lstrip("+").isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of 1 or more, got {value!r}")
    return int(value)


def _non_negative_int(value: str) -> int:
    if not value.lstrip("+").isdigit():
        raise argparse.ArgumentTypeError(
            f"expected a whole number of 0 or more, got {value!r}")
    return int(value)


def _load(cfg_path, overrides: dict) -> Config:
    cfg = Config.load(cfg_path)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _html_path(out: str, html_out: str | None) -> Path:
    return Path(html_out) if html_out else Path(out).with_suffix(".html")


def _emit_html(results: dict, out: str, html_out: str | None, disabled: bool) -> None:
    """Write the standalone HTML report beside the results JSON."""
    if disabled:
        return
    path = _html_path(out, html_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(results), encoding="utf-8")
    print(f"\nwrote {path}  —  open it in a browser for the charted report:")
    print(f"    file://{path.resolve()}")


# The three measured phases, in the order they run: CLI selector flag -> the
# Config field that decides whether the phase happens at all.
PHASES = {"decode": "run_single_stream",
          "prefill": "run_prefill",
          "concurrency": "run_concurrency"}


def _phase_overrides(args) -> dict:
    """Resolve --decode / --prefill / --concurrency into Config overrides.

    Naming a phase selects it and *only* it, so `--prefill` is the whole
    command needed to re-measure prompt processing; naming several runs those
    several. With none named, nothing here overrides the config file — the
    older `--no-*` flags still switch a phase off, and a `run_*: false` in a
    --config file is finally honoured instead of being overwritten.
    """
    selected = set(args.only or ())
    contradicted = [f"--{n}" for n in ("prefill", "concurrency")
                    if n in selected and getattr(args, f"no_{n}")]
    if contradicted:
        sys.exit(f"{' and '.join(contradicted)} asked for and switched off in the "
                 f"same command — drop one.")
    if selected:
        return {field: (name in selected) for name, field in PHASES.items()}
    return {"run_prefill": False if args.no_prefill else None,
            "run_concurrency": False if args.no_concurrency else None}


def cmd_run(args):
    # --passes wins over the config file; --quick only fills in what wasn't asked
    # for explicitly, so `--quick --warmup 3` keeps the 3 warmups you asked for.
    passes, warmup = args.passes, args.warmup
    if args.quick:
        passes = QUICK_PASSES if passes is None else passes
        warmup = QUICK_WARMUP if warmup is None else warmup

    cfg = _load(args.config, {
        "runs_per_category": passes, "warmup": warmup,
        "greedy": args.greedy or None,
        "seed": args.seed, "max_model_len": args.max_model_len,
        **_phase_overrides(args),
    })
    phases = [name for name, field in PHASES.items() if getattr(cfg, field)]
    if not phases:
        sys.exit("every phase is switched off — nothing to measure. Pass "
                 "--decode, --prefill or --concurrency to pick one.")
    # The prefill sweep synthesises its own prompts by depth, so a prefill-only
    # run neither needs a corpus nor should record one it never read.
    needs_corpus = cfg.run_single_stream or cfg.run_concurrency
    corpus = load_corpus(args.corpus, args.categories) if needs_corpus else {}
    if not corpus and needs_corpus:
        if args.corpus:
            sys.exit(f"no prompts found in --corpus {args.corpus!r} "
                     f"(need *.jsonl files; check --categories {args.categories})")
        from .corpus import corpus_search_paths
        searched = "\n  ".join(str(p) for p in corpus_search_paths())
        sys.exit("no prompts found. Looked in:\n  " + searched +
                 "\nIf you installed with `pip install -e .`, this is a known "
                 "path issue — update to the latest version, or pass "
                 "--corpus /path/to/corpus/v1 explicitly.")
    banner = f"BetterBench {__version__} · corpus v{CORPUS_VERSION}"
    if needs_corpus:
        banner += (f" · {sum(len(v) for v in corpus.values())} prompts "
                   f"in {len(corpus)} categories")
    print(banner)
    print(f"phases: {', '.join(phases)}")
    if cfg.run_single_stream:
        print(f"{cfg.warmup} warmup + {cfg.runs_per_category} measured passes per category"
              f"{'  (quick mode — smoke check, not a publishable result)' if args.quick else ''}")

    results = {
        "schema": RESULTS_SCHEMA,
        "betterbench_version": __version__,
        "corpus_version": CORPUS_VERSION,
        "config": cfg.as_dict(),
        "env": fingerprint(args.endpoint, args.model,
                           {"categories": list(corpus.keys()),
                            "notes": dict(args.note or [])}),
    }
    results["env"]["corpus_hash"] = content_hash(
        {c: [p.id for p in ps] for c, ps in corpus.items()})

    # Resolve the model's context window: explicit override, else auto-detect.
    # Used to skip prefill depths that wouldn't fit (avoids HTTP 400s mid-sweep),
    # so only the prefill sweep is worth an extra /v1/models round trip.
    max_ctx = cfg.max_model_len
    if cfg.run_prefill:
        max_ctx = max_ctx or get_model_context(args.endpoint, args.model)
        if max_ctx:
            print(f"model max context: {max_ctx} tokens"
                  f"{'' if cfg.max_model_len else ' (auto-detected from /v1/models)'}")
        else:
            print("model max context: unknown — will skip any depth the server rejects "
                  "(pass --max-model-len to skip up front)")
    results["env"]["max_model_len"] = max_ctx

    if cfg.run_single_stream:
        results["single_stream"] = asyncio.run(
            single_stream(args.endpoint, args.model, corpus, cfg))
    if cfg.run_prefill:
        results["prefill"] = asyncio.run(
            prefill_sweep(args.endpoint, args.model, cfg, max_ctx=max_ctx))
    if cfg.run_concurrency:
        results["concurrency"] = asyncio.run(
            concurrency_sweep(args.endpoint, args.model, corpus, cfg))

    # Persist which percentiles are under-sampled, so the report's footnote is
    # a recorded fact rather than a claim recomputed at render time.
    results["sample_gate"] = sample_gate(results)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    print(render_markdown(results))
    _emit_html(results, args.out, args.html_out, args.no_html)


def cmd_report(args):
    results = json.loads(Path(args.results).read_text())
    if args.html:
        path = _html_path(args.results, args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(results), encoding="utf-8")
        print(f"wrote {path}")
        print(f"    file://{path.resolve()}")
        return
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
    # An A/B result that doesn't record which box, which GPU or which day it ran
    # is not archivable — and A/B is this tool's headline claim.
    ab = {"schema": RESULTS_SCHEMA, "betterbench_version": __version__,
          "corpus_version": CORPUS_VERSION, "config": cfg.as_dict(),
          "env": fingerprint(args.endpoint_a, args.model,
                             {"categories": list(corpus.keys()),
                              "endpoint_b": args.endpoint_b,
                              "notes": dict(args.note or [])}),
          **ab}
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
    group = r.add_mutually_exclusive_group()
    group.add_argument("--passes", type=_positive_int, metavar="N", dest="passes",
                       help=f"measured passes per category "
                            f"(default {Config().runs_per_category}, or the config file's value)")
    # --runs is the pre-0.2.2 spelling: same dest, so old command lines keep
    # working, and SUPPRESS keeps it out of --help and the usage line.
    group.add_argument("--runs", type=_positive_int, metavar="N", dest="passes",
                       help=argparse.SUPPRESS)
    group.add_argument("--quick", action="store_true",
                       help=f"short smoke run: {QUICK_PASSES} passes per category "
                            f"after {QUICK_WARMUP} warmup — too few passes for a "
                            f"publishable number, use it to check a setup")
    r.add_argument("--warmup", type=_non_negative_int, metavar="N",
                   help=f"discarded passes per category "
                        f"(default {Config().warmup}; {QUICK_WARMUP} under --quick)")
    r.add_argument("--seed", type=int)
    r.add_argument("--greedy", action="store_true", help="temperature=0 (reproducibility)")
    # Phase selection. Naming any phase runs only the phases named, so
    # `--prefill` is the whole command for "just re-measure prompt processing".
    phase = r.add_argument_group(
        "phase selection",
        "By default all three phases run. Name one or more to run only those.")
    phase.add_argument("--decode", dest="only", action="append_const", const="decode",
                       help="single-stream (batch = 1) decode only")
    phase.add_argument("--prefill", dest="only", action="append_const", const="prefill",
                       help="prompt-processing (prefill) depth sweep only "
                            "(needs no corpus)")
    phase.add_argument("--concurrency", dest="only", action="append_const",
                       const="concurrency", help="concurrency sweep only")
    r.add_argument("--no-concurrency", action="store_true", help="skip the concurrency sweep")
    r.add_argument("--no-prefill", action="store_true", help="skip the prompt-processing sweep")
    r.add_argument("--max-model-len", type=int,
                   help="model context window in tokens; overrides auto-detection. "
                        "Prefill depths that don't fit are skipped instead of erroring.")
    r.add_argument("--note", action="append", type=_kv, metavar="KEY=VALUE",
                   help='free-form metadata recorded in results.json, e.g. --note image=v0.9.3 --note quant=mxfp4 --note tp=2. Repeatable. These are the details that decide whether two results are comparable at all, and there was no way to record them.')
    r.add_argument("--out", default="results/run.json")
    r.add_argument("--no-html", action="store_true",
                   help="skip the standalone HTML report (written beside --out by default)")
    r.add_argument("--html-out", help="path for the HTML report (default: --out with .html)")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="render a results.json as markdown")
    rep.add_argument("results"); rep.add_argument("--out")
    rep.add_argument("--html", action="store_true",
                     help="render the standalone HTML report instead of markdown")
    rep.set_defaults(func=cmd_report)

    ab = sub.add_parser("ab", help="interleaved paired A/B between two endpoints")
    ab.add_argument("--endpoint-a", required=True); ab.add_argument("--endpoint-b", required=True)
    ab.add_argument("--model", required=True)
    ab.add_argument("--config"); ab.add_argument("--corpus"); ab.add_argument("--categories", nargs="*")
    ab.add_argument("--mde", type=float, default=1.0, help="target minimum detectable effect %%")
    ab.add_argument("--max-pairs", type=int, default=200)
    ab.add_argument("--seed", type=int)
    ab.add_argument("--sampled", action="store_true", help="keep sampling (default greedy)")
    ab.add_argument("--note", action="append", type=_kv, metavar="KEY=VALUE",
                    help="free-form metadata recorded in the A/B JSON (repeatable)")
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
