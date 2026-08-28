"""Harness self-test (plan §7H null-test gate). No GPU, no model, no network.

1. Timing accuracy: run against an in-process mock server with known TTFT + ITL
   and assert BetterBench recovers them within tolerance.
2. Null test: paired A/B of the SAME endpoint vs itself must report
   'within noise' (no phantom winner).

Exit 0 = pass.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.mock_server import make_handler                 # noqa: E402
from betterbench.client import stream_chat_sync            # noqa: E402
from betterbench.config import Config                      # noqa: E402
from betterbench.corpus import load_corpus                 # noqa: E402
from betterbench.runner import paired_ab                   # noqa: E402

PORT = 8202
TTFT_MS, ITL_MS, TOKENS = 25.0, 10.0, 30
BASE = f"http://127.0.0.1:{PORT}/v1"


def start_server() -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT),
                              make_handler(TTFT_MS, ITL_MS, TOKENS))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)
    return srv


def timing_test() -> bool:
    stream_chat_sync(BASE, "mock", [{"role": "user", "content": "hi"}],
                     max_tokens=TOKENS, temperature=0)   # warm
    ttfts, itls, tpss = [], [], []
    for _ in range(8):
        r = stream_chat_sync(BASE, "mock", [{"role": "user", "content": "hi"}],
                             max_tokens=TOKENS, temperature=0)
        assert r.ok, r.error
        ttfts.append(r.ttft_ms); itls += r.itl_ms; tpss.append(r.decode_tps)
    ttft, itl, tps = (float(np.median(x)) for x in (ttfts, itls, tpss))
    exp_tps = 1000.0 / ITL_MS
    print(f"  TTFT  {ttft:6.1f} ms  (expect ~{TTFT_MS}, +overhead)")
    print(f"  ITL   {itl:6.2f} ms  (expect ~{ITL_MS}, +overhead)")
    print(f"  decode {tps:6.1f} t/s (expect ~{exp_tps:.0f})")
    # allow loopback/HTTP overhead, but the harness must be in the ballpark
    return (0 <= ttft - TTFT_MS < 20) and (0 <= itl - ITL_MS < 6) and (tps < exp_tps + 5)


def null_test() -> bool:
    # fixed n (no early-stop peeking) + 99% CI: a true-null should be flagged
    # "not significant" ~99% of the time. Catches phantom-win noise-model bugs.
    cfg = Config(greedy=True, warmup=2, ab_min_pairs=24, ab_max_pairs=24,
                 target_mde_pct=0.0, conf=0.99, unique_nonce=True)
    corpus = load_corpus(ROOT / "corpus" / "v1", ["reasoning", "code"])
    ab = asyncio.run(paired_ab(BASE, BASE, "mock", corpus, cfg, log=lambda *a: None))
    ok = True
    for key in ("decode_tps", "gap_median"):
        d = ab[key]
        print(f"  null A/B [{key}]: pairs={ab['pairs']}  Δ={d['pct_diff']:+.2f}%  "
              f"CI[{d['ci_low_pct']:+.2f}%,{d['ci_high_pct']:+.2f}%]  "
              f"significant={d['significant']}")
        # Both rows must be null AND well-formed. The latency row went unchecked
        # here for three releases, which is how it shipped with an inverted sign
        # and its CI bounds printed back-to-front.
        if d["significant"]:
            print(f"    ! {key}: phantom win on a null comparison")
            ok = False
        if d["ci_low_pct"] > d["ci_high_pct"]:
            print(f"    ! {key}: CI bounds are reversed "
                  f"({d['ci_low_pct']:+.2f} > {d['ci_high_pct']:+.2f})")
            ok = False
    return ok


def main():
    srv = start_server()
    try:
        print("[1/2] timing accuracy vs mock server")
        t_ok = timing_test()
        print("[2/2] null A/B test (same endpoint vs itself)")
        n_ok = null_test()
    finally:
        srv.shutdown()
    print()
    print(f"timing accuracy : {'PASS' if t_ok else 'FAIL'}")
    print(f"null-test gate  : {'PASS' if n_ok else 'FAIL'}")
    sys.stdout.flush()
    sys.exit(0 if (t_ok and n_ok) else 1)


if __name__ == "__main__":
    main()
