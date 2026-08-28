"""A tiny, stdlib-only mock OpenAI-compatible streaming server for validating
BetterBench's timing accuracy — no GPU, no model, no extra deps.

Streams `n` tokens with a fixed TTFT and a fixed inter-token delay, and reports
usage in the final chunk. Used by the harness self-test (tools/self_test.py).

Run:  python tools/mock_server.py --port 8099 --ttft-ms 40 --itl-ms 10
Then: betterbench run --endpoint http://127.0.0.1:8099/v1 --model mock ...
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(ttft_ms: float, itl_ms: float, tokens: int, max_ctx: int = 0, *,
                 tokens_per_chunk: int = 1, stall_every: int = 0,
                 stall_ms: float = 0.0, usage_extra_tokens: int = 0,
                 no_usage: bool = False, reasoning: str = "off",
                 reasoning_tokens: int = 0, finish_reason: str = "stop"):
    """Build the request handler.

    Beyond the fixed-timing defaults (one token per chunk, which is what the
    self-test's timing gate measures), the keyword knobs let a test reproduce
    the stream shapes real servers produce:

      tokens_per_chunk    pack N tokens into each SSE delta, as speculative
                          decoding does — the shape that has no per-token ITL
      stall_every/_ms     inject a known long gap, so a tail assertion has a
                          ground truth
      usage_extra_tokens  report more completion_tokens than deltas sent (EOS),
                          the off-by-one that a strict 1:1 test misreads
      no_usage            omit the usage chunk entirely
      reasoning           "channel" (reasoning_content deltas) or "inline"
                          (a literal <think>...</think> inside content)
      finish_reason       force "length" for the truncated-mid-thought path
    """
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            if self.path.rstrip("/").endswith("/v1/models"):
                entry = {"id": "mock"}
                if max_ctx:                      # advertise context like vLLM does
                    entry["max_model_len"] = max_ctx
                body = json.dumps({"data": [entry]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            n = min(int(body.get("max_tokens", tokens)), tokens)
            # realistic prompt token count (~4 chars/token) so the prefill sweep
            # produces meaningful numbers under test
            in_chars = sum(len(str(m.get("content", "")))
                           for m in body.get("messages", []))
            prompt_tokens = max(1, in_chars // 4)
            # Emulate a limited-context server: reject over-long prompts the way
            # vLLM/OpenAI do, so the harness's ctx-skip logic can be tested.
            if max_ctx and prompt_tokens + n > max_ctx:
                msg = (f"This model's maximum context length is {max_ctx} tokens. "
                       f"However, you requested {n} output tokens and your prompt "
                       f"contains {prompt_tokens} input tokens, for a total of "
                       f"{prompt_tokens + n} tokens. Please reduce the length of the "
                       f"input prompt or the number of requested output tokens.")
                err = json.dumps({"error": {"message": msg, "type": "BadRequestError",
                                            "code": 400}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return
            # Connection: close + no Content-Length => client reads until EOF.
            # Simpler and more robust than hand-rolled chunked framing.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def sse(obj: dict):
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            time.sleep(ttft_ms / 1000.0)
            per = max(1, tokens_per_chunk)
            n_think = min(reasoning_tokens, n) if reasoning != "off" else 0
            emitted = 0
            chunk_i = 0
            opened = False
            while emitted < n:
                k = min(per, n - emitted)
                # Which channel do the tokens in this update belong to?
                in_think = emitted < n_think
                text = "x " * k
                if reasoning == "channel" and in_think:
                    delta = {"reasoning_content": text}
                elif reasoning == "inline":
                    if in_think and not opened:
                        text, opened = "<think>" + text, True
                    elif not in_think and opened:
                        text, opened = "</think>" + text, False
                    delta = {"content": text}
                else:
                    delta = {"content": text}
                sse({"choices": [{"delta": delta, "finish_reason": None}]})
                emitted += k
                chunk_i += 1
                if emitted < n:
                    gap = itl_ms * k
                    if stall_every and chunk_i % stall_every == 0:
                        gap += stall_ms
                    time.sleep(gap / 1000.0)
            last = {"choices": [{"delta": {}, "finish_reason": finish_reason}]}
            if not no_usage:
                comp = n + usage_extra_tokens
                last["usage"] = {"prompt_tokens": prompt_tokens,
                                 "completion_tokens": comp,
                                 "total_tokens": prompt_tokens + comp}
            sse(last)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ttft-ms", type=float, default=40.0)
    ap.add_argument("--itl-ms", type=float, default=10.0)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--tokens-per-chunk", type=int, default=1,
                    help="pack N tokens into each SSE update (speculative decoding)")
    ap.add_argument("--stall-every", type=int, default=0)
    ap.add_argument("--stall-ms", type=float, default=0.0)
    ap.add_argument("--usage-extra-tokens", type=int, default=0)
    ap.add_argument("--no-usage", action="store_true")
    ap.add_argument("--reasoning", choices=["off", "channel", "inline"], default="off")
    ap.add_argument("--reasoning-tokens", type=int, default=0)
    ap.add_argument("--finish-reason", default="stop")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port),
                              make_handler(a.ttft_ms, a.itl_ms, a.tokens,
                                           tokens_per_chunk=a.tokens_per_chunk,
                                           stall_every=a.stall_every,
                                           stall_ms=a.stall_ms,
                                           usage_extra_tokens=a.usage_extra_tokens,
                                           no_usage=a.no_usage,
                                           reasoning=a.reasoning,
                                           reasoning_tokens=a.reasoning_tokens,
                                           finish_reason=a.finish_reason))
    print(f"mock server on http://127.0.0.1:{a.port}  ttft={a.ttft_ms}ms itl={a.itl_ms}ms")
    srv.serve_forever()


if __name__ == "__main__":
    main()
