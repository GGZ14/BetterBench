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


def make_handler(ttft_ms: float, itl_ms: float, tokens: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            if self.path.rstrip("/").endswith("/v1/models"):
                body = json.dumps({"data": [{"id": "mock"}]}).encode()
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
            for i in range(n):
                sse({"choices": [{"delta": {"content": "x "}, "finish_reason": None}]})
                if i < n - 1:
                    time.sleep(itl_ms / 1000.0)
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                           "total_tokens": prompt_tokens + n}})
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
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port),
                              make_handler(a.ttft_ms, a.itl_ms, a.tokens))
    print(f"mock server on http://127.0.0.1:{a.port}  ttft={a.ttft_ms}ms itl={a.itl_ms}ms")
    srv.serve_forever()


if __name__ == "__main__":
    main()
