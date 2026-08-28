"""OpenAI-compatible streaming client with high-resolution timing.

Stdlib-only (http.client) so BetterBench needs no HTTP dependency; concurrency
is provided by wrapping the blocking request in asyncio.to_thread, which gives
real parallelism for I/O-bound token streaming.

Captures, per request:
  * TTFT             — time from send to the first content-bearing chunk (ms)
  * update gaps      — wall-clock gaps between stream updates (ms), as measured
  * ITL              — per-token latency (ms), only when the server streams one
                       token per chunk; otherwise there is nothing to report
  * decode / total tokens-per-second
  * prompt / completion token counts (reasoning tokens included)

A note on ITL and speculative decoding. Most servers stream one token per SSE
chunk, and then the gap between chunks *is* the inter-token latency. Servers
doing speculative decoding (MTP, EAGLE, Medusa, n-gram, ...) verify several
tokens in a single forward pass and write all the accepted ones into one chunk.
Those tokens did not arrive at different times — they arrived together, in the
same network write — so for tokens inside a chunk "time between tokens" has no
answer. BetterBench does not invent one: it reports the measured update gap and
the tokens per update instead, and leaves `itl_ms` empty.
"""
from __future__ import annotations

import asyncio
import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


# How far usage.completion_tokens may exceed the chunk count before we stop
# calling a stream one-token-per-chunk. Both a ratio and an absolute slack:
# `usage` counts tokens that never carried a delta (EOS, and on some servers a
# role-only opening chunk), so a genuinely 1:1 stream lands one or two tokens
# above its chunk count regardless of length — 15 of 40 records in a real
# non-speculative reference run are exactly comp == n_chunks + 1. A ratio alone
# would pass that on a 900-token generation and fail it on a 16-token prefill
# probe. The ratio is deliberately tight: whatever it admits is the worst-case
# error in any per-token latency we then report, so the 10% BetterBench used
# through 0.2.3 was ten times too loose.
CHUNK_TOKEN_TOL = 0.02
CHUNK_TOKEN_SLACK = 2


def classify_chunking(completion_tokens: int | None,
                     n_chunks: int) -> tuple[str, float | None]:
    """Did the server stream one token per update, or several?

    Returns (chunking, tokens_per_update) where chunking is one of
    "per_token", "batched" or "unknown". "unknown" means we have no trustworthy
    token count to divide by — we do not guess 1:1, because assuming a ratio we
    cannot see is the same mistake as scaling by one.
    """
    if not completion_tokens or n_chunks <= 1:
        return "unknown", None
    tpu = completion_tokens / n_chunks
    if tpu < 0.90:                # fewer tokens than chunks: usage is unreliable
        return "unknown", tpu
    if (completion_tokens - n_chunks) <= max(CHUNK_TOKEN_SLACK,
                                             CHUNK_TOKEN_TOL * n_chunks):
        return "per_token", tpu
    return "batched", tpu


@dataclass
class RunResult:
    ok: bool
    category: str = ""
    prompt_id: str = ""
    ttft_ms: float | None = None
    update_gaps_ms: list[float] = field(default_factory=list)  # measured, per chunk
    tokens_per_update: float | None = None              # completion_tokens / n_chunks
    chunking: str = "unknown"                           # per_token | batched | unknown
    decode_tps: float | None = None
    total_tps: float | None = None
    pp_tps: float | None = None                          # prompt-processing (prefill) t/s
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    n_chunks: int = 0
    chunk_token_mismatch: bool = False
    finish_reason: str | None = None
    wall_ms: float | None = None
    error: str | None = None

    @property
    def itl_ms(self) -> list[float]:
        """Per-token inter-token latency, in ms.

        Only meaningful when the server streams one token per update, in which
        case the update gap *is* the inter-token latency. Empty otherwise —
        never synthesised by scaling a batched gap down. Being a property, it
        also stays out of `as_dict()`, so results.json carries the measured
        series once rather than twice.
        """
        return self.update_gaps_ms if self.chunking == "per_token" else []

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _delta_text(delta: dict[str, Any]) -> str:
    """Content across normal + reasoning channels (Qwen3.6 etc.)."""
    out = []
    for key in ("content", "reasoning_content", "reasoning"):
        v = delta.get(key)
        if isinstance(v, str) and v:
            out.append(v)
    return "".join(out)


def _finalize(res: RunResult, chunk_times: list[float], t0: float, t_end: float,
              usage: dict | None) -> RunResult:
    res.wall_ms = (t_end - t0) * 1000.0
    res.n_chunks = len(chunk_times)
    if usage:
        res.prompt_tokens = usage.get("prompt_tokens")
        res.completion_tokens = usage.get("completion_tokens")
    if res.n_chunks == 0:
        res.error = res.error or "no content chunks received"
        return res

    res.ttft_ms = (chunk_times[0] - t0) * 1000.0
    # Prompt-processing throughput: prompt tokens / TTFT. TTFT includes a small
    # first-token decode + network term, negligible for large prompts; this is
    # the standard prefill-throughput approximation.
    if res.prompt_tokens and res.ttft_ms and res.ttft_ms > 0:
        res.pp_tps = res.prompt_tokens / (res.ttft_ms / 1000.0)
    # The measured series: wall-clock time between stream updates. Never derived.
    res.update_gaps_ms = [(chunk_times[i] - chunk_times[i - 1]) * 1000.0
                          for i in range(1, len(chunk_times))]

    comp = res.completion_tokens or res.n_chunks
    res.chunking, res.tokens_per_update = classify_chunking(res.completion_tokens,
                                                            res.n_chunks)
    res.chunk_token_mismatch = (res.chunking == "batched")   # kept for old readers

    decode_wall = chunk_times[-1] - chunk_times[0]
    if decode_wall > 0 and comp > 1:
        res.decode_tps = (comp - 1) / decode_wall     # first token belongs to TTFT
    total_wall = t_end - t0
    if total_wall > 0 and comp:
        res.total_tps = comp / total_wall
    res.ok = True
    return res


def _build_payload(model, messages, *, max_tokens, temperature, top_p, top_k,
                   seed, extra_body):
    p: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_k is not None:
        p["top_k"] = top_k
    if seed is not None:
        p["seed"] = seed
    if extra_body:
        p.update(extra_body)
    return p


def stream_chat_sync(base_url: str, model: str, messages: list[dict], *,
                     max_tokens: int, temperature: float, top_p: float = 0.95,
                     top_k: int | None = 20, seed: int | None = None,
                     category: str = "", prompt_id: str = "",
                     extra_body: dict | None = None,
                     timeout: float = 600.0) -> RunResult:
    url = base_url.rstrip("/") + "/chat/completions"
    u = urllib.parse.urlparse(url)
    payload = _build_payload(model, messages, max_tokens=max_tokens,
                             temperature=temperature, top_p=top_p, top_k=top_k,
                             seed=seed, extra_body=extra_body)
    res = RunResult(ok=False, category=category, prompt_id=prompt_id)
    ConnCls = (http.client.HTTPSConnection if u.scheme == "https"
               else http.client.HTTPConnection)
    conn = ConnCls(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                   timeout=timeout)
    chunk_times: list[float] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        path = u.path + (("?" + u.query) if u.query else "")
        conn.request("POST", path, json.dumps(payload),
                     {"Content-Type": "application/json", "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            res.error = f"HTTP {resp.status}: {resp.read()[:300].decode('utf-8','replace')}"
            return res
        for raw in resp:                       # yields dechunked lines
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason"):
                res.finish_reason = ch["finish_reason"]
            if _delta_text(ch.get("delta") or {}):
                chunk_times.append(time.perf_counter())
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return _finalize(res, chunk_times, t0, time.perf_counter(), usage)


async def stream_chat(base_url: str, model: str, messages: list[dict],
                      **kwargs) -> RunResult:
    """Async wrapper — real parallelism for the concurrency sweep via threads."""
    return await asyncio.to_thread(stream_chat_sync, base_url, model, messages, **kwargs)


def get_model_context(base_url: str, model: str,
                      timeout: float = 5.0) -> int | None:
    """Best-effort probe of the model's max context window via GET /v1/models.

    vLLM (and several other OpenAI-compatible servers) expose `max_model_len`
    per model entry. Returns the length in tokens, or None if unavailable — in
    which case the caller falls back to runtime detection of context-length
    errors. Never raises.
    """
    url = base_url.rstrip("/") + "/models"
    u = urllib.parse.urlparse(url)
    ConnCls = (http.client.HTTPSConnection if u.scheme == "https"
               else http.client.HTTPConnection)
    try:
        conn = ConnCls(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                       timeout=timeout)
        conn.request("GET", u.path + (("?" + u.query) if u.query else ""))
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            return None
        obj = json.loads(resp.read().decode("utf-8", "replace"))
        conn.close()
    except Exception:
        return None
    data = obj.get("data") or []
    if not data:
        return None
    entry = next((m for m in data if m.get("id") == model), data[0])
    for key in ("max_model_len", "max_context_length", "context_length",
                "max_seq_len", "context_window"):
        v = entry.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return None


def is_context_length_error(err: str | None) -> bool:
    """True if a request error is the server rejecting an over-long prompt."""
    if not err:
        return False
    e = err.lower()
    return ("maximum context length" in e
            or "context length" in e
            or "reduce the length" in e
            or "longer than the maximum" in e
            or "please reduce" in e)


def ping(base_url: str, timeout: float = 3.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    u = urllib.parse.urlparse(url)
    ConnCls = (http.client.HTTPSConnection if u.scheme == "https"
               else http.client.HTTPConnection)
    try:
        conn = ConnCls(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                       timeout=timeout)
        conn.request("GET", u.path + (("?" + u.query) if u.query else ""))
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except Exception:
        return False
