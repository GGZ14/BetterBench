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


# Servers expose the thinking phase one of two ways: a separate delta field
# (`reasoning_content` / `reasoning`), where the channel switch is unambiguous;
# or inline in `content` between literal markers.
_OPEN_MARKERS = ("<think>", "<thinking>")
_CLOSE_MARKERS = ("</think>", "</thinking>")
_CARRY = max(len(m) for m in _OPEN_MARKERS + _CLOSE_MARKERS) - 1


def _find_marker(carry: str, text: str, markers) -> tuple[int, int] | None:
    """Locate the earliest marker spanning `carry + text`.

    Returns (end_of_before, start_of_after) as offsets into `text`, clamped to
    zero for the part of the marker that landed in the previous chunk. The carry
    is why a marker split across two SSE deltas is still found.
    """
    buf = carry + text
    best = None
    for m in markers:
        i = buf.find(m)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, i + len(m))
    if best is None:
        return None
    off = len(carry)
    return max(0, best[0] - off), max(0, best[1] - off)


@dataclass
class _Timeline:
    """Per-chunk bookkeeping gathered while the stream is still arriving.

    It only *counts*: which channel carried text, how much, and when the answer
    first appeared. Interpreting those counts is deferred to `_resolve_split`,
    because until the stream ends we do not know whether unmarked content was
    thinking or answering. Deliberately tiny — no transcript, one short carry
    buffer for a marker that straddles two updates.
    """
    times: list[float] = field(default_factory=list)
    # reasoning arriving on its own delta field
    chan_chars: int = 0
    chan_chunks: int = 0
    # everything arriving on `content`
    content_chars: int = 0
    content_chunks: int = 0
    first_content_at: int | None = None
    # inline <think>...</think> bookkeeping
    saw_open_marker: bool = False
    closed: bool = False
    pre_marker_chars: int = 0
    pre_marker_chunks: int = 0
    post_marker_at: int | None = None
    carry: str = ""

    @property
    def saw_reasoning_channel(self) -> bool:
        return self.chan_chunks > 0

    def add(self, content: str | None, reasoning: str | None) -> None:
        idx = len(self.times)
        self.times.append(time.perf_counter())
        if reasoning:
            self.chan_chars += len(reasoning)
            self.chan_chunks += 1
        if not content:
            return
        self.content_chars += len(content)
        self.content_chunks += 1
        if self.first_content_at is None:
            self.first_content_at = idx
        if self.saw_reasoning_channel or self.closed:
            return                      # the channel already told us where we are
        if not self.saw_open_marker and _find_marker(self.carry, content,
                                                     _OPEN_MARKERS):
            self.saw_open_marker = True
        hit = _find_marker(self.carry, content, _CLOSE_MARKERS)
        if hit is None:
            self.pre_marker_chars += len(content)
            self.pre_marker_chunks += 1
            self.carry = (self.carry + content)[-_CARRY:]
            return
        end, start = hit
        self.closed = True
        self.carry = ""
        if end:
            self.pre_marker_chars += end
            self.pre_marker_chunks += 1
        if len(content) - start > 0:
            self.post_marker_at = idx
        elif self.post_marker_at is None:
            self.post_marker_at = idx + 1   # answer begins with the next update


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
    # Thinking vs answering. `reasoning_source` says how (or whether) the split
    # was established; a None split means "we could not tell", never zero.
    ttfa_ms: float | None = None                 # time to first ANSWER token
    reasoning_source: str = "none"               # channel|inline_marker|none|unknown
    reasoning_chars: int | None = None
    answer_chars: int | None = None
    reasoning_chunks: int | None = None
    answer_chunks: int | None = None
    reasoning_tokens_est: int | None = None
    answer_tokens_est: int | None = None
    truncated_in_reasoning: bool = False
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


def _resolve_split(res: RunResult, tl: _Timeline, t0: float, comp: int) -> None:
    """Decide what, if anything, we can say about thinking vs answering.

    The case that matters: Qwen3-style templates open `<think>` in the *prompt*,
    so a generation cut off at max_tokens emits no marker at all — it is
    genuinely indistinguishable from a non-thinking model that got truncated.
    Recording `reasoning_tokens=0, answer_tokens=everything` there credits an
    answer that never arrived, and the error flatters. So: "unknown", and the
    split stays None.
    """
    truncated = res.finish_reason == "length"
    if tl.saw_reasoning_channel:
        # Unambiguous: reasoning came on its own field, so all `content` is answer.
        res.reasoning_source = "channel"
        r_chars, r_chunks = tl.chan_chars, tl.chan_chunks
        a_chars, a_chunks = tl.content_chars, tl.content_chunks
        answer_at = tl.first_content_at
    elif tl.closed:
        res.reasoning_source = "inline_marker"
        r_chars, r_chunks = tl.pre_marker_chars, tl.pre_marker_chunks
        a_chars = tl.content_chars - tl.pre_marker_chars
        a_chunks = tl.content_chunks - tl.pre_marker_chunks
        answer_at = tl.post_marker_at
    elif tl.saw_open_marker:
        # Opened a think block and never closed it: all thinking, no answer.
        res.reasoning_source = "inline_marker"
        res.truncated_in_reasoning = True
        r_chars, r_chunks = tl.content_chars, tl.content_chunks
        a_chars = a_chunks = 0
        answer_at = None
    elif truncated:
        # No channel, no marker, and cut off — we cannot tell. Say so.
        res.reasoning_source = "unknown"
        return
    else:
        res.reasoning_source = "none"
        r_chars = r_chunks = 0
        a_chars, a_chunks = tl.content_chars, tl.content_chunks
        answer_at = tl.first_content_at

    if tl.saw_reasoning_channel and not a_chunks and truncated:
        res.truncated_in_reasoning = True

    res.reasoning_chars, res.answer_chars = r_chars, a_chars
    res.reasoning_chunks, res.answer_chunks = r_chunks, a_chunks
    if answer_at is not None and answer_at < len(tl.times):
        res.ttfa_ms = (tl.times[answer_at] - t0) * 1000.0

    # The server reports one total, so the split is apportioned by character
    # count. Chars-per-token is not constant across phases — a JSON or code
    # answer is punctuation-dense where chain-of-thought is ordinary prose — so
    # these are estimates, named `_est`, with the raw char and chunk counts kept
    # so an exact tokenizer-based split can be computed later without re-running.
    total = r_chars + a_chars
    if comp and total:
        res.reasoning_tokens_est = round(comp * r_chars / total)
        res.answer_tokens_est = comp - res.reasoning_tokens_est


def _finalize(res: RunResult, tl: _Timeline, t0: float, t_end: float,
              usage: dict | None) -> RunResult:
    chunk_times = tl.times
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
    _resolve_split(res, tl, t0, comp)

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
    tl = _Timeline()
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
            delta = ch.get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if not isinstance(content, str) or not content:
                content = None
            if not isinstance(reasoning, str) or not reasoning:
                reasoning = None
            if content is not None or reasoning is not None:
                tl.add(content, reasoning)
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return _finalize(res, tl, t0, time.perf_counter(), usage)


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
