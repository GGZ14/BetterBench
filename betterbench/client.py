"""OpenAI-compatible streaming client with high-resolution timing.

Stdlib-only (http.client) so BetterBench needs no HTTP dependency; concurrency
is provided by wrapping the blocking request in asyncio.to_thread, which gives
real parallelism for I/O-bound token streaming.

Captures, per request:
  * TTFT   — time from send to the first content-bearing chunk (ms)
  * ITL    — per-token inter-token latency (ms), reconciled against usage
  * decode / total tokens-per-second
  * prompt / completion token counts (reasoning tokens included)
"""
from __future__ import annotations

import asyncio
import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    ok: bool
    category: str = ""
    prompt_id: str = ""
    ttft_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)   # per-token (reconciled)
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
    gaps = [(chunk_times[i] - chunk_times[i - 1]) * 1000.0
            for i in range(1, len(chunk_times))]

    comp = res.completion_tokens or res.n_chunks
    if comp and abs(comp - res.n_chunks) / max(comp, 1) > 0.10:
        res.chunk_token_mismatch = True
    scale = (res.n_chunks / comp) if comp else 1.0   # per-chunk gap -> per-token
    res.itl_ms = [g * scale for g in gaps]

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
