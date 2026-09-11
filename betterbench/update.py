"""Is there a newer BetterBench than the one running?

A benchmark a version behind can be measuring something the current release
already fixed, so the CLI says so — at most once a day, and never in a way
that can slow a run down, perturb it, or fail it.

Three properties, in the order they matter:

1. **It cannot touch a measurement.** The check runs on a background thread
   started before any measuring and joined by `settle()` before the first
   measured request, so an HTTPS round trip to github.com is never in flight
   while a request to the endpoint under test is being timed.
2. **It cannot fail a run.** Offline, DNS refused, rate limited, a proxy that
   swallows it, garbage JSON, an unwritable cache — every one of those is
   silence. This feature may never be the reason a benchmark did not run.
3. **It is nearly always free.** The answer is cached in
   `$BETTERBENCH_HOME/update-check.json` for a day, so the ordinary
   invocation makes no network call at all.

Off entirely with `--no-update-check` or `BETTERBENCH_NO_UPDATE_CHECK=1`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from . import __version__
from .runs import betterbench_home

REPO = os.environ.get("BETTERBENCH_REPO", "GGZ14/BetterBench")
CACHE_NAME = "update-check.json"
CACHE_TTL_S = 24 * 60 * 60
FETCH_TIMEOUT_S = 3.0

_thread: threading.Thread | None = None
_latest: str | None = None


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """`v0.4.1` -> `(0, 4, 1)`; None for anything that is not a plain release.

    Deliberately strict. A tag carrying a suffix — `0.5.0rc1`, `0.4.0-hotfix`
    — is not something to tell users to upgrade to, and a lenient parse that
    read it as `0.5.0` would announce a release candidate as the latest
    version. Unparseable means "ignore this tag", never "guess".
    """
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def _is_newer(latest: str, current: str) -> bool:
    """Compare padded, so 0.4 == 0.4.0 and 0.10.0 > 0.9.0 (not a string sort)."""
    a, b = _version_tuple(latest), _version_tuple(current)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def _fetch_latest() -> str | None:
    """Highest release tag on the repo, or None.

    Tags rather than `releases/latest`: a plain `git push --tag` is enough to
    be found, with no GitHub Release object needed. The order GitHub returns
    tags in is not semver order, so pick the maximum rather than the first.
    """
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/tags?per_page=100",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": f"betterbench/{__version__}"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
        if r.status != 200:
            return None
        tags = json.loads(r.read().decode("utf-8", "replace"))
    best: tuple[tuple[int, ...], str] | None = None
    for t in tags if isinstance(tags, list) else []:
        name = (t or {}).get("name") or "" if isinstance(t, dict) else ""
        v = _version_tuple(name)
        if v is not None and (best is None or v > best[0]):
            best = (v, name)
    return best[1] if best else None


def _read_cache(path: Path) -> str | None:
    """The cached answer, or None when there isn't a fresh one.

    `""` is a real answer meaning "checked, and the repo had no release tag" —
    distinct from None, or a repo without tags would be re-fetched on every
    single invocation.
    """
    try:
        c = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(c["checked_at"]) < CACHE_TTL_S:
            return str(c.get("latest") or "")
    except Exception:      # noqa: BLE001 - absent, unreadable, corrupt: recheck
        pass
    return None


def _write_cache(path: Path, latest: str | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": time.time(),
                                    "latest": latest or ""}), encoding="utf-8")
    except OSError:        # a read-only home is not this feature's business
        pass


def _worker(cache: Path) -> None:
    global _latest
    cached = _read_cache(cache)
    if cached is not None:
        _latest = cached or None
        return
    try:
        latest = _fetch_latest()
    except Exception:      # noqa: BLE001 - see property 2 in the module docstring
        return
    _write_cache(cache, latest)
    _latest = latest


def start(disabled: bool = False) -> None:
    """Begin the check in the background. Never raises, never blocks."""
    global _thread, _latest
    _thread, _latest = None, None
    if disabled or os.environ.get("BETTERBENCH_NO_UPDATE_CHECK"):
        return
    try:
        # Resolved here, on the calling thread: betterbench_home() warns once
        # per process about a relative $BETTERBENCH_HOME, and two threads
        # racing that warning could print it twice or interleave it.
        cache = betterbench_home() / CACHE_NAME
    except Exception:      # noqa: BLE001
        return
    _thread = threading.Thread(target=_worker, args=(cache,), daemon=True)
    _thread.start()


def settle(timeout: float = 0.5) -> None:
    """Wait, briefly, for the check to be done before measuring starts.

    A check that has not answered by then is abandoned rather than waited on:
    the thread is a daemon, and its answer will be in the cache next time. The
    point is only that nothing of ours is on the wire once timing begins.
    """
    if _thread is not None:
        _thread.join(timeout=timeout)


def notice() -> str | None:
    """The one-line upgrade notice, or None when there is nothing to say."""
    if not _latest or not _is_newer(_latest, __version__):
        return None
    return (f"\nA newer BetterBench is available: {_latest} "
            f"(running {__version__}) — https://github.com/{REPO}/releases\n"
            f"    (silence this with --no-update-check or "
            f"BETTERBENCH_NO_UPDATE_CHECK=1)")
