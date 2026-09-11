"""Where run output goes between invocations.

Every run lands in its own directory under `~/.betterbench/runs/`
(`$BETTERBENCH_HOME/runs/` when the variable is set), named
`<YYYYMMDD-HHMMSS><-slug>` where the slug is the model name sanitised
(`Qwen3-30B-A3B-instruct` -> `qwen3-30b-a3b-instruct`) or a caller-supplied
`--name` label. The directory is created when allocated, so the caller can
start writing immediately — and no two runs ever collide: a second run within
the same second gets `-2`, `-3`, ... on the end instead of overwriting.

An explicit `--out` bypasses this module entirely.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

DEFAULT_HOME_NAME = ".betterbench"
RUNS_SUBDIR = "runs"


def betterbench_home() -> Path:
    """Base directory: `$BETTERBENCH_HOME` if set, else `~/.betterbench`.

    Not created here — creating it is the caller's business, in case it does
    not need one (e.g. `report` with no `--out`)."""
    env = os.environ.get("BETTERBENCH_HOME")
    return Path(env).expanduser() if env else Path.home() / DEFAULT_HOME_NAME


def slug(text: str) -> str:
    """Sanitise a model name or label into a path-safe, lowercase slug.

    'Qwen3-30B / mx-FP8 (v2)' -> 'qwen3-30b-mx-fp8-v2'; '' if nothing remains.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def allocate_run_dir(model: str, name: str | None = None) -> Path:
    """Create and return a uniquely-named run directory.

    The default label is the model name; `--name` overrides the slug but not
    the timestamp prefix, so directories stay sortable by time first. A
    collision (same second, same label) appends `-2`, `-3`, ... — it never
    reuses an existing path.
    """
    root = betterbench_home() / RUNS_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    s = slug(name if name is not None else model)
    base = f"{stamp}" + (f"-{s}" if s else "")
    d = root / base
    n = 2
    while d.exists():
        d = root / f"{base}-{n}"
        n += 1
    d.mkdir(exist_ok=False)
    return d
