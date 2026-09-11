"""Where run output goes between invocations.

Every run lands in its own directory under `~/.betterbench/runs/`
(`$BETTERBENCH_HOME/runs/` when the variable is set), named
`<YYYYMMDD-HHMMSS><-slug>` where the slug is the model name sanitised
(`Qwen3-30B-A3B-instruct` -> `qwen3-30b-a3b-instruct`) or a caller-supplied
`--name` label. The directory is created when it is *allocated* (at the
moment the results are about to be written — validation failures before
then leave nothing on disk), and no two runs ever collide: a second run
within the same second gets `-2`, `-3`, ... on the end instead of
overwriting. `plan_run_dir` gives the same name with no filesystem effect.

An explicit `--out` bypasses this module entirely.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_HOME_NAME = ".betterbench"
RUNS_SUBDIR = "runs"
SLUG_MAX = 64

# one stderr warning per process for a relative BETTERBENCH_HOME
_warned_relative_home = False


def betterbench_home() -> Path:
    """Base directory: `$BETTERBENCH_HOME` if set, else `~/.betterbench`.

    Not created here — creating it is the caller's business, in case it does
    not need one (e.g. `report` with no `--out`). A *relative*
    `$BETTERBENCH_HOME` is resolved against the process cwd explicitly (a
    relative path would land there anyway — the resolution is just made
    visible) and warns on stderr once per process, so the "nothing lands
    in the cwd without `--out`" guarantee stops being silent.
    `~`-prefixed, absolute, and unset values behave exactly as before."""
    global _warned_relative_home
    env = os.environ.get("BETTERBENCH_HOME")
    if not env:
        return Path.home() / DEFAULT_HOME_NAME
    p = Path(env).expanduser()   # ~ and ~user are handled here
    if p.is_absolute():
        return p
    if not _warned_relative_home:
        _warned_relative_home = True
        print(f"warning: BETTERBENCH_HOME is relative ('{env}') — runs "
              f"will land in ./{p}/runs/ (the cwd). Use an absolute path "
              "or ~ to keep runs out of the cwd.", file=sys.stderr)
    return Path.cwd() / p


def slug(text: str) -> str:
    """Sanitise a model name or label into a path-safe, lowercase slug.

    'Qwen3-30B / mx-FP8 (v2)' -> 'qwen3-30b-mx-fp8-v2'; '' if nothing remains.
    Capped to SLUG_MAX characters on the *sanitised* string so a name made
    of pure punctuation can't eat the cap; when truncated, an 8-char sha256
    prefix of the full sanitised slug is appended so distinct long labels
    collide only rarely. Readable by default, e.g.
    'qwen3-480b-instruct-9a3c1f02'.
    """
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(base) <= SLUG_MAX:
        return base
    tail = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:SLUG_MAX]}-{tail}"


def plan_run_dir(model: str, name: str | None = None) -> Path:
    """Name a run directory *without creating it*.

    Returns the candidate location a `allocate_run_dir` call would take
    (same timestamp + slug, no collision suffix) — i.e. the "where the
    results go if nothing else got there first" answer. No filesystem side
    effects: not even the `runs/` root is created, so a pre-run "output: "
    banner can use it without risking an empty orphaned directory. It may
    be one second behind (or, in a race, a `-2` off) the write-time
    allocation; the final "wrote ..." line always shows the exact name.
    """
    root = betterbench_home() / RUNS_SUBDIR
    stamp = time.strftime("%Y%m%d-%H%M%S")
    s = slug(name if name is not None else model)
    return root / (f"{stamp}" + (f"-{s}" if s else ""))


def allocate_run_dir(model: str, name: str | None = None) -> Path:
    """Create and return a uniquely-named run directory.

    The default label is the model name; `--name` overrides the slug but not
    the timestamp prefix, so directories stay sortable by time first. A
    collision (same second, same label) appends `-2`, `-3`, ... — it never
    reuses an existing path.

    Call this at the moment the results are about to be written, not at the
    start of a run that is hours away: a run that dies in validation (bad
    --config, no corpus) must not leave an empty directory behind.
    """
    root = plan_run_dir(model, name).parent
    root.mkdir(parents=True, exist_ok=True)
    base = plan_run_dir(model, name).name
    d = root / base
    n = 2
    while True:
        try:
            d.mkdir()          # exist_ok=False: the mkdir outcome is the check
            return d
        except FileExistsError:
            d = root / f"{base}-{n}"
            n += 1
