"""Where run output goes between invocations.

Every run lands in its own directory under `~/.betterbench/runs/`
(`$BETTERBENCH_HOME/runs/` when the variable is set), named
`<YYYYMMDD-HHMMSS><-slug>` where the slug is the model name sanitised
(`Qwen3-30B-A3B-instruct` -> `qwen3-30b-a3b-instruct`) or a caller-supplied
`--name` label. The directory is created when it is *allocated* (once the run
has cleared validation and is about to start measuring — a run that fails
before then leaves nothing on disk), and no two runs ever collide: a second
run within the same second gets `-2`, `-3`, ... on the end instead of
overwriting. `plan_run_dir` gives the same name with no filesystem effect.

An explicit `--out` bypasses this module entirely.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import unicodedata
from pathlib import Path


def _now_stamp() -> str:
    """The run directory's timestamp. Named so a test can freeze it without
    monkeypatching `time.strftime` itself, which is the one the environment
    fingerprint stamps results with."""
    return time.strftime("%Y%m%d-%H%M%S")


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

    'Qwen3-30B / mx-FP8 (v2)' -> 'qwen3-30b-mx-fp8-v2';
    'Módèle-XL (mx-fp8)' -> 'modele-xl-mx-fp8'; '' if nothing remains.
    Accented non-ASCII letters are transliterated (NFKD + ASCII) before any
    sanitising, so 'Módèle' becomes 'modele' instead of 'm-d-le'.
    Capped to SLUG_MAX characters on the *sanitised* string so a name made
    of pure punctuation can't eat the cap; when truncated, an 8-char sha256
    prefix of the full sanitised slug is appended so distinct long labels
    collide only rarely. Readable by default, e.g.
    'qwen3-480b-instruct-9a3c1f02'.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(base) <= SLUG_MAX:
        return base
    tail = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
    # rstrip: the cap can land on a separator, and 'qwen3-30b--9a3c1f02' reads
    # like a missing path element rather than a truncation.
    return f"{base[:SLUG_MAX].rstrip('-')}-{tail}"


def plan_run_dir(model: str, name: str | None = None) -> Path:
    """Name a run directory *without creating it*.

    Returns the candidate location a `allocate_run_dir` call would take
    (same timestamp + slug, no collision suffix) — i.e. the "where the
    results go if nothing else got there first" answer. No filesystem side
    effects: not even the `runs/` root is created. It may be one second behind
    (or, in a race, a `-2` off) a later allocation, so a caller that needs the
    real path allocates rather than planning.
    """
    root = betterbench_home() / RUNS_SUBDIR
    stamp = _now_stamp()
    s = slug(name if name is not None else model)
    return root / (f"{stamp}" + (f"-{s}" if s else ""))


def allocate_run_dir(model: str, name: str | None = None) -> Path:
    """Create and return a uniquely-named run directory.

    The default label is the model name; `--name` overrides the slug but not
    the timestamp prefix, so directories stay sortable by time first. A
    collision (same second, same label) appends `-2`, `-3`, ... — it never
    reuses an existing path.

    Call this once the run is committed to measuring — after the validation
    that can `sys.exit` (bad --config, no corpus), so a run that never took a
    measurement leaves no empty directory behind, but before the measuring
    itself, so an unwritable destination is found in the first second rather
    than after hours of work that then has nowhere to go.
    """
    candidate = plan_run_dir(model, name)
    root, base = candidate.parent, candidate.name
    root.mkdir(parents=True, exist_ok=True)
    d = candidate
    n = 2
    while True:
        try:
            d.mkdir()          # exist_ok=False: the mkdir outcome is the check
            return d
        except FileExistsError:
            d = root / f"{base}-{n}"
            n += 1
