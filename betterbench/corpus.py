"""Load the versioned prompt corpus (JSONL per category)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def corpus_search_paths(version: str = "v1") -> list[Path]:
    """Candidate corpus locations, in priority order.

    A built wheel copies ``corpus`` to ``betterbench/corpus`` (see pyproject
    force-include), so the in-package path wins there. An editable install
    (``pip install -e .``) or a git checkout uses the source tree as-is, where
    the corpus sits at the repo root — hence the parent fallback. Without this,
    editable installs raise "no prompts found".
    """
    return [
        _HERE / "corpus" / version,          # installed wheel layout
        _HERE.parent / "corpus" / version,   # source tree / editable layout
    ]


def _resolve_default_dir(version: str = "v1") -> Path:
    for d in corpus_search_paths(version):
        if d.is_dir() and any(d.glob("*.jsonl")):
            return d
    # nothing found — return the first candidate so callers can report a path
    return corpus_search_paths(version)[0]


_DEFAULT_DIR = _resolve_default_dir()


@dataclass
class Prompt:
    id: str
    category: str
    input_len_bucket: str      # short | medium | long
    output_len_bucket: str     # short | medium | long
    max_tokens: int
    messages: list[dict]

    @classmethod
    def from_obj(cls, o: dict) -> "Prompt":
        return cls(
            id=o["id"],
            category=o["category"],
            input_len_bucket=o.get("input_len_bucket", "short"),
            output_len_bucket=o.get("output_len_bucket", "medium"),
            max_tokens=int(o.get("max_tokens", 512)),
            messages=o["messages"],
        )


def load_corpus(corpus_dir: str | Path | None = None,
                categories: list[str] | None = None) -> dict[str, list[Prompt]]:
    d = Path(corpus_dir) if corpus_dir else _DEFAULT_DIR
    out: dict[str, list[Prompt]] = {}
    for f in sorted(d.glob("*.jsonl")):
        cat = f.stem
        if categories and cat not in categories:
            continue
        prompts = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(Prompt.from_obj(json.loads(line)))
        if prompts:
            out[cat] = prompts
    return out


def with_nonce(messages: list[dict], nonce: str) -> list[dict]:
    """Prepend a unique tag to the first user message so repeated runs do NOT
    hit the server's prefix cache (honest prefill timing). See plan §3.3."""
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m.get("role") == "user":
            m["content"] = f"[bb:{nonce}] " + str(m["content"])
            break
    else:
        msgs.insert(0, {"role": "user", "content": f"[bb:{nonce}]"})
    return msgs


def nonce(rng: random.Random) -> str:
    return f"{rng.randrange(16**8):08x}"
