"""Run configuration. Defaults live here; a JSON file and CLI flags override."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    # sampling
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = 20
    seed: int | None = None          # set (with greedy) for repeatability mode
    greedy: bool = False             # temperature=0 override for tight A/B

    # single-stream
    warmup: int = 3                  # discarded runs per category
    runs_per_category: int = 20      # measured runs per category
    unique_nonce: bool = True        # defeat prefix cache (honest prefill)

    # concurrency sweep
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    concurrency_requests: int = 48   # completed requests per level
    run_concurrency: bool = True

    # prompt-processing (prefill) sweep — depth in tokens, tiny decode
    run_prefill: bool = True
    prefill_depths: list[int] = field(default_factory=lambda: [2000, 8000, 16000, 32000, 64000])
    prefill_runs: int = 8
    prefill_warmup: int = 2
    prefill_max_tokens: int = 16     # keep decode tiny so TTFT is dominated by prefill
    prefill_ctx_margin: int = 256    # headroom (chat template/system tokens) when ctx-checking

    # context window: override the model's max context; if None it is auto-detected
    # from GET /v1/models (max_model_len). Depths that don't fit are skipped, not run.
    max_model_len: int | None = None

    # combined-score category weights (must cover the categories you run)
    weights: dict[str, float] = field(default_factory=lambda: {
        "code": 0.30, "reasoning": 0.20, "prose": 0.15, "json": 0.15,
        "file_edit": 0.10, "summarization": 0.10,
    })

    # repeatability / A/B
    conf: float = 0.95
    target_mde_pct: float = 1.0      # minimum detectable effect goal
    ab_max_pairs: int = 200          # ceiling for sequential run-to-confidence
    ab_min_pairs: int = 12

    # misc
    timeout_s: float = 600.0

    def effective_temp(self) -> float:
        return 0.0 if self.greedy else self.temperature

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg
