"""Best-effort environment fingerprint for reproducibility (plan §6.5)."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import time
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


def gpu_info() -> dict[str, Any]:
    """Try AMD then NVIDIA. Never fails; returns {} if nothing found."""
    info: dict[str, Any] = {}
    if shutil.which("rocm-smi"):
        name = _run(["rocm-smi", "--showproductname", "--csv"])
        clk = _run(["rocm-smi", "--showsclkrange"])
        if name:
            info["vendor"] = "amd"
            info["rocm_smi_productname"] = name.splitlines()[-3:] if name else None
        if clk:
            info["sclk_range_raw"] = clk.splitlines()[:4]
    elif shutil.which("nvidia-smi"):
        q = _run(["nvidia-smi",
                  "--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw",
                  "--format=csv,noheader"])
        if q:
            info["vendor"] = "nvidia"
            info["nvidia_smi"] = q.splitlines()
    return info


def fingerprint(endpoint: str, model: str, extra: dict | None = None) -> dict[str, Any]:
    fp = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "endpoint": endpoint,
        "model": model,
        "gpu": gpu_info(),
    }
    if extra:
        # Never let caller-supplied metadata shadow a measured fact: --note
        # host=whatever must not overwrite the real host this ran on.
        clash = sorted(set(extra) & set(fp))
        if clash:
            raise ValueError(f"metadata keys collide with the environment "
                             f"fingerprint: {', '.join(clash)}")
        fp.update(extra)
    return fp


def content_hash(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]
