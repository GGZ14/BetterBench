"""Best-effort environment fingerprint for reproducibility (plan §6.5)."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


_PCI_VENDORS = {"0x8086": "intel", "0x1002": "amd", "0x10de": "nvidia"}


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def pci_gpus(sysfs: str = "/sys/class/drm") -> list[dict[str, Any]]:
    """PCIe link and BAR sizes of every GPU, read from sysfs (Linux only).

    Vendor-neutral and tool-free, so it also works for Intel Arc, where no
    vendor CLI may be installed. The largest BAR tells whether Resizable BAR is
    on: about VRAM-sized when on, 256 MiB (or less) when off. Inside a VM the
    link can read as a placeholder (e.g. 2.5 GT/s x1), so it is reported as
    read, not trusted. Never fails; returns [] if nothing is found.
    """
    gpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = Path(sysfs)
    if not root.is_dir():
        return gpus
    for card in sorted(root.glob("card[0-9]*")):
        dev = card / "device"
        vendor = _read(dev / "vendor")
        if vendor not in _PCI_VENDORS:
            continue
        addr = dev.resolve().name
        if addr in seen:
            continue
        seen.add(addr)
        g: dict[str, Any] = {"pci_address": addr, "vendor": _PCI_VENDORS[vendor],
                             "pci_device_id": _read(dev / "device")}
        drv = dev / "driver"
        if drv.exists():
            g["kernel_driver"] = drv.resolve().name
        # Cards with their own PCIe switch (Intel Arc, some AMD) show an internal
        # link on the GPU function, so read the link at the topmost bridge of the
        # GPU's own vendor: the switch's upstream port, i.e. the link into the slot.
        port = dev.resolve()
        while (_read(port.parent / "class") or "").startswith("0x0604") and _read(port.parent / "vendor") == vendor:
            port = port.parent
        if port.name != addr:
            g["pcie_link_port"] = port.name
        for key in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
            v = _read(port / key)
            if v is not None:
                g[key] = v
        bars = []
        for line in (_read(dev / "resource") or "").splitlines():
            try:
                start, end, _flags = (int(x, 16) for x in line.split())
            except ValueError:
                continue
            if end > start:
                bars.append((end - start + 1) >> 20)
        if bars:
            g["largest_bar_mib"] = max(bars)
            g["resizable_bar"] = "likely on" if max(bars) > 256 else "likely off"
        gpus.append(g)
    return gpus


def gpu_info() -> dict[str, Any]:
    """Try AMD, then NVIDIA, then Intel, plus sysfs PCIe/BAR facts for every
    GPU. Never fails; returns {} if nothing found."""
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
    elif shutil.which("xpu-smi"):
        q = _run(["xpu-smi", "discovery", "-j"])
        if q:
            info["vendor"] = "intel"
            info["xpu_smi_discovery"] = q[:4000]
    pci = pci_gpus()
    if pci:
        info["pci"] = pci
        if "vendor" not in info and len({g["vendor"] for g in pci}) == 1:
            info["vendor"] = pci[0]["vendor"]
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
