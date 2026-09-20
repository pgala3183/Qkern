"""Runtime environment diagnostics for QKern benchmarks and development."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _safe_torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "error": "torch not installed",
        }

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        info.update(
            {
                "device_index": idx,
                "device_name": torch.cuda.get_device_name(idx),
                "compute_capability": f"{major}.{minor}",
                "device_count": torch.cuda.device_count(),
            }
        )
    return info


def _nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        completed = subprocess.run(
            [nvcc, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    for line in completed.stdout.splitlines():
        if "release" in line.lower():
            return line.strip()
    return completed.stdout.strip() or None


def collect_diagnostics() -> dict[str, Any]:
    """Collect host/GPU/toolchain facts for benchmark metadata."""
    return {
        "qkern_version": __import__("qkern").__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvcc": _nvcc_version(),
        "torch": _safe_torch_info(),
    }


def main() -> None:
    payload = collect_diagnostics()
    print(json.dumps(payload, indent=2))
    torch_info = payload["torch"]
    if not torch_info.get("available"):
        print("\n[warn] PyTorch is not installed.", file=sys.stderr)
    elif not torch_info.get("cuda_is_available"):
        print(
            "\n[warn] PyTorch cannot see a CUDA device "
            f"(build={torch_info.get('version')}, torch_cuda={torch_info.get('torch_cuda_version')}). "
            "Native CUDA smoke binaries may still work; install a CUDA-enabled PyTorch wheel for Phase 1+.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
