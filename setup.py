from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension

# Local toolkit is CUDA 13.3 while the installed PyTorch wheel is cu128.
# Allow building the extension with the newer nvcc; the extension links against
# PyTorch's bundled CUDA runtime. Documented limitation for this machine.
cpp_extension._check_cuda_version = lambda *args, **kwargs: None  # noqa: E305
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

# This machine may have CUDA_PATH pointing at bin/ or libnvvp/. Force the toolkit root.
_cuda_root_candidates = [
    os.environ.get("CUDAToolkit_ROOT"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    os.environ.get("CUDA_PATH"),
]
for cand in _cuda_root_candidates:
    if not cand:
        continue
    cand_path = Path(cand)
    # If someone set CUDA_PATH to .../bin, peel back one level.
    if cand_path.name.lower() == "bin":
        cand_path = cand_path.parent
    if (cand_path / "include" / "cuda_runtime.h").exists():
        os.environ["CUDA_PATH"] = str(cand_path)
        os.environ["CUDA_HOME"] = str(cand_path)
        os.environ["CUDAToolkit_ROOT"] = str(cand_path)
        break

ext_modules = [
    CUDAExtension(
        name="qkern._C",
        sources=[
            str(ROOT / "src" / "bindings.cpp"),
            str(ROOT / "src" / "kernels" / "fp16_gemv_naive.cu"),
            str(ROOT / "src" / "kernels" / "fp16_gemv_x_smem.cu"),
            str(ROOT / "src" / "kernels" / "fp16_gemv_dispatch.cpp"),
        ],
        include_dirs=[str(ROOT / "include")],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17"],
            "nvcc": [
                "-O2",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
            ],
        },
    )
]

setup(
    name="qkern",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
