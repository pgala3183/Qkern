"""QKern: architecture-aware quantized CUDA kernels and autotuning."""

from qkern.metrics import (
    error_summary,
    max_abs_error,
    mean_abs_error,
    relative_error,
    rmse,
)
from qkern.quantization import (
    INT4_QMAX,
    INT4_QMIN,
    INT8_QMAX,
    INT8_QMIN,
    QuantizedWeights,
    dequantize,
    dequantize_int4,
    dequantize_int8,
    expand_scales,
    int4_representable_values,
    pack_int4,
    quantize_int4,
    quantize_int8,
    unpack_int4,
)
from qkern.reference import (
    gemv_from_dequant,
    int4_gemv_reference,
    int8_gemv_reference,
)
from qkern.reference import fp16_gemv as fp16_gemv_ref

__version__ = "0.6.0"

try:
    from qkern.cuda_ops import (
        fp16_gemv,
        fp16_gemv_naive,
        fp16_gemv_vec2,
        fp16_gemv_x_smem,
        int4_gemv_fused,
        int4_gemv_fused_from_qw,
        int4_gemv_unfused,
        int4_gemv_unfused_from_qw,
        int8_gemv_fused,
        int8_gemv_fused_from_qw,
        int8_gemv_unfused,
        int8_gemv_unfused_from_qw,
        is_cuda_extension_available,
    )
except ImportError:  # pragma: no cover - unbuilt extension

    def fp16_gemv(*_args, **_kwargs):  # type: ignore[misc]
        raise ImportError(
            "qkern CUDA extension is not built. "
            "Run: python setup.py build_ext --inplace"
        )

    fp16_gemv_naive = fp16_gemv  # type: ignore[assignment]
    fp16_gemv_x_smem = fp16_gemv  # type: ignore[assignment]
    fp16_gemv_vec2 = fp16_gemv  # type: ignore[assignment]
    int8_gemv_fused = fp16_gemv  # type: ignore[assignment]
    int8_gemv_unfused = fp16_gemv  # type: ignore[assignment]
    int8_gemv_fused_from_qw = fp16_gemv  # type: ignore[assignment]
    int8_gemv_unfused_from_qw = fp16_gemv  # type: ignore[assignment]
    int4_gemv_fused = fp16_gemv  # type: ignore[assignment]
    int4_gemv_unfused = fp16_gemv  # type: ignore[assignment]
    int4_gemv_fused_from_qw = fp16_gemv  # type: ignore[assignment]
    int4_gemv_unfused_from_qw = fp16_gemv  # type: ignore[assignment]

    def is_cuda_extension_available() -> bool:
        return False


__all__ = [
    "INT4_QMAX",
    "INT4_QMIN",
    "INT8_QMAX",
    "INT8_QMIN",
    "QuantizedWeights",
    "__version__",
    "dequantize",
    "dequantize_int4",
    "dequantize_int8",
    "error_summary",
    "expand_scales",
    "fp16_gemv",
    "fp16_gemv_naive",
    "fp16_gemv_ref",
    "fp16_gemv_vec2",
    "fp16_gemv_x_smem",
    "gemv_from_dequant",
    "int4_gemv_fused",
    "int4_gemv_fused_from_qw",
    "int4_gemv_reference",
    "int4_gemv_unfused",
    "int4_gemv_unfused_from_qw",
    "int4_representable_values",
    "int8_gemv_fused",
    "int8_gemv_fused_from_qw",
    "int8_gemv_reference",
    "int8_gemv_unfused",
    "int8_gemv_unfused_from_qw",
    "is_cuda_extension_available",
    "max_abs_error",
    "mean_abs_error",
    "pack_int4",
    "quantize_int4",
    "quantize_int8",
    "relative_error",
    "rmse",
    "unpack_int4",
]
