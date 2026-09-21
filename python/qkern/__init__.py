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
    fp16_gemv,
    gemv_from_dequant,
    int4_gemv_reference,
    int8_gemv_reference,
)

__version__ = "0.1.0"

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
    "gemv_from_dequant",
    "int4_gemv_reference",
    "int4_representable_values",
    "int8_gemv_reference",
    "max_abs_error",
    "mean_abs_error",
    "pack_int4",
    "quantize_int4",
    "quantize_int8",
    "relative_error",
    "rmse",
    "unpack_int4",
]
