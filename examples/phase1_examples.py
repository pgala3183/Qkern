"""Small numerical examples for Phase 1 reference quantization / GEMV."""

from __future__ import annotations

import torch

from qkern import (
    dequantize,
    error_summary,
    fp16_gemv_ref,
    pack_int4,
    quantize_int4,
    quantize_int8,
    unpack_int4,
)


def main() -> None:
    print("=== FP16 GEMV ===")
    W = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]], dtype=torch.float16)
    x = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float16)
    y = fp16_gemv_ref(W, x)
    print(f"W=\n{W.float()}")
    print(f"x={x.float().tolist()}")
    print(f"y={y.tolist()}")

    print("\n=== INT8 per-tensor ===")
    W8 = torch.tensor([[1.5, -3.0]], dtype=torch.float32)
    qw8 = quantize_int8(W8, granularity="tensor")
    W8_hat = dequantize(qw8)
    print(f"scale={float(qw8.scales):.8f}")
    print(f"q={qw8.qweight.tolist()}")
    print(f"W_hat={W8_hat.tolist()}")
    print(f"errors={error_summary(W8_hat, W8)}")

    print("\n=== INT4 pack/unpack ===")
    q = torch.tensor([[-8, 7]], dtype=torch.int8)
    packed = pack_int4(q)
    print(f"q={q.tolist()} packed_byte=0x{int(packed[0, 0]):02x} unpack={unpack_int4(packed, k=2).tolist()}")

    print("\n=== INT4 group-128 primary target ===")
    torch.manual_seed(0)
    W4 = torch.randn(4, 256)
    x4 = torch.randn(256)
    qw4 = quantize_int4(W4, granularity="group", group_size=128, pack=True)
    y4 = fp16_gemv_ref(dequantize(qw4, dtype=torch.float16), x4.half())
    y_fp = fp16_gemv_ref(W4.half(), x4.half())
    print(f"qweight_shape={tuple(qw4.qweight.shape)} scales_shape={tuple(qw4.scales.shape)}")
    print(f"GEMV errors vs FP16: {error_summary(y4, y_fp)}")


if __name__ == "__main__":
    main()
