"""步骤 6：沿 top-k 维度归约，路由权重已在 GEMM2 应用。"""

import torch
import triton
import triton.language as tl

from ._validation import ACTIVATION_DTYPES, check_tensor


@triton.jit
def _sum_topk_kernel(weighted, out, R: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(R):
        values = tl.load(weighted + (token * R + rank) * H + j, j < H, other=0)
        acc += values.to(tl.float32)
    tl.store(out + token * H + j, acc, j < H)


def sum_topk(weighted_output: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """[M,R,H] → [M,H]；FP32 累加，不再乘路由权重。"""
    check_tensor(weighted_output, "weighted_output", ndim=3, dtypes=ACTIVATION_DTYPES)
    check_tensor(out, "out", ndim=2, dtypes=(weighted_output.dtype,), device=weighted_output.device)
    m, r, h = weighted_output.shape
    if not r or not h or out.shape != (m, h):
        raise ValueError("归约输入必须为 [M,R,H]，输出必须为 [M,H]，且 R、H > 0")
    if m:
        block = 256
        with torch.cuda.device(weighted_output.device):
            _sum_topk_kernel[(m, triton.cdiv(h, block))](weighted_output, out, r, h, block)
    return out
