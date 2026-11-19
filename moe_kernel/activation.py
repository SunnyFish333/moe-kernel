"""步骤 4：逐路由 SwiGLU。"""

import torch
import triton
import triton.language as tl

from ._validation import ACTIVATION_DTYPES, check_tensor


@triton.jit
def _silu_and_mul_kernel(gate_up, out, numel, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    q, j = i // N, i % N
    addr = gate_up + q * (2 * N) + j
    gate = tl.load(addr, i < numel, other=0).to(tl.float32)
    up = tl.load(addr + N, i < numel, other=0).to(tl.float32)
    tl.store(out + i, (gate * tl.sigmoid(gate)) * up, i < numel)


def silu_and_mul(gate_up: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """[M*R,2N] → [M*R,N]；FP32 计算 silu(gate)*up 后转换 dtype。"""
    check_tensor(gate_up, "gate_up", ndim=2, dtypes=ACTIVATION_DTYPES)
    check_tensor(out, "out", ndim=2, dtypes=(gate_up.dtype,), device=gate_up.device)
    rows, cols = gate_up.shape
    if not cols or cols % 2 or out.shape != (rows, cols // 2):
        raise ValueError("gate_up 和 out 必须分别为 [Q,2N] 和 [Q,N]，N > 0")
    if out.numel():
        block = 256
        with torch.cuda.device(gate_up.device):
            _silu_and_mul_kernel[(triton.cdiv(out.numel(), block),)](
                gate_up, out, out.numel(), cols // 2, block,
            )
    return out
