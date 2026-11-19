"""已路由专家入口和从 logits 开始的完整 MoE 入口。"""

import torch

from ._validation import ACTIVATION_DTYPES, check_tensor
from .activation import silu_and_mul
from .align import align
from .gemm import indirect_gemm
from .reduce import sum_topk
from .router import fused_topk
from .workspace import MoEWorkspace


def _prepare_experts(x, w13, w2, top_k, workspace, out):
    check_tensor(x, "x", ndim=2, dtypes=ACTIVATION_DTYPES)
    check_tensor(w13, "w13", ndim=3, dtypes=(x.dtype,), device=x.device)
    check_tensor(w2, "w2", ndim=3, dtypes=(x.dtype,), device=x.device)
    e, h, n = w2.shape
    if x.shape[1] != h or w13.shape != (e, 2 * n, h):
        raise ValueError("x、w13 和 w2 必须分别为 [M,H]、[E,2N,H] 和 [E,H,N]")
    if not 1 <= top_k <= e:
        raise ValueError("top_k 必须满足 1 <= top_k <= E")
    block_m = workspace._check_experts(x, num_experts=e, top_k=top_k, intermediate_dim=n)
    if out is not None:
        check_tensor(out, "out", ndim=2, dtypes=(x.dtype,), device=x.device)
        if out.shape != x.shape:
            raise ValueError("out 的形状必须与 x 一致")
    return block_m


def _run_experts(x, w13, w2, topk_weights, topk_ids, workspace, block_m, out):
    if out is None:
        out = torch.empty_like(x)
    m, h = x.shape
    if m == 0:
        return out

    r = topk_ids.shape[1]
    q = m * r
    buffers = workspace.buffers
    gate_up = buffers["gate_up"][:q]
    activation = buffers["activation"][:q]
    weighted = buffers["weighted_output"][:q]
    routes = align(topk_ids, num_experts=w2.shape[0], block_size=block_m, workspace=workspace)
    indirect_gemm(x, w13, routes, input_row_divisor=r, num_routes=q, out=gate_up,
                  block_m=block_m)
    silu_and_mul(gate_up, out=activation)
    indirect_gemm(activation, w2, routes, input_row_divisor=1, num_routes=q,
                  out=weighted, topk_weights=topk_weights, block_m=block_m)
    return sum_topk(weighted.view(m, r, h), out=out)


def experts(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
            *, workspace: MoEWorkspace, out: torch.Tensor | None = None) -> torch.Tensor:
    """用已有路由算专家 FFN；ID 必须合法，传入的 out 必须与输入和 scratch 独立。"""
    check_tensor(topk_ids, "topk_ids", ndim=2, dtypes=(torch.int32, torch.int64), device=x.device)
    check_tensor(topk_weights, "topk_weights", ndim=2, dtypes=(torch.float32,), device=x.device)
    r = topk_ids.shape[1]
    block_m = _prepare_experts(x, w13, w2, r, workspace, out)
    if topk_ids.shape != (x.shape[0], r) or topk_weights.shape != topk_ids.shape:
        raise ValueError("路由张量必须为 [M, R]")
    return _run_experts(x, w13, w2, topk_weights, topk_ids, workspace, block_m, out)


def moe(x: torch.Tensor, logits: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
        *, top_k: int, renormalize: bool = True, workspace: MoEWorkspace,
        out: torch.Tensor | None = None) -> torch.Tensor:
    """自定义 top-k + experts；不包含 Router Linear。"""
    block_m = _prepare_experts(x, w13, w2, top_k, workspace, out)
    check_tensor(logits, "logits", ndim=2, dtypes=(torch.bfloat16, torch.float32), device=x.device)
    m = x.shape[0]
    if logits.shape != (m, w2.shape[0]):
        raise ValueError("logits 必须为 [M, E]")
    buffers = workspace.buffers
    weights, ids = fused_topk(
        logits, top_k, renormalize,
        out=(buffers["topk_weights"][:m], buffers["topk_ids"][:m]),
    )
    return _run_experts(x, w13, w2, weights, ids, workspace, block_m, out)
