"""步骤 3/5：共享的间接寻址 Grouped GEMM。"""

import torch
import triton
import triton.language as tl

from ._validation import ACTIVATION_DTYPES, check_tensor
from .align import AlignedRoutes
from .workspace import max_route_blocks


@triton.jit
def _indirect_gemm_kernel(
    a, weights, sorted_ids, expert_ids, num_post_pad, route_weights, out,
    num_routes, OUT_DIM: tl.constexpr, REDUCE_DIM: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    ROW_DIVISOR: tl.constexpr, APPLY_WEIGHT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DIRECT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start = pid_m * BM
    # grid 使用当前路由数的上界；实际行数由 GPU 上的 P 决定。
    if DIRECT or start < tl.load(num_post_pad):
        e = tl.load(expert_ids + pid_m).to(tl.int64)
        if DIRECT:
            q = tl.where(tl.arange(0, BM) == 0, pid_m, num_routes).to(tl.int64)
            valid_expert = (e >= 0) & (e < NUM_EXPERTS)
        else:
            q = tl.load(sorted_ids + start + tl.arange(0, BM)).to(tl.int64)
            valid_expert = True
        valid_q = q < num_routes
        rows = q // ROW_DIVISOR
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)

        acc = tl.full((BM, BN), 0, tl.float32)
        expert_weights = weights + e * OUT_DIM * REDUCE_DIM
        for tile in range(tl.cdiv(REDUCE_DIM, BK)):
            kk = tile * BK + k
            a_tile = tl.load(
                a + rows[:, None] * REDUCE_DIM + kk[None, :],
                valid_q[:, None] & (kk[None, :] < REDUCE_DIM), other=0,
            )
            b_tile = tl.load(
                expert_weights + cols[None, :].to(tl.int64) * REDUCE_DIM + kk[:, None],
                valid_expert & (kk[:, None] < REDUCE_DIM) & (cols[None, :] < OUT_DIM), other=0,
            )
            acc = tl.dot(a_tile, b_tile, acc)

        if APPLY_WEIGHT:
            scales = tl.load(route_weights + q, valid_q, other=0)
            acc = acc * scales[:, None]
        # 按原始 q 写出；tl.store 将 FP32 转为输出张量的 dtype。
        tl.store(
            out + q[:, None] * OUT_DIM + cols[None, :], acc,
            valid_q[:, None] & (cols[None, :] < OUT_DIM),
        )


def indirect_gemm(a: torch.Tensor, weights: torch.Tensor, routes: AlignedRoutes,
                  *, input_row_divisor: int, num_routes: int, out: torch.Tensor,
                  topk_weights: torch.Tensor | None = None,
                  block_m: int = 16, block_n: int = 64, block_k: int = 32,
                  num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """GEMM1 读 q//R 行；GEMM2 读 q 行并可加权。两者均按 q 写回。

    routes 必须来自同一批输入的 align()，out 必须独立于输入和路由缓冲区。
    """
    check_tensor(a, "a", ndim=2, dtypes=ACTIVATION_DTYPES)
    check_tensor(weights, "weights", ndim=3, dtypes=(a.dtype,), device=a.device)
    check_tensor(out, "out", ndim=2, dtypes=(a.dtype,), device=a.device)
    num_experts, out_dim, reduce_dim = weights.shape
    if min(num_experts, out_dim, reduce_dim, input_row_divisor) <= 0 or num_routes < 0:
        raise ValueError("GEMM 维度或路由参数不合法")
    if (a.shape != (triton.cdiv(num_routes, input_row_divisor), reduce_dim)
            or out.shape != (num_routes, out_dim)):
        raise ValueError("a、weights 和 out 的形状与路由参数不匹配")
    for tile in (block_m, block_n, block_k):
        if tile < 16 or tile & (tile - 1):
            raise ValueError("GEMM 的 BM、BN、BK 必须是至少为 16 的 2 的幂")
    for tensor, name in zip(routes, ("sorted_route_ids", "expert_ids", "num_post_pad")):
        check_tensor(tensor, name, ndim=1, dtypes=(torch.int32,), device=a.device)
    if (routes.sorted_route_ids.numel() != routes.expert_ids.numel() * block_m
            or routes.sorted_route_ids.numel() < num_routes or routes.num_post_pad.numel() != 1):
        raise ValueError("routes 容量或对齐粒度与 GEMM 不匹配")
    if topk_weights is not None:
        check_tensor(topk_weights, "topk_weights", ndim=2,
                     dtypes=(torch.float32,), device=a.device)
        if topk_weights.numel() != num_routes:
            raise ValueError("topk_weights 的元素数必须等于 num_routes")
    if not num_routes:
        return out

    max_blocks = max_route_blocks(num_routes, num_experts, block_m)
    grid = (min(max_blocks, routes.expert_ids.numel()), triton.cdiv(out_dim, block_n))
    with torch.cuda.device(a.device):
        _indirect_gemm_kernel[grid](
            a, weights, *routes, topk_weights, out, num_routes,
            out_dim, reduce_dim, num_experts, input_row_divisor, topk_weights is not None,
            block_m, block_n, block_k, False, num_warps=num_warps, num_stages=num_stages,
        )
    return out

def direct_gemm(a, weights, topk_ids, *, input_row_divisor, out, topk_weights=None,
                block_n=64, block_k=128, num_warps=4, num_stages=3):
    """小批次直接用每条路由的 expert ID，不做分组。供已验证形状的 MoE 入口使用。"""
    num_routes = topk_ids.numel()
    num_experts, out_dim, reduce_dim = weights.shape
    grid = (num_routes, triton.cdiv(out_dim, block_n))
    with torch.cuda.device(a.device):
        _indirect_gemm_kernel[grid](
            a, weights, None, topk_ids, None, topk_weights, out, num_routes,
            out_dim, reduce_dim, num_experts, input_row_divisor, topk_weights is not None,
            16, block_n, block_k, True, num_warps=num_warps, num_stages=num_stages,
        )
