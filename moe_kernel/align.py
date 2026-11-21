"""步骤 2：仅整理路由索引 q=t*R+r，不复制 hidden states。"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ._validation import check_tensor
from .workspace import MoEWorkspace, max_route_blocks


class AlignedRoutes(NamedTuple):
    sorted_route_ids: torch.Tensor  # [C]，padding sentinel=M*R
    expert_ids: torch.Tensor        # [C/BM]，每个 GEMM 行块的专家
    num_post_pad: torch.Tensor      # [1]，GPU 上的有效填充长度 P


@triton.jit
def _reset_kernel(counts, cursors, sorted_ids, expert_ids, pad, E: tl.constexpr,
                  C: tl.constexpr, NUM_BLOCKS: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    tl.store(counts + i, 0, i < E)
    tl.store(cursors + i, 0, i < E)
    tl.store(sorted_ids + i, pad, i < C)
    tl.store(expert_ids + i, 0, i < NUM_BLOCKS)


@triton.jit
def _count_kernel(ids, counts, num_routes, E: tl.constexpr, TILE: tl.constexpr):
    q = tl.program_id(0) * TILE + tl.arange(0, TILE)
    e = tl.load(ids + q, q < num_routes, other=-1)
    valid = (q < num_routes) & (e >= 0) & (e < E)
    tl.atomic_add(counts + e, 1, mask=valid, sem="relaxed")


@triton.jit
def _prefix_kernel(counts, offsets, num_post_pad, E: tl.constexpr,
                   BM: tl.constexpr, TILE: tl.constexpr):
    e = tl.arange(0, TILE)
    count = tl.load(counts + e, e < E, other=0)
    num_blocks = tl.cdiv(count, BM)
    inclusive = tl.cumsum(num_blocks, 0)
    # 先对块数做前缀和，再转成行偏移；空专家不占块。
    tl.store(offsets + e, (inclusive - num_blocks) * BM, e < E)
    p = tl.sum(num_blocks, 0) * BM
    tl.store(offsets + E, p)
    tl.store(num_post_pad, p)


@triton.jit
def _fill_experts_kernel(offsets, expert_ids, num_post_pad, E: tl.constexpr,
                         BM: tl.constexpr, TILE: tl.constexpr):
    block = tl.program_id(0)
    start = block * BM
    if start < tl.load(num_post_pad):
        e = tl.arange(0, TILE)
        first = tl.load(offsets + e, e < E, other=0)
        end = tl.load(offsets + e + 1, e < E, other=0)
        # 半开区间会自动跳过起止相同的空专家。
        owner = tl.max(tl.where((e < E) & (first <= start) & (start < end), e, -1), 0)
        tl.store(expert_ids + block, owner)


@triton.jit
def _scatter_kernel(ids, cursors, offsets, sorted_ids, num_routes,
                    E: tl.constexpr, TILE: tl.constexpr):
    q = tl.program_id(0) * TILE + tl.arange(0, TILE)
    e = tl.load(ids + q, q < num_routes, other=-1)
    valid = (q < num_routes) & (e >= 0) & (e < E)
    slot = tl.atomic_add(cursors + e, 1, mask=valid, sem="relaxed")
    start = tl.load(offsets + e, valid, other=0)
    tl.store(sorted_ids + start + slot, q, valid)


def align(topk_ids: torch.Tensor, *, num_experts: int, block_size: int,
          workspace: MoEWorkspace) -> AlignedRoutes:
    """按专家分组，返回复用的 GPU INT32 缓冲区；下次调用会覆盖它们。

    topk_ids 为连续的 CUDA 张量 [M, R]，专家 ID 必须在 [0, num_experts)。
    """
    check_tensor(topk_ids, "topk_ids", ndim=2, dtypes=(torch.int32, torch.int64))
    if workspace._align_spec is None:
        raise ValueError("请先调用 workspace.prepare_align() 或 workspace.prepare()")
    max_tokens, top_k, prepared_e, prepared_bm = workspace._align_spec
    if (topk_ids.shape[1], num_experts, block_size) != (top_k, prepared_e, prepared_bm):
        raise ValueError("top_k、num_experts 和 block_size 必须与 workspace 一致")
    if topk_ids.shape[0] > max_tokens:
        raise ValueError("token 数超过 workspace 的 max_tokens")

    buffers = workspace.buffers
    counts, cursors, offsets = (buffers[name] for name in ("counts", "cursors", "offsets"))
    sorted_ids, expert_ids, p = (buffers[name] for name in
                                 ("sorted_route_ids", "expert_ids", "num_post_pad"))
    if topk_ids.device != sorted_ids.device:
        raise ValueError("topk_ids 和 workspace 必须在同一个 CUDA 设备上")

    num_routes = topk_ids.numel()
    capacity = sorted_ids.numel()
    num_blocks = expert_ids.numel()
    route_tile = 256
    expert_tile = triton.next_power_of_2(num_experts)
    with torch.cuda.device(topk_ids.device):
        _reset_kernel[(triton.cdiv(max(capacity, num_experts), route_tile),)](
            counts, cursors, sorted_ids, expert_ids, num_routes,
            num_experts, capacity, num_blocks, route_tile,
        )
        if num_routes:
            route_grid = (triton.cdiv(num_routes, route_tile),)
            _count_kernel[route_grid](topk_ids, counts, num_routes, num_experts, route_tile)
        _prefix_kernel[(1,)](counts, offsets, p, num_experts, block_size, expert_tile)
        if num_routes:
            _fill_experts_kernel[(max_route_blocks(num_routes, num_experts, block_size),)](
                offsets, expert_ids, p, num_experts, block_size, expert_tile,
            )
            _scatter_kernel[route_grid](
                topk_ids, cursors, offsets, sorted_ids, num_routes, num_experts, route_tile,
            )
    return AlignedRoutes(sorted_ids, expert_ids, p)
