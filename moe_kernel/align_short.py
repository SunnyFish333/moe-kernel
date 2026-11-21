"""小批量 MoE 的单 launch 路由对齐。"""

import torch
import triton
import triton.language as tl

from .align import AlignedRoutes


@triton.jit
def _align_short_kernel(ids, sorted_ids, expert_ids, p, Q: tl.constexpr,
                        E: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_E: tl.constexpr,
                        MAX_BLOCKS: tl.constexpr):
    q = tl.arange(0, BLOCK_Q)
    raw = tl.load(ids + q, q < Q, other=-1)
    valid = (q < Q) & (raw >= 0) & (raw < E)
    e = tl.where(valid, raw, 0).to(tl.int32)
    count = tl.histogram(e, BLOCK_E, mask=valid)
    blocks = tl.cdiv(count, 16)
    inclusive = tl.cumsum(blocks)
    first = inclusive - blocks
    owner = tl.arange(0, BLOCK_E)

    # 按输入顺序算每条路由在同一专家内的槽位。
    slot = tl.sum(((e[:, None] == e[None, :]) & (q[:, None] > q[None, :]) &
                   valid[None, :]).to(tl.int32), axis=1)
    offset = tl.gather(first, e, axis=0) * 16
    tl.store(sorted_ids + offset + slot, q, valid)

    # 只填实际使用范围；padding 和有效路由不会写到同一位置。
    pad = tl.arange(0, 16)
    tail = first * 16 + count
    tl.store(sorted_ids + tail[:, None] + pad[None, :], Q,
             pad[None, :] < (blocks * 16 - count)[:, None])
    for i in tl.static_range(MAX_BLOCKS):
        tl.store(expert_ids + first + i, owner, i < blocks)
    tl.store(p, tl.sum(blocks) * 16)


def align_short(ids, *, num_experts, workspace):
    """返回内部对齐结果；P 之外的 workspace 不初始化。"""
    q = ids.numel()
    buffers = workspace.buffers
    routes = AlignedRoutes(buffers["sorted_route_ids"], buffers["expert_ids"], buffers["num_post_pad"])
    with torch.cuda.device(ids.device):
        _align_short_kernel[(1,)](ids, *routes, q, num_experts, triton.next_power_of_2(q),
                                  triton.next_power_of_2(num_experts), triton.cdiv(q, 16), num_warps=4)
    return routes
