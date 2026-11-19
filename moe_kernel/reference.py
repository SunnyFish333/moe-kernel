"""首先实现本文件，作为后续 Triton kernel 的数值标准。"""

import torch


def route_reference(logits: torch.Tensor, top_k: int, renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 softmax、top-k、可选归一化；返回 weights 和 INT32 ids。"""
    if logits.ndim != 2:
        raise ValueError("logits 必须是 [M, E]")

    if not 1 <= top_k <= logits.shape[1]:
        raise ValueError("top_k 必须满足 1 <= top_k <= E")

    # 直接按 logits 排序，避免概率下溢后不同分数都变成 0。
    # stable=True 让同分时编号小的专家优先，和 Triton 的约定一致。
    logits = logits.float()
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :top_k]
    if renormalize:
        weights = torch.softmax(logits.gather(-1, ids), dim=-1)
    else:
        weights = torch.softmax(logits, dim=-1).gather(-1, ids)
    return weights, ids.to(torch.int32)



def experts_reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """按专家分组执行批量 FFN，按原路由顺序写回并沿 Top-k 求和。"""
    num_tokens, hidden_dim = x.shape
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        return x.new_empty(x.shape)

    # q = token * top_k + rank。按专家排序后，同一专家的 q 连续存放。
    flat_expert_ids = topk_ids.reshape(-1).long()
    sorted_route_ids = flat_expert_ids.argsort()
    route_counts = torch.bincount(flat_expert_ids, minlength=w13.shape[0]).tolist()
    flat_weights = topk_weights.reshape(-1)
    x_fp32 = x.float()
    weighted = x.new_empty((num_tokens * top_k, hidden_dim))

    start = 0
    for expert_id, count in enumerate(route_counts):
        if count == 0:
            continue
        route_ids = sorted_route_ids[start : start + count]
        start += count
        expert_input = x_fp32.index_select(0, route_ids // top_k)

        # GEMM1 与 SwiGLU 各自按约定写回 BF16。
        gate_up = (expert_input @ w13[expert_id].float().T).to(x.dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = (
            torch.nn.functional.silu(gate.float()) * up.float()
        ).to(x.dtype)

        # GEMM2 先在 FP32 中加权，再转 BF16 并按原始 q 写回。
        down = activation.float() @ w2[expert_id].float().T
        weights = flat_weights.index_select(0, route_ids)
        weighted.index_copy_(0, route_ids, (down * weights[:, None]).to(x.dtype))

    return weighted.view(num_tokens, top_k, hidden_dim).float().sum(dim=1).to(x.dtype)


def moe_reference(x: torch.Tensor, logits: torch.Tensor, w13: torch.Tensor,
                  w2: torch.Tensor, *, top_k: int, renormalize: bool = True) -> torch.Tensor:
    """logits 起始的 routed MoE，不包含 Router Linear。"""
    topk_weights, topk_ids = route_reference(logits, top_k, renormalize=renormalize)
    return experts_reference(x, w13, w2, topk_weights, topk_ids)
