"""步骤 1：Router Linear 包装与 Triton softmax/top-k。"""

import torch
import triton
import triton.language as tl

from ._validation import check_tensor


# 每块的 token 数参考 vLLM v0.11.0 的 NVIDIA CUDA Top-k 快速模板。
# 按实际专家数精确匹配，不按补齐后的列数匹配。线程布局由 Triton 决定。
_BLOCK_M_BY_NUM_EXPERTS = {
    1: 128,
    2: 128,
    4: 128,
    8: 64,
    16: 32,
    32: 16,
    64: 8,
    128: 4,
    192: 4,
    256: 4,
    320: 4,
    384: 4,
    448: 4,
    512: 4,
    576: 4,
}


def router_linear(x: torch.Tensor, wrouter: torch.Tensor) -> torch.Tensor:
    """[M,K] 与 [E,K] 权重产生 [M,E] logits。"""
    return torch.nn.functional.linear(x, wrouter)


@triton.jit
def _fused_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_tokens,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    experts = tl.arange(0, BLOCK_E)
    valid_tokens = tokens < num_tokens
    logits = tl.load(
        logits_ptr + tokens[:, None] * E + experts[None, :],
        mask=valid_tokens[:, None] & (experts[None, :] < E),
        other=-float("inf"),
    ).to(tl.float32)

    # 先按原始 logits 选专家，避免指数下溢使不同分数变成并列。
    row_max = tl.max(logits, axis=1)
    # 将尾块无效行的最大值设为零，避免计算 -inf 减 -inf。
    row_max = tl.where(valid_tokens, row_max, 0.0)
    candidates = logits
    ranks = tl.arange(0, BLOCK_K)
    selected = tl.full((BLOCK_M, BLOCK_K), 0.0, tl.float32)
    output_offsets = tokens * TOP_K

    for rank in range(TOP_K):
        best_logit, expert = tl.max(
            candidates, axis=1, return_indices=True, return_indices_tie_break_left=True,
        )
        numerator = tl.exp(best_logit - row_max)
        selected = tl.where(ranks[None, :] == rank, numerator[:, None], selected)
        tl.store(ids_ptr + output_offsets + rank, expert, mask=valid_tokens)
        candidates = tl.where(experts[None, :] == expert[:, None], -float("inf"), candidates)

    # 重新归一化时，全行 softmax 的分母会抵消。
    if RENORMALIZE:
        denominator = tl.sum(selected, axis=1)
    else:
        denominator = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
    denominator = tl.where(valid_tokens, denominator, 1.0)

    tl.store(
        weights_ptr + output_offsets[:, None] + ranks[None, :],
        selected / denominator[:, None],
        mask=valid_tokens[:, None] & (ranks[None, :] < TOP_K),
    )


def fused_topk(
    logits: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在一个 kernel 中转换 logits 并路由，返回 FP32 权重和 INT32 ID。"""
    check_tensor(logits, "logits", ndim=2, dtypes=(torch.bfloat16, torch.float32))

    num_tokens, num_experts = logits.shape
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k 必须满足 1 <= top_k <= E")

    shape = (num_tokens, top_k)
    if out is None:
        weights = torch.empty(shape, dtype=torch.float32, device=logits.device)
        ids = torch.empty(shape, dtype=torch.int32, device=logits.device)
    else:
        weights, ids = out
        check_tensor(weights, "router weights", ndim=2, dtypes=(torch.float32,), device=logits.device)
        check_tensor(ids, "router ids", ndim=2, dtypes=(torch.int32,), device=logits.device)
        if weights.shape != shape or ids.shape != shape:
            raise ValueError("router 输出必须为 [M, top_k]")
    if num_tokens == 0:
        return weights, ids

    block_e = triton.next_power_of_2(num_experts)
    # 未命中模板时默认每块处理四行。专家维度很宽时保留单行，避免小批次变慢。
    fallback_block_m = 1 if block_e >= 1024 else 4
    block_m = _BLOCK_M_BY_NUM_EXPERTS.get(num_experts, fallback_block_m)

    with torch.cuda.device(logits.device):
        _fused_topk_kernel[(triton.cdiv(num_tokens, block_m),)](
            logits,
            weights,
            ids,
            num_tokens,
            E=num_experts,
            TOP_K=top_k,
            BLOCK_M=block_m,
            BLOCK_E=block_e,
            BLOCK_K=triton.next_power_of_2(top_k),
            RENORMALIZE=renormalize,
            num_warps=4,
        )
    return weights, ids
