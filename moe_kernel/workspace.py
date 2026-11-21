"""预分配 scratch；并发调用应使用独立 workspace 或明确同步。"""

from dataclasses import dataclass, field

import torch

from ._validation import ACTIVATION_DTYPES
from .config import MoEConfig


_EXPERT_BUFFERS = ("gate_up", "activation", "weighted_output", "topk_weights", "topk_ids")


def max_route_blocks(num_routes: int, num_experts: int, block_size: int) -> int:
    """任意路由分布下的行块数上界。"""
    nonempty = min(num_experts, num_routes)
    return nonempty + (num_routes - nonempty) // block_size


@dataclass
class MoEWorkspace:
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    _prefill_align: dict[int, 'MoEWorkspace'] = field(default_factory=dict, init=False, repr=False)
    _wide_k: bool = field(default=False, init=False, repr=False)
    _align_spec: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)

    def prepare(self, *, max_tokens: int, config: MoEConfig, device: torch.device,
                dtype: torch.dtype, block_m: int = 16) -> None:
        """在调用或 Graph capture 前准备索引、路由和专家中间张量。"""
        h, n = config.hidden_dim, config.intermediate_dim
        if min(h, n) <= 0 or dtype not in ACTIVATION_DTYPES:
            raise ValueError("H、N 必须为正数，dtype 必须是 BF16 或 FP16")
        if block_m < 16 or block_m & (block_m - 1):
            raise ValueError("完整 MoE 的 block_m 必须是至少为 16 的 2 的幂")
        self.prepare_align(max_tokens=max_tokens, top_k=config.top_k,
                           num_experts=config.num_experts, block_size=block_m, device=device)
        device = self.buffers["counts"].device
        # 仅在实测相同架构和矩阵尺寸上使用这组 tile。
        self._wide_k = (h, n, config.num_experts, config.top_k, dtype) == (
            2048, 512, 256, 8, torch.bfloat16,
        ) and torch.cuda.get_device_capability(device) == (12, 0)
        if self._wide_k and block_m == 16:
            for minimum, bm in ((96, 64), (1024, 128)):
                if max_tokens >= minimum:
                    scratch = MoEWorkspace()
                    scratch.prepare_align(max_tokens=max_tokens, top_k=config.top_k,
                                          num_experts=config.num_experts, block_size=bm, device=device)
                    self._prefill_align[bm] = scratch
        q = max_tokens * config.top_k
        self.buffers.update(
            gate_up=torch.empty((q, 2 * n), device=device, dtype=dtype),
            activation=torch.empty((q, n), device=device, dtype=dtype),
            weighted_output=torch.empty((q, h), device=device, dtype=dtype),
            topk_weights=torch.empty((max_tokens, config.top_k), device=device, dtype=torch.float32),
            topk_ids=torch.empty((max_tokens, config.top_k), device=device, dtype=torch.int32),
        )

    def prepare_align(self, *, max_tokens: int, top_k: int, num_experts: int,
                      block_size: int, device: torch.device) -> None:
        """只为对齐分配 INT32 张量，不准备 GEMM 所需的空间。"""
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("align workspace 必须在 CUDA 设备上")
        if max_tokens < 0 or num_experts <= 0 or block_size <= 0 or not 1 <= top_k <= num_experts:
            raise ValueError("要求 max_tokens >= 0、block_size > 0 且 1 <= top_k <= num_experts")

        max_routes = max_tokens * top_k
        num_blocks = max_route_blocks(max_routes, num_experts, block_size)
        capacity = num_blocks * block_size
        if max(num_experts, capacity) > torch.iinfo(torch.int32).max:
            raise ValueError("align 索引容量超出 INT32 范围")

        shapes = {"counts": num_experts, "cursors": num_experts, "offsets": num_experts + 1,
                  "sorted_route_ids": capacity, "expert_ids": num_blocks, "num_post_pad": 1}
        buffers = {name: torch.empty(size, dtype=torch.int32, device=device)
                   for name, size in shapes.items()}
        for name in _EXPERT_BUFFERS:
            self.buffers.pop(name, None)
        self.buffers.update(buffers)
        self._wide_k = False
        self._prefill_align.clear()
        self._align_spec = (max_tokens, top_k, num_experts, block_size)

    def _check_experts(self, x: torch.Tensor, *, num_experts: int, top_k: int,
                       intermediate_dim: int) -> int:
        if self._align_spec is None or "activation" not in self.buffers:
            raise ValueError("请先调用 workspace.prepare() 为完整 MoE 分配空间")
        max_tokens, r, e, bm = self._align_spec
        m, h = x.shape
        act = self.buffers["activation"]
        if (m > max_tokens or (top_k, num_experts) != (r, e)
                or h != self.buffers["weighted_output"].shape[1]
                or intermediate_dim != act.shape[1]
                or x.dtype != act.dtype or x.device != act.device):
            raise ValueError("输入的容量、模型配置、dtype 或设备与 workspace 不匹配")
        return bm
