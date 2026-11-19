"""Qwen3-30B-A3B 的目标配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    hidden_dim: int = 2048
    intermediate_dim: int = 768
    num_experts: int = 128
    top_k: int = 8
    renormalize: bool = True
