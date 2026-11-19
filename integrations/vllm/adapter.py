"""单卡 vLLM 0.20.0 路由专家适配；不修改 vLLM 安装目录。"""

import logging
from functools import wraps
from importlib.metadata import version
import inspect
import os

import torch

from moe_kernel import MoEConfig, MoEWorkspace, experts, moe


_LOG = logging.getLogger("vllm.moe_kernel")
_MODES = ("native", "experts", "full")
_EXPECTED_ARGS = ("self", "layer", "hidden_states", "router_logits", "shared_experts_input", "input_ids")


def check_supported_configuration(runner, layer, hidden_states, router_logits, *,
                                  method_type, triton_backend, router_type):
    """运行前检查布局和语义。显式启用自定义模式时，遇到不支持的路径立即报错。"""
    method = runner.quant_method
    config = layer.moe_config
    parallel = config.moe_parallel_config
    if (parallel.tp_size, parallel.dp_size, parallel.ep_size, parallel.pcp_size) != (1, 1, 1, 1):
        raise ValueError("moe-kernel only supports single-GPU MoE")
    if (parallel.use_ep or parallel.enable_eplb or layer.is_sequence_parallel or runner.enable_dbo
            or layer.expert_map is not None):
        raise ValueError("moe-kernel does not support EP, EPLB, sequence parallel or DBO")
    if type(method) is not method_type or method.is_monolithic or method.unquantized_backend != triton_backend:
        raise ValueError("moe-kernel requires unquantized MoE with --moe-backend triton")
    if (config.has_bias or not config.is_act_and_mul or layer.activation.value != "silu"
            or layer.apply_router_weight_on_input):
        raise ValueError("moe-kernel requires bias-free SwiGLU and output-side routing weights")
    if (type(runner.router) is not router_type or layer.use_grouped_topk
            or layer.custom_routing_function is not None or layer.scoring_func != "softmax"):
        raise ValueError("moe-kernel requires the standard softmax Top-k router")
    w13, w2 = layer.w13_weight, layer.w2_weight
    e, h, n = w2.shape
    m = hidden_states.shape[0]
    if (e != layer.global_num_experts or w13.shape != (e, 2 * n, h)
            or hidden_states.shape != (m, h) or router_logits.shape != (m, e)):
        raise ValueError("moe-kernel requires unpadded [M,H], [M,E], [E,2N,H], [E,H,N] layouts")
    if (not hidden_states.is_cuda or hidden_states.dtype not in (torch.bfloat16, torch.float16)
            or not all(t.device == hidden_states.device and t.is_contiguous()
                       for t in (hidden_states, w13, w2, router_logits))
            or w13.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype
            or router_logits.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("moe-kernel requires contiguous CUDA activations, weights and router logits")
    return MoEConfig(hidden_dim=h, intermediate_dim=n, num_experts=e, top_k=layer.top_k,
                     renormalize=layer.renormalize)


def _run_routed(runner, layer, hidden_states, router_logits, shared_input, input_ids, *,
                mode, workspace, config, order):
    # 复用原生 shared expert 运行时序；绕过 modular kernel 时补上它负责的阶段。
    runner._maybe_apply_shared_experts(shared_input, order.NO_OVERLAP)
    if mode == "experts":
        weights, ids = runner.router.select_experts(
            hidden_states=hidden_states, router_logits=router_logits, input_ids=input_ids,
        )
        routed = experts(hidden_states, layer.w13_weight, layer.w2_weight, weights, ids,
                         workspace=workspace)
    else:
        routed = moe(hidden_states, router_logits, layer.w13_weight, layer.w2_weight,
                     top_k=config.top_k, renormalize=config.renormalize, workspace=workspace)
    runner._maybe_apply_shared_experts(shared_input, order.MK_INTERNAL_OVERLAPPED)
    runner._maybe_apply_shared_experts(shared_input, order.MULTI_STREAM_OVERLAPPED)
    shared = runner.shared_experts
    return (shared.output if shared is not None else None), routed


def _install(runner_type, *, mode, max_tokens, method_type, triton_backend, router_type, order,
             get_config):
    original = runner_type._apply_quant_method
    installed = getattr(original, "_moe_kernel_options", None)
    if installed == (mode, max_tokens):
        return
    if installed is not None:
        raise RuntimeError("moe-kernel mode cannot change inside an existing worker")
    if tuple(inspect.signature(original).parameters) != _EXPECTED_ARGS:
        raise RuntimeError("Unexpected vLLM 0.20.0 MoERunner._apply_quant_method signature")

    # 每个 worker 按配置和 CUDA stream 共用 scratch，避免给每一层重复分配。
    workspaces = {}
    original_init = runner_type.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        cfg = get_config()
        model = cfg.model_config
        if (model is None or model.hf_config.model_type != "qwen3_5_moe"
                or not model.enforce_eager or model.enable_return_routed_experts
                or cfg.lora_config is not None or cfg.speculative_config is not None
                or cfg.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("moe-kernel requires eager Qwen3.5 MoE, without LoRA, MTP, PP or routed-expert tracing")
        original_init(self, *args, **kwargs)

    @wraps(original)
    def apply(self, layer, hidden_states, router_logits, shared_experts_input, input_ids=None):
        config = check_supported_configuration(
            self, layer, hidden_states, router_logits, method_type=method_type,
            triton_backend=triton_backend, router_type=router_type,
        )
        if hidden_states.shape[0] > max_tokens:
            raise ValueError("MoE batch exceeds MOE_KERNEL_MAX_TOKENS; set it at least as high as the batch token budget")
        stream = torch.cuda.current_stream(hidden_states.device)
        key = (stream, hidden_states.dtype, config)
        workspace = workspaces.get(key)
        if workspace is None:
            workspace = MoEWorkspace()
            workspace.prepare(max_tokens=max_tokens, config=config, device=hidden_states.device,
                              dtype=hidden_states.dtype)
            workspaces[key] = workspace
            _LOG.info("moe-kernel active: mode=%s E=%s R=%s H=%s N=%s max_tokens=%s",
                      mode, config.num_experts, config.top_k, config.hidden_dim,
                      config.intermediate_dim, max_tokens)
        return _run_routed(
            self, layer, hidden_states, router_logits, shared_experts_input, input_ids,
            mode=mode, workspace=workspace, config=config, order=order,
        )

    apply._moe_kernel_options = (mode, max_tokens)
    runner_type.__init__ = init
    runner_type._apply_quant_method = apply


def register():
    """vLLM general-plugin 入口，每个 worker 启动时调用。默认 native，不打补丁。"""
    mode = os.getenv("MOE_KERNEL_MODE", "native")
    if mode not in _MODES:
        raise ValueError(f"MOE_KERNEL_MODE must be one of {_MODES}, got {mode!r}")
    if mode == "native":
        return
    installed = version("vllm")
    if installed.split("+", 1)[0] != "0.20.0":
        raise RuntimeError(f"moe-kernel requires vLLM 0.20.0, found {installed}")
    try:
        max_tokens = int(os.getenv("MOE_KERNEL_MAX_TOKENS", "2048"))
        if max_tokens <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError("MOE_KERNEL_MAX_TOKENS must be a positive integer") from exc

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    _install(MoERunner, mode=mode, max_tokens=max_tokens, method_type=UnquantizedFusedMoEMethod,
             triton_backend=UnquantizedMoeBackend.TRITON, router_type=FusedTopKRouter,
             order=SharedExpertsOrder, get_config=get_current_vllm_config)
