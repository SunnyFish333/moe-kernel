"""离线运行原生 vLLM，只采样一次 prefill 和少量 decode；不计时。"""

import argparse
from functools import wraps
from hashlib import sha256
import os
from pathlib import Path
import re
from unittest.mock import patch

from .common import require_vllm


def _cpu(tensor):
    return tensor.detach().to("cpu", copy=True).contiguous()


def _install_capture(model, *, layer_index, prompt_tokens, decode_samples, model_label, config_hash,
                     batch_size=1):
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from integrations.vllm.adapter import check_supported_configuration

    pattern = re.compile(r"(?:^|\.)layers\." + str(layer_index) + r"\.mlp\.experts$")
    matches = [(name, layer) for name, layer in model.named_modules()
               if pattern.search(name) and hasattr(layer, "w13_weight")]
    if len(matches) != 1:
        raise RuntimeError(f"需要唯一的 Qwen3.5 MoE 层，匹配结果：{[name for name, _ in matches]}")
    layer_name, layer = matches[0]
    runner = layer.runner
    original = runner._apply_quant_method
    trace = dict(schema=1, model=model_label, config_sha256=config_hash, vllm=require_vllm(),
                 layer=layer_name, renormalize=layer.renormalize,
                 w13=_cpu(layer.w13_weight), w2=_cpu(layer.w2_weight), samples=[])
    runner._moe_bench_capture = (original, trace)

    @wraps(original)
    def capture(layer, hidden_states, router_logits, shared_experts_input, input_ids=None):
        samples = trace["samples"]
        if len(samples) >= 1 + decode_samples:
            return original(layer, hidden_states, router_logits, shared_experts_input, input_ids)
        check_supported_configuration(
            runner, layer, hidden_states, router_logits, method_type=UnquantizedFusedMoEMethod,
            triton_backend=UnquantizedMoeBackend.TRITON, router_type=FusedTopKRouter,
        )
        expected_m = batch_size if samples else batch_size * prompt_tokens
        if hidden_states.shape[0] != expected_m:
            raise RuntimeError(f"采样要求整批不分块 prefill 和等长 decode：M={hidden_states.shape[0]}，预期 {expected_m}")
        # 原生实现可能就地写入 hidden_states，必须在调用前复制。
        record = dict(phase="decode" if samples else "prefill",
                      x=_cpu(hidden_states), logits=_cpu(router_logits))
        select = runner.router.select_experts

        def select_and_record(*args, **kwargs):
            weights, ids = select(*args, **kwargs)
            record.update(weights=_cpu(weights), ids=_cpu(ids))
            return weights, ids

        with patch.object(runner.router, "select_experts", select_and_record):
            shared_out, routed_out = original(
                layer, hidden_states, router_logits, shared_experts_input, input_ids,
            )
        if "ids" not in record:
            raise RuntimeError("原生路由没有命中标准 Top-k 路径")
        record["native_output"] = _cpu(routed_out)
        samples.append(record)
        return shared_out, routed_out

    runner._apply_quant_method = capture
    return layer_name


def _save_capture(model, *, layer_name, path, num_samples):
    import torch

    runner = dict(model.named_modules())[layer_name].runner
    original, trace = runner._moe_bench_capture
    runner._apply_quant_method = original
    del runner._moe_bench_capture
    if len(trace["samples"]) != num_samples:
        raise RuntimeError(f"只采集到 {len(trace['samples'])}/{num_samples} 个样本")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(trace, tmp)
    tmp.replace(target)
    return dict(layer=layer_name, path=path, samples=num_samples)


class MoECaptureWorker:
    # vLLM 只传方法名和普通参数，避免序列化 Python callable。
    def moe_bench_install_capture(self, **kwargs):
        return _install_capture(self.get_model(), **kwargs)

    def moe_bench_save_capture(self, **kwargs):
        return _save_capture(self.get_model(), **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="已经下载的 Qwen3.5 模型目录")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--decode-samples", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--prompt", default="Explain why grouped matrix multiplication helps inference. " * 16)
    args = p.parse_args()
    cfg = args.model / "config.json"
    if not cfg.is_file() or args.layer < 0 or min(args.decode_samples, args.batch_size) < 1:
        p.error("需要本地模型配置、非负 layer 和至少一个 decode 样本")
    os.environ.update(MOE_KERNEL_MODE="native", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_ENABLE_V1_MULTIPROCESSING="0")
    require_vllm()
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(args.model.resolve()), tensor_parallel_size=1, dtype="bfloat16",
              language_model_only=True, enforce_eager=True, moe_backend="triton",
              worker_extension_cls="benchmarks.capture_layer.MoECaptureWorker",
              enable_prefix_caching=False, max_num_seqs=args.batch_size,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    tokens = llm.get_tokenizer().encode(args.prompt, add_special_tokens=False)
    if (len(tokens) < 2 or len(tokens) * args.batch_size > args.max_num_batched_tokens
            or len(tokens) + args.decode_samples + 1 > args.max_model_len):
        p.error("prompt 必须在一次未分块 prefill 中处理，并为 decode 留出上下文")
    layer_name, = llm.collective_rpc("moe_bench_install_capture", kwargs=dict(
        layer_index=args.layer, prompt_tokens=len(tokens), decode_samples=args.decode_samples,
        batch_size=args.batch_size,
        model_label=args.model.resolve().name, config_hash=sha256(cfg.read_bytes()).hexdigest(),
    ))
    llm.enqueue([{"prompt_token_ids": list(tokens)} for _ in range(args.batch_size)],
                SamplingParams(temperature=0, max_tokens=args.decode_samples + 1, ignore_eos=True),
                use_tqdm=False)
    llm.wait_for_completion(use_tqdm=False)
    result, = llm.collective_rpc("moe_bench_save_capture", kwargs=dict(
        layer_name=layer_name, path=str(args.trace.resolve()), num_samples=1 + args.decode_samples,
    ))
    print(result)


if __name__ == "__main__":
    main()
