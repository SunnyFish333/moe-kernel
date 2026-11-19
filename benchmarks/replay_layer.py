"""用同一份捕获的真实权重和路由回放原生与自定义 MoE。"""

import argparse
from functools import partial
from pathlib import Path

from .common import compare_routes, environment, error_stats, measure, native_ops, route_stats, write_json


def validate_trace(trace):
    """校验 schema 和形状，防止错误文件传到 CUDA kernel。"""
    import torch

    if trace.get("schema") != 1 or not trace.get("samples"):
        raise ValueError("需要 schema=1 且含样本的 capture_layer trace")
    if not isinstance(trace.get("renormalize"), bool):
        raise ValueError("缺少 renormalize 配置")
    w13, w2 = trace["w13"], trace["w2"]
    if w2.ndim != 3:
        raise ValueError("w2 必须是 [E,H,N]")
    e, h, n = w2.shape
    if (min(e, h, n) < 1 or w13.shape != (e, 2 * n, h) or w13.dtype != w2.dtype
            or w2.dtype not in (torch.bfloat16, torch.float16)):
        raise ValueError("专家权重的形状或 dtype 不合法")
    first_ids = trace["samples"][0]["ids"]
    if first_ids.ndim != 2:
        raise ValueError("ids 必须是 [M,R]")
    r = first_ids.shape[1]
    if not 1 <= r <= e:
        raise ValueError("top-k 或专家数不合法")
    for sample in trace["samples"]:
        x, logits, weights, ids, output = (sample[key] for key in
                                           ("x", "logits", "weights", "ids", "native_output"))
        if x.ndim != 2:
            raise ValueError("x 必须是 [M,H]")
        m = x.shape[0]
        if (sample["phase"] not in ("prefill", "decode") or not m or x.shape != (m, h)
                or output.shape != x.shape or logits.shape != (m, e) or ids.shape != (m, r)
                or weights.shape != ids.shape or x.dtype != w2.dtype or output.dtype != x.dtype
                or logits.dtype not in (torch.bfloat16, torch.float32) or weights.dtype != torch.float32
                or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("样本形状或 dtype 不合法")
        if not ((ids >= 0) & (ids < e)).all().item():
            raise ValueError("trace 含越界 expert ID")
    return h, n, e, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--backend", choices=["all", "native", "experts", "full"], default="all")
    p.add_argument("--mode", choices=["eager", "graph"], default="eager")
    p.add_argument("--workspace-max-tokens", type=int)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--atol", type=float, default=0.01)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=Path("results/replay_layer.json"))
    args = p.parse_args()
    if min(args.warmup, args.repeats, args.inner) <= 0 or min(args.atol, args.rtol) < 0:
        p.error("预热和采样次数必须为正，容差必须非负")

    import torch
    from moe_kernel import MoEConfig, MoEWorkspace, experts, moe
    from moe_kernel.router import fused_topk

    if not torch.cuda.is_available():
        p.error("回放需要 CUDA")
    trace = torch.load(args.trace, weights_only=True, map_location="cpu")
    h, n, e, r = validate_trace(trace)
    config = MoEConfig(h, n, e, r, trace["renormalize"])
    max_m = max(sample["x"].shape[0] for sample in trace["samples"])
    if args.workspace_max_tokens is not None and args.workspace_max_tokens < max_m:
        p.error("workspace 容量不足")
    paths = ("native", "experts", "full") if args.backend == "all" else (args.backend,)
    if "native" in paths:
        native_route, native_experts = native_ops()
    device = torch.device("cuda", torch.cuda.current_device())
    report = dict(schema=1, benchmark="layer_replay", environment=environment(device),
                  source={key: trace[key] for key in ("model", "config_sha256", "layer", "vllm")},
                  settings={**vars(args), "trace": str(args.trace), "output": str(args.output)},
                  native_baseline="functional vLLM Triton vs captured runner output; routed branch only",
                  shape=dict(hidden=h, intermediate=n, experts=e, top_k=r), rows=[])
    failed = False
    with torch.inference_mode():
        w13, w2 = trace["w13"].to(device).contiguous(), trace["w2"].to(device).contiguous()
        ws = MoEWorkspace()
        if any(path != "native" for path in paths):
            ws.prepare(max_tokens=args.workspace_max_tokens or max_m, config=config, device=device,
                       dtype=w2.dtype)
        for idx, sample in enumerate(trace["samples"]):
            x, logits, weights, ids, expected = (
                sample[key].to(device).contiguous() for key in
                ("x", "logits", "weights", "ids", "native_output")
            )
            row = dict(sample=idx, phase=sample["phase"], tokens=x.shape[0], routing=route_stats(ids, e, 16),
                       output_comparison={})
            calls = {}
            for path in paths:
                if path == "native":
                    # 两个边界分别对照：固定路由，以及从同一份 logits 开始。
                    def native_full():
                        rw, ri = native_route(x, logits, r, config.renormalize)
                        return native_experts(x, w13, w2, rw, ri)
                    cases = {
                        "native_experts": partial(native_experts, x, w13, w2, weights, ids.int()),
                        "native_full": native_full,
                    }
                elif path == "experts":
                    cases = {"experts": partial(experts, x, w13, w2, weights, ids, workspace=ws)}
                else:
                    rw, ri = fused_topk(logits, r, config.renormalize)
                    row["topk_comparison"] = compare_routes(rw, ri, weights, ids)
                    cases = {"full": partial(moe, x, logits, w13, w2, top_k=r,
                                            renormalize=config.renormalize, workspace=ws)}
                for name, fn in cases.items():
                    metrics = error_stats(fn(), expected, atol=args.atol, rtol=args.rtol)
                    row["output_comparison"][name] = metrics
                    if metrics["close"]:
                        calls[name] = fn
                    else:
                        failed = True
            row["latency"] = measure(calls, device=device, warmup=args.warmup, repeats=args.repeats,
                                     inner=args.inner, mode=args.mode) if calls else {}
            for native_name, custom_name in (("native_experts", "experts"), ("native_full", "full")):
                latencies = row["latency"]
                if native_name in latencies and custom_name in latencies and latencies[custom_name]["median_us"] > 0:
                    row[f"speedup_{custom_name}"] = (latencies[native_name]["median_us"]
                                                    / latencies[custom_name]["median_us"])
            report["rows"].append(row)
            write_json(args.output, report)
            print(f"sample={idx} phase={sample['phase']} M={x.shape[0]}: "
                  f"{ {k: v['close'] for k, v in row['output_comparison'].items()} }", flush=True)
    if failed:
        raise SystemExit("部分路径输出超出容差，已写入报告；这些路径没有计时")


if __name__ == "__main__":
    main()
