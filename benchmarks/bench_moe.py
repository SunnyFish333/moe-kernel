"""固定形状和输入的 routed MoE 微基准；从项目根运行 python -m benchmarks.bench_moe。"""

import argparse
from functools import partial
from pathlib import Path

from .common import (compare_routes, environment, error_stats, export_profile, measure,
                     native_ops, route_stats, write_json)


def make_logits(m, e, r, distribution, generator, device):
    import torch

    logits = torch.rand((m, e), device=device, generator=generator) * 0.125
    if distribution == "random":
        return logits
    if distribution not in ("balanced", "skewed"):
        raise ValueError(f"未知路由分布：{distribution}")
    rank = torch.arange(r, device=device)
    if distribution == "balanced":
        start = torch.arange(m, device=device)[:, None] * r
        ids = (start + rank) % e
    else:
        ids = rank.expand(m, r)
    scores = (r - rank).to(torch.float32).expand(m, r)
    return logits.scatter_(1, ids, scores)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["both", "custom", "vllm"], default="both")
    p.add_argument("--boundary", choices=["experts", "routed-moe", "block"], default="experts")
    p.add_argument("--mode", choices=["eager", "graph"], default="eager")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048])
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--logits-dtype", choices=["bfloat16", "float32"],
                   help="默认 BF16 模式用 BF16 logits，FP16 模式用 FP32")
    p.add_argument("--distribution", choices=["random", "balanced", "skewed"], nargs="+",
                   help="默认 experts/routed-moe 测 balanced 和 skewed；block 只测 random")
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--intermediate-dim", type=int, default=512)
    p.add_argument("--workspace-max-tokens", type=int,
                   help="默认按 M 预分配；设 2048 可测与服务相同的容量")
    p.add_argument("--renormalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--atol", type=float, default=0.01)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--profile-dir", type=Path, help="只为第一组单独导出 CUDA trace，不计入延迟")
    p.add_argument("--output", type=Path, default=Path("results/bench_moe.json"))
    return p


def main():
    p = parser()
    args = p.parse_args()
    if (min(*args.tokens, args.hidden_dim, args.intermediate_dim, args.warmup, args.repeats,
            args.inner) <= 0 or not 1 <= args.top_k <= args.num_experts or
            min(args.atol, args.rtol) < 0):
        p.error("形状、预热和采样次数必须有效，且 1 <= top-k <= num-experts")
    if args.workspace_max_tokens is not None and args.workspace_max_tokens < max(args.tokens):
        p.error("workspace 容量必须覆盖所有 M")
    distributions = args.distribution or (["random"] if args.boundary == "block" else ["balanced", "skewed"])
    if args.boundary == "block" and distributions != ["random"]:
        p.error("block 的路由由实际 Router Linear 产生，只接受 random")

    import torch
    from moe_kernel import MoEConfig, MoEWorkspace, experts, moe
    from moe_kernel.reference import route_reference
    from moe_kernel.router import fused_topk, router_linear

    if not torch.cuda.is_available():
        p.error("需要 CUDA；可先使用 --help 检查参数")
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = MoEConfig(args.hidden_dim, args.intermediate_dim, args.num_experts, args.top_k,
                    args.renormalize)
    has_custom = args.backend != "vllm"
    has_native = args.backend != "custom"
    if has_native:
        route_native, experts_native = native_ops()
    report = dict(schema=1, benchmark="routed_moe_microbenchmark", environment=environment(device),
                  settings={**vars(args), "output": str(args.output),
                            "profile_dir": str(args.profile_dir) if args.profile_dir else None,
                            "distribution": distributions},
                  timing="CUDA events; eager includes GPU idle gaps; graph is standalone kernel replay",
                  native_baseline="vLLM 0.20.0 functional Triton fused_experts; not the server runner",
                  rows=[])
    h, n, e, r = cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts, cfg.top_k
    gen = torch.Generator(device=device).manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    logits_dtype = getattr(torch, args.logits_dtype or
                           ("bfloat16" if dtype == torch.bfloat16 else "float32"))

    with torch.inference_mode():
        w13 = torch.empty((e, 2 * n, h), device=device, dtype=dtype).normal_(std=h**-0.5, generator=gen)
        w2 = torch.empty((e, h, n), device=device, dtype=dtype).normal_(std=n**-0.5, generator=gen)
        wr = (torch.empty((e, h), device=device, dtype=dtype).normal_(std=h**-0.5, generator=gen)
              if args.boundary == "block" else None)
        for m in args.tokens:
            x = torch.randn((m, h), device=device, dtype=dtype, generator=gen)
            ws = None
            capacity = args.workspace_max_tokens or m
            if has_custom:
                ws = MoEWorkspace()
                ws.prepare(max_tokens=capacity, config=cfg, device=device, dtype=dtype)
            def make_router_logits():
                values = router_linear(x, wr)
                # 两条路径共用同样的输入类型，转换也计入 block 耗时。
                return values.to(logits_dtype)

            for dist in distributions:
                logits = (make_router_logits() if wr is not None else
                          make_logits(m, e, r, dist, gen, device).to(logits_dtype))
                weights, ids = (route_native(x, logits, r, cfg.renormalize) if has_native
                                else route_reference(logits, r, cfg.renormalize))
                calls = {}
                if has_native:
                    def native_routed(logits):
                        scales, choices = route_native(x, logits, r, cfg.renormalize)
                        return experts_native(x, w13, w2, scales, choices)

                    if args.boundary == "experts":
                        calls["vllm"] = partial(experts_native, x, w13, w2, weights, ids)
                    elif args.boundary == "routed-moe":
                        calls["vllm"] = partial(native_routed, logits)
                    else:
                        calls["vllm"] = lambda: native_routed(make_router_logits())
                if has_custom:
                    def custom_routed(logits):
                        return moe(x, logits, w13, w2, top_k=r,
                                   renormalize=cfg.renormalize, workspace=ws)

                    if args.boundary == "experts":
                        calls["custom"] = partial(experts, x, w13, w2, weights, ids, workspace=ws)
                    elif args.boundary == "routed-moe":
                        calls["custom"] = partial(custom_routed, logits)
                    else:
                        calls["custom"] = lambda: custom_routed(make_router_logits())

                row = dict(tokens=m, distribution=dist, logits_dtype=str(logits.dtype), workspace_capacity=capacity if has_custom else None,
                           routing=route_stats(ids, e, 16))
                if has_custom and args.boundary != "experts":
                    route_weights, route_ids = fused_topk(logits, r, cfg.renormalize)
                    row["topk_comparison"] = compare_routes(route_weights, route_ids, weights, ids)
                    row["custom_routing"] = route_stats(route_ids, e, 16)
                if len(calls) == 2:
                    row["output_comparison"] = error_stats(calls["custom"](), calls["vllm"](),
                                                           atol=args.atol, rtol=args.rtol)
                    if not row["output_comparison"]["close"]:
                        report["rows"].append(row)
                        write_json(args.output, report)
                        raise SystemExit(f"输出超出容差，未计时：M={m}, distribution={dist}")
                row["latency"] = measure(calls, device=device, mode=args.mode, warmup=args.warmup,
                                         repeats=args.repeats, inner=args.inner)
                if len(calls) == 2 and row["latency"]["custom"]["median_us"] > 0:
                    row["speedup_vllm_over_custom"] = (row["latency"]["vllm"]["median_us"]
                                                       / row["latency"]["custom"]["median_us"])
                if args.profile_dir and not report["rows"]:
                    export_profile(calls, device, args.profile_dir)
                report["rows"].append(row)
                write_json(args.output, report)
                print(m, dist, {k: round(v["median_us"], 2) for k, v in row["latency"].items()}, flush=True)


if __name__ == "__main__":
    main()
