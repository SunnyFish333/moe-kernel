"""共享的计时、统计及 vLLM 0.20.0 原生函数适配。"""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from statistics import median


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def require_vllm():
    installed = package_version("vllm")
    if installed is None or installed.split("+", 1)[0] != "0.20.0":
        raise RuntimeError(f"原生基线需要 vLLM 0.20.0，当前：{installed}")
    return installed


def environment(device):
    import torch
    import triton

    props = torch.cuda.get_device_properties(device)
    return dict(utc=datetime.now(timezone.utc).isoformat(), gpu=props.name,
                gpu_memory_bytes=props.total_memory, capability=[props.major, props.minor],
                torch=str(torch.__version__), triton=triton.__version__,
                cuda=torch.version.cuda, vllm=package_version("vllm"))


def native_ops():
    require_vllm()
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    def route(x, logits, top_k, renormalize):
        weights, ids, _ = fused_topk(x, logits, top_k, renormalize)
        return weights, ids

    return route, fused_experts


def route_stats(ids, num_experts, block):
    import torch

    counts = torch.bincount(ids.reshape(-1).long(), minlength=num_experts)
    padded = int((((counts + block - 1) // block).sum() * block).item())
    q = ids.numel()
    return dict(routes=q, active_experts=int((counts > 0).sum().item()),
                max_routes_per_expert=int(counts.max().item()), padded_rows=padded,
                padding_ratio=padded / q if q else None, block_m=block)


def error_stats(actual, expected, *, atol, rtol):
    errors = (actual.float() - expected.float()).abs()
    ref = expected.float()
    rmse = errors.square().mean().sqrt().item()
    rms_ref = ref.square().mean().sqrt().item()
    maximum = errors.max().item()
    return dict(max_abs=maximum if maximum < float("inf") else None,
                nrmse=rmse / rms_ref if 0 < rms_ref < float("inf") and rmse < float("inf") else None,
                atol=atol, rtol=rtol,
                close=bool((errors <= atol + rtol * ref.abs()).all().item()))


def compare_routes(actual_weights, actual_ids, expected_weights, expected_ids, *, atol=1e-6, rtol=1e-4):
    """按 expert ID 对齐；行内仅排名顺序不同不算选择不匹配。"""
    a, pa = actual_ids.sort(dim=-1)
    b, pb = expected_ids.sort(dim=-1)
    same = (a == b).all(dim=-1)
    stats = dict(equal_id_sets=int(same.sum().item()), rows=a.shape[0],
                 ordered_id_equal=bool((actual_ids == expected_ids).all().item()))
    stats["matching_row_weights"] = (
        error_stats(actual_weights.gather(-1, pa)[same], expected_weights.gather(-1, pb)[same],
                    atol=atol, rtol=rtol) if same.any().item() else None
    )
    return stats


def measure(calls, *, device, mode, warmup, repeats, inner):
    """交替测量各路径；结果为一组事件间隔除以内循环次数。"""
    import torch

    runners = dict(calls)
    with torch.cuda.device(device):
        for fn in calls.values():
            for _ in range(warmup):
                fn()
        torch.cuda.synchronize(device)
        if mode == "graph":
            capture_stream = torch.cuda.Stream(device=device)
            for name, fn in calls.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    for _ in range(inner):
                        output = fn()
                # 保留 graph 和最后一个输出的存储生命周期。
                def replay(graph=graph, output=output):
                    graph.replay()
                    return output
                runners[name] = replay
                replay()
            torch.cuda.synchronize(device)

        samples = {name: [] for name in calls}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for repeat in range(repeats):
            for name in list(runners)[::1 if repeat % 2 == 0 else -1]:
                start.record()
                for _ in range(inner if mode == "eager" else 1):
                    runners[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0 / inner)

    return {name: dict(median_us=median(values), min_us=min(values), max_us=max(values),
                       samples_us=values) for name, values in samples.items()}


def export_profile(calls, device, directory):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, fn in calls.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function(f"moe_bench_{name}"):
                fn()
            torch.cuda.synchronize(device)
        profile.export_chrome_trace(str(directory / f"{name}.json"))
