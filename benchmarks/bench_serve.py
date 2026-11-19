"""用 vLLM 0.20.0 官方客户端对比单卡 native / experts / full。"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from .common import require_vllm, write_json


MODES = ("native", "experts", "full")
METRICS = ("request_throughput", "output_throughput", "p50_ttft_ms", "p95_ttft_ms",
           "p50_tpot_ms", "p95_tpot_ms", "p50_itl_ms", "p95_itl_ms",
           "p50_e2el_ms", "p95_e2el_ms")


def make_plan(args, cli, directory):
    model = str(args.model.resolve())
    serve = [cli, "serve", model, "--served-model-name", "qwen-moe-bench",
             "--host", "127.0.0.1", "--port", str(args.port), "--tensor-parallel-size", "1",
             "--language-model-only", "--enforce-eager", "--moe-backend", "triton",
             "--dtype", "bfloat16", "--no-enable-prefix-caching",
             "--gpu-memory-utilization", str(args.gpu_memory_utilization),
             "--max-model-len", str(args.max_model_len),
             "--max-num-seqs", str(max(args.concurrency)),
             "--max-num-batched-tokens", str(args.batch_tokens)]
    plan = []
    for round_idx in range(args.rounds):
        for mode in MODES[round_idx % len(MODES):] + MODES[:round_idx % len(MODES)]:
            run_id = f"r{round_idx + 1:02d}_{mode}"
            workloads = []
            for input_len, concurrency in product(args.input_lengths, args.concurrency):
                name = f"{run_id}_i{input_len}_o{args.output_length}_c{concurrency}"
                file = directory / f"{name}.json"
                command = [cli, "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
                           "--base-url", f"http://127.0.0.1:{args.port}",
                           "--model", "qwen-moe-bench", "--tokenizer", model,
                           "--dataset-name", "random", "--random-input-len", str(input_len),
                           "--random-output-len", str(args.output_length), "--random-range-ratio", "0",
                           "--num-prompts", str(args.num_prompts), "--seed", str(args.seed),
                           "--num-warmups", str(args.warmup_requests), "--temperature", "0", "--ignore-eos",
                           "--max-concurrency", str(concurrency), "--request-rate", args.request_rate,
                           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
                           "--save-result", "--save-detailed", "--result-dir", str(directory),
                           "--result-filename", file.name]
                workloads.append(dict(input_tokens=input_len, output_tokens=args.output_length,
                                      concurrency=concurrency, file=str(file), command=command))
            plan.append(dict(round=round_idx + 1, mode=mode, command=serve,
                             env={"MOE_KERNEL_MODE": mode, "MOE_KERNEL_MAX_TOKENS": str(args.batch_tokens),
                                  "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                             log=str(directory / f"{run_id}.server.log"), workloads=workloads))
    return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="本地 BF16 模型目录")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--input-lengths", type=int, nargs="+", default=[128, 4096])
    p.add_argument("--output-length", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--warmup-requests", type=int, default=12)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--request-rate", default="inf", help="inf 测饱和吞吐，或正数 req/s")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--workload-timeout", type=int, default=3600)
    p.add_argument("--plan", action="store_true", help="只打印命令；不运行服务、GPU 或下载")
    return p


def wait_ready(process, url, timeout):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM 启动失败，退出码 {process.returncode}")
        try:
            with urlopen(url + "/health", timeout=2) as reply:
                if reply.status == 200:
                    return
        except (URLError, TimeoutError):
            sleep(1)
    raise TimeoutError("等待 vLLM /health 超时")


def smoke(url, timeout):
    query = dict(model="qwen-moe-bench", prompt="Explain what an expert is.", max_tokens=4,
                 temperature=0, ignore_eos=True)
    req = Request(url + "/v1/completions", data=json.dumps(query).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as reply:
        response = json.load(reply)
    if not response.get("choices"):
        raise RuntimeError("短请求没有返回模型输出")


@contextmanager
def server(run):
    with Path(run["log"]).open("wb") as output:
        process = subprocess.Popen(run["command"], env={**os.environ, **run["env"]},
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            # 进程组里还有 vLLM engine/worker，退出时一起终止。
            for signum in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    continue
            process.wait(timeout=5)


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("mode", "input_tokens", "output_tokens", "concurrency"))
        groups.setdefault(key, []).append(row)
    summary = []
    for (mode, in_len, out_len, concurrency), group in sorted(groups.items()):
        metrics = {}
        for metric in METRICS:
            samples = [row["metrics"][metric] for row in group if metric in row["metrics"]]
            if samples:
                metrics[metric] = {"median_across_rounds": median(samples), "min": min(samples),
                                   "max": max(samples)}
        summary.append(dict(mode=mode, input_tokens=in_len, output_tokens=out_len,
                            concurrency=concurrency, rounds=len(group),
                            completed=sum(row["completed"] for row in group), metrics=metrics))
    return summary


def compare_modes(rows):
    # 同一轮、同一 workload 的配对结果，部分完成的运行也不会混比。
    paired = {}
    for row in rows:
        if row["success_rate"] == 1:
            key = tuple(row[k] for k in ("round", "input_tokens", "output_tokens", "concurrency"))
            paired.setdefault(key, {})[row["mode"]] = row["metrics"]
    comparisons = []
    for key, modes in sorted(paired.items()):
        for before, after in (("native", "experts"), ("experts", "full"), ("native", "full")):
            if before not in modes or after not in modes:
                continue
            ratios = {metric: modes[after][metric] / modes[before][metric]
                      for metric in METRICS if metric in modes[after] and modes[before].get(metric, 0) > 0}
            comparisons.append(dict(zip(("round", "input_tokens", "output_tokens", "concurrency"), key),
                                    baseline=before, candidate=after, candidate_over_baseline=ratios))
    return comparisons


def main():
    p = parser()
    args = p.parse_args()
    try:
        rate = float(args.request_rate)
    except ValueError:
        p.error("request-rate 必须是 inf 或正数")
    if (not rate > 0 or not 0 < args.gpu_memory_utilization < 1 or not 1 <= args.port <= 65535
            or min(*args.input_lengths, *args.concurrency, args.output_length, args.rounds,
                   args.num_prompts, args.warmup_requests, args.batch_tokens, args.max_model_len,
                   args.startup_timeout, args.workload_timeout) <= 0
            or max(args.input_lengths) + args.output_length + 1 > args.max_model_len):
        p.error("并发、长度和计时参数不合法，或上下文不足")
    args.request_rate = "inf" if math.isinf(rate) else str(rate)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (args.output_dir or Path("results") / f"serve_{timestamp}").resolve()
    cli = str(Path(sys.executable).with_name("vllm" if os.name == "posix" else "vllm.exe"))
    plan = make_plan(args, cli, directory)
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if os.name != "posix" or not Path(cli).is_file() or not (args.model / "config.json").is_file():
        p.error("实跑需要 Linux、当前 Python 环境的 vllm CLI 和已下载的本地模型")
    vllm = require_vllm()
    import torch
    from .common import environment

    if not torch.cuda.is_available():
        p.error("实跑需要 CUDA")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", plan)
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urlopen(url + "/health", timeout=2):
            raise RuntimeError("端口已有服务；请换一个 --port")
    except URLError as exc:
        # HTTPError 说明端口已被占用，即便 /health 返回非 200。
        if getattr(exc, "code", None) is not None:
            raise RuntimeError("端口已有 HTTP 服务；请换一个 --port") from exc

    report = dict(schema=1, benchmark="vllm_serve", vllm=vllm,
                  environment=environment(torch.device("cuda", torch.cuda.current_device())),
                  utc=timestamp, model_config_sha256=sha256((args.model / "config.json").read_bytes()).hexdigest(),
                  rows=[], summary=[], comparisons=[])
    for run in plan:
        print(f"round={run['round']} mode={run['mode']}", flush=True)
        with server(run) as process:
            wait_ready(process, url, args.startup_timeout)
            smoke(url, min(args.startup_timeout, 300))
            hit = f"moe-kernel active: mode={run['mode']}"
            log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
            if run["mode"] != "native" and hit not in log:
                raise RuntimeError(f"已发请求但未在 {run['log']} 中确认自定义路径")
            if run["mode"] == "native" and "moe-kernel active: mode=" in log:
                raise RuntimeError("native 基线意外进入自定义路径")
            for workload in run["workloads"]:
                target = Path(workload["file"])
                with target.with_suffix(".client.log").open("wb") as handle:
                    subprocess.run(workload["command"], env={**os.environ, **run["env"]},
                                   stdout=handle, stderr=subprocess.STDOUT, check=True,
                                   timeout=args.workload_timeout)
                raw = json.loads(target.read_text(encoding="utf-8"))
                complete = raw["completed"]
                row = {key: workload[key] for key in
                       ("input_tokens", "output_tokens", "concurrency", "file")}
                row.update(round=run["round"], mode=run["mode"], completed=complete,
                           requested=args.num_prompts, success_rate=complete / args.num_prompts,
                           total_input_tokens=raw.get("total_input_tokens"),
                           total_output_tokens=raw.get("total_output_tokens"),
                           metrics={key: raw[key] for key in METRICS if raw.get(key) is not None})
                report["rows"].append(row)
                report["summary"] = summarize([r for r in report["rows"] if r["success_rate"] == 1])
                report["comparisons"] = compare_modes(report["rows"])
                write_json(directory / "summary.json", report)
                if complete != args.num_prompts or raw.get("failed", 0):
                    raise RuntimeError(f"有失败请求，见 {target}；该组不计入汇总")
    print(f"结果：{directory / 'summary.json'}")


if __name__ == "__main__":
    main()
