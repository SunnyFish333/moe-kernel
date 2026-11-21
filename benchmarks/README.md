# MoE 性能评测

从仓库根目录运行命令。算子微基准使用随机权重，`--backend custom` 只需要项目的 CUDA 环境。原生算子对照、真实单层采集和服务压测使用 [vLLM 插件环境](../integrations/vllm/README.md)。采集和服务压测读取现有的本地 BF16 checkpoint。

## 算子微基准

```bash
python -m benchmarks.bench_moe --backend both --boundary experts
python -m benchmarks.bench_moe --backend both --boundary routed-moe --mode graph \
  --tokens 1 2 4 8 32 128 512 1024 2048 --distribution balanced skewed random \
  --workspace-max-tokens 2048 --warmup 12 --repeats 20 --inner 12 \
  --output results/routed_moe.json
```

默认形状为 Qwen3.5-35B-A3B 路由专家的 `E=256、R=8、H=2048、I=512`。

| `--boundary` | 计时范围 |
| --- | --- |
| `experts` | 两边使用同一份专家 ID 和路由权重，只计专家前向 |
| `routed-moe` | 两边从同一份 logits 各自执行 Top-K 和专家前向 |
| `block` | 计入共同的 Router Linear、Top-K 及专家前向，排除共享专家。路由由矩阵乘法产生，使用 `random` |

`experts` 和 `routed-moe` 默认覆盖均衡和集中路由。输出的 padding 指标统一以 BM=16 统计，实际 GEMM tile 随派发而定。BF16 激活默认配 BF16 logits，FP16 激活默认配 FP32 logits。可用 `--logits-dtype` 显式选择。

默认 workspace 容量等于本组 M。用 `--workspace-max-tokens` 模拟服务固定的批次预算。原生算子基线调用 vLLM 的 functional Triton 实现。双后端模式先比较输出，超容差时保存误差并退出计时。

CUDA Event 报告每次调用耗时（µs）。`eager` 数据可能包含 Python 提交造成的 GPU 空隙，`graph` 数据来自预热后的独立算子重放。指定 `--profile-dir results/profile` 可为首组另行导出 CUDA trace。

## 真实单层采集与回放

```bash
python -m benchmarks.capture_layer --model /path/to/Qwen3.5-35B-A3B \
  --layer 0 --batch-size 1 --decode-samples 2 --trace results/layer0.pt
python -m benchmarks.replay_layer --trace results/layer0.pt --mode graph \
  --workspace-max-tokens 2048 --warmup 12 --repeats 30 --inner 12 \
  --output results/replay_layer0.json
```

采集使用原生 eager 模式，关闭 prefix caching，记录目标层的一次 prefill 和指定数量的 decode 样本，以及真实权重、激活、logits、路由和 routed 输出。用 `--batch-size 4` 或 `8` 采集批量场景时，同一批请求使用相同提示词，总 prefill token 数须能在一次调度内处理。单层 trace 约 1.5 GiB，包含专家权重，保存在本地结果目录。

回放覆盖固定路由与从 logits 开始的两条路径。输出以采集的原生 routed 输出为参考，Top-K 权重按专家 ID 对齐后比较。回放排除 Router Linear、共享专家和最终输出合并。批量回放用于观察指定路由分布下的算子耗时。

## 整模型服务压测

```bash
python -m benchmarks.bench_serve --model /path/to/Qwen3.5-35B-A3B --plan

python -m benchmarks.bench_serve --model /path/to/Qwen3.5-35B-A3B \
  --rounds 3 --input-lengths 128 --output-length 16 --concurrency 8 \
  --num-prompts 32 --warmup-requests 8 --output-dir results/serve
```

`--plan` 只打印命令，可在没有模型、vLLM 或 GPU 的环境运行。实跑在 Linux 上依次启动三种模式，服务绑定到 `127.0.0.1`，由同一个 vLLM 环境的 `vllm bench serve` 发起请求。三种模式使用相同模型、随机种子和启动参数，关闭 prefix caching，并按轮次轮换模式顺序。

脚本默认覆盖输入 128/4096、输出 128、并发 1/8，共 36 组。`--request-rate inf` 测饱和负载，指定正数可模拟固定到达率。每组保存详细 JSON 和客户端日志，每次服务启动保存服务日志。某组失败时保留对应记录并结束运行。

| 指标 | 含义 |
| --- | --- |
| Output throughput | 每秒生成的输出 token 数 |
| TTFT | 请求发出至首个输出 token 的时间 |
| TPOT | 首 token 之后，每个输出 token 的平均时间 |
| ITL | 相邻输出 token 的间隔 |
| E2EL | 完成整个请求的时间 |

`summary.json` 汇总吞吐、延迟及成功率。P50/P95 汇总先计算各轮分位数，再取这些分位数的中位数。配对比较在同轮、同输入长度和同并发下计算候选与基线的比值。`native → experts` 衡量替换专家计算的效果，`experts → full` 衡量继续替换 Top-K 的效果。
