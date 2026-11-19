# MoE benchmark 使用说明

从仓库根目录运行。在服务器实跑前，先用同一份本地 BF16 权重分别确认
native、experts、full 能生成短回复。以下命令需要安装本项目和匹配的
vLLM 0.20.0；仅自定义算子微基准不依赖 vLLM。采样和服务模式都不会下载模型。

## 1. 算子微基准

    python -m benchmarks.bench_moe --backend both --boundary experts
    python -m benchmarks.bench_moe --backend both --boundary routed-moe --output results/routed_moe.json

默认尺寸是 Qwen3.5-35B-A3B 的 routed experts：E=256、R=8、H=2048、I=512。
experts 边界两边使用同一份 vLLM IDs/weights；routed-moe 两边各自从
同一份 logits 选 expert。block 另外包含 Router Linear，不含 shared expert，
路由分布由矩阵乘法产生。两边的输出超出容差时不计时。
默认分别测均衡和集中分布，并报告自定义 BM=16 的 padding。BF16 模式默认输入 BF16 logits；可用 --logits-dtype float32 对照。
默认每组按 M 分配空间；可加 --workspace-max-tokens 2048 模拟服务容量。
原生基线调用 vLLM 的 functional Triton API，不等同于完整服务 runner。

CUDA events 报告每调用的微秒数；eager 可能包含 Python 提交不及时造成的
GPU 空闲时间。--mode graph 可单独比较重复重放的 kernel 路径，不代表
服务插件已支持 CUDA Graph。--profile-dir results/profile 单独抓第一组，
可在 Perfetto 等工具中查看各 kernel；profiler 不参与计时。

## 2. 真实单层采样与回放

    python -m benchmarks.capture_layer --model /path/to/Qwen3.5-35B-A3B --layer 0 --trace results/layer0.pt
    python -m benchmarks.replay_layer --trace results/layer0.pt

capture 在 native 的单请求、无 prefix caching 场景中采集一次不分块 prefill
及少量 M=1 的 decode，记录原生路由和 routed 输出。trace 保存一份真实层权重，
目标尺寸约 1.5 GiB；不要把权重和 trace 提交到 GitHub。
replay 分别测试固定路由和从 logits 开始的两套公平基线，用 captured native
输出做验收。Top-k 先按 expert ID 对齐，再比较权重；仅排列顺序不同不算选错。
超容差的路径保留误差记录，不参与计时。两步都排除 Router Linear、shared expert
和最终合并。可为回放和微基准指定同一套容差，以便结果可比。

## 3. 整模型服务压测

    python -m benchmarks.bench_serve --model /path/to/Qwen3.5-35B-A3B --plan
    python -m benchmarks.bench_serve --model /path/to/Qwen3.5-35B-A3B --output-dir results/serve_first

--plan 只打印命令，无需模型、vLLM 或 GPU。实跑依次启动三种模式，
仅绑定本地地址，由当前 vLLM 环境的官方 bench serve 发送请求。
默认 3 轮，轮换模式顺序；对每种模式测输入 128/4096、输出 128、
并发 1/8，共 36 组。以 eager + Triton、禁用 prefix caching 的同配置
native 作为基线。默认 request-rate=inf 用于饱和测试；如需模拟业务负载，
用 --request-rate 设置固定请求速率。每组保留原始 JSON、客户端日志和服务日志，
summary.json 汇总成功率相关记录、吞吐和延迟；某组失败会保留结果并停止。配对比较中的 candidate_over_baseline 是候选值除以基线值，吞吐比值大于 1 和延迟比值小于 1 表示改善。
首次试跑可减到 --rounds 1 --input-lengths 128 --concurrency 1 --num-prompts 8 --warmup-requests 2。

对比时分别看 native → experts、experts → full。P50/P95 汇总是各轮
分位数的中位数，不是合并所有请求后重新计算的分位数。正式性能结论使用
GPU 实测数据，不用无卡检查或本机小尺寸 smoke 的结果。
