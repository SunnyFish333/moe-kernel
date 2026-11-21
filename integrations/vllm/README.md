# vLLM 插件

插件接入 Qwen3.5 MoE 的路由专家计算，保留模型的 Router Linear、共享专家、输出合并和 vLLM 调度。项目通过 `vllm.general_plugins` 注册，安装后由 vLLM 在 worker 启动时加载。

## 支持范围

- 兼容 vLLM 0.20.0 的 MoE runner。安装包带构建后缀时按基础版本匹配。
- 单卡、未量化 BF16 或 FP16 专家、无 bias、SwiGLU、标准 Softmax Top-K，路由权重应用于专家输出。
- 使用 `--enforce-eager` 与 `--moe-backend triton`。自定义服务模式的支持范围限定为 eager。
- 自定义路径要求未填充的连续张量布局。`x` 为 `[M,H]`、logits 为 `[M,E]`、专家权重为 `[E,2I,H]` 和 `[E,H,I]`。logits 支持 BF16 或 FP32。
- 自定义模式遇到不支持的模型、量化、并行、路由或运行配置时会明确报错。LoRA、推测解码、并行专家和路由追踪均需使用适配范围之外的运行方式。

## 安装

在已有的兼容 vLLM 环境中，从项目根目录运行。

```bash
python -m pip install --no-deps --no-build-isolation .

python -c "from importlib.metadata import entry_points, version; print(version('vllm')); print([e.value for e in entry_points(group='vllm.general_plugins') if e.name == 'moe_kernel'])"
```

入口应显示 `integrations.vllm.adapter:register`。本地模型目录应含模型配置、tokenizer 文件及完整权重。

## 启动

```bash
MODEL=/path/to/Qwen3.5-35B-A3B

MOE_KERNEL_MODE=experts MOE_KERNEL_MAX_TOKENS=2048 \
  vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
    --served-model-name qwen-moe \
    --tensor-parallel-size 1 --language-model-only \
    --enforce-eager --moe-backend triton --dtype bfloat16 \
    --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048
```

| `MOE_KERNEL_MODE` | Top-K | 路由专家计算 |
| --- | --- | --- |
| `native`（默认） | vLLM | vLLM |
| `experts` | vLLM | Triton 自定义专家算子 |
| `full` | Triton 自定义 Top-K | Triton 自定义专家算子 |

切换模式时重启 worker。`MOE_KERNEL_MAX_TOKENS` 是单个 MoE 批次的 token 容量，默认 2048。将它设置为至少覆盖 `--max-num-batched-tokens`，确保 workspace 可容纳调度器提交的批次。它与单条请求的上下文上限 `--max-model-len` 分别配置。

首次执行自定义 MoE 时，worker 日志会输出 `moe-kernel active: mode=experts` 或 `mode=full`，以及专家数、Top-K、矩阵尺寸和 workspace 容量。可用以下请求检查服务。

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-moe","prompt":"Explain grouped GEMM in one sentence.","max_tokens":24,"temperature":0}'
```

对照三种模式时保持服务参数、模型和请求一致。自动化复现命令见 [评测指南](../../benchmarks/README.md)。
