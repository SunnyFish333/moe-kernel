# MoE Kernel

面向 Qwen3.5 的单卡 Triton MoE 推理算子。支持 BF16/FP16 专家前向，可从 logits 执行 Softmax Top-K，也可复用已有路由。项目提供 vLLM 插件，并针对小批次 decode 做了优化。

[快速开始](#快速开始) · [性能](#性能) · [vLLM 集成](#vllm-集成) · [实现](#实现) · [评测指南](benchmarks/README.md)

## 性能

在单张 **NVIDIA RTX PRO 6000 Blackwell** 上，使用 **Qwen3.5-35B-A3B BF16** 和 **vLLM 0.20.0**。

**真实单层 decode 回放** 使用相同的专家 ID、路由权重和模型层输入。CUDA Event 计时独立算子的 CUDA Graph 重放，单位为 µs。

| 批次 token | vLLM Triton | MoE Kernel | 耗时变化 |
| :---: | :---: | :---: | :---: |
| 1 | 22.99 | 15.03 | **−34.6%** |
| 4 | 55.48 | 23.71 | **−57.3%** |
| 8 | 99.01 | 25.13 | **−74.6%** |

回放覆盖第 0、20、39 层的 13 个样本。1,740 个 token 的 Top-K 专家集合全部匹配原生实现，routed 输出最大绝对误差为 0.0078125。实测的三个 prefill 批次（130、520、1040 token）在专家计算上与原生接近。

**整模型服务压测** 使用 eager 模式、128 输入 token、16 输出 token、并发 8。共 3 轮、9 组、288 个计量请求，下表为相对同轮 `native` 的变化中位数。

| 模式 | 输出吞吐 | P95 首 token 延迟 |
| :---: | :---: | :---: |
| `experts` | +2.33% | +11.23% |
| `full` | +0.70% | +5.01% |

该负载下，专家算子回放的收益只部分反映在服务吞吐上，P95 首 token 延迟上升。复现命令、对照边界和指标口径见 [评测指南](benchmarks/README.md)。

## 快速开始

在具备 NVIDIA CUDA GPU 的环境中准备 Python 3.10+、PyTorch 2.8+、与 PyTorch/CUDA 匹配的 Triton，以及 setuptools 68+。

```bash
git clone https://github.com/SunnyFish333/moe-kernel.git
cd moe-kernel
python -m pip install --no-deps --no-build-isolation .
```

下面用随机权重运行一次 routed MoE，适合先验证本地 CUDA 环境。

```python
import torch

from moe_kernel import MoEConfig, MoEWorkspace, moe

m, h, i, e, r = 8, 64, 32, 16, 4
device, dtype = torch.device("cuda"), torch.bfloat16
config = MoEConfig(hidden_dim=h, intermediate_dim=i, num_experts=e, top_k=r)

with torch.inference_mode():
    x = torch.randn(m, h, device=device, dtype=dtype)
    logits = torch.randn(m, e, device=device, dtype=dtype)
    w13 = torch.randn(e, 2 * i, h, device=device, dtype=dtype) / h**0.5
    w2 = torch.randn(e, h, i, device=device, dtype=dtype) / i**0.5

    workspace = MoEWorkspace()
    workspace.prepare(max_tokens=m, config=config, device=device, dtype=dtype)

    y = moe(x, logits, w13, w2, top_k=r, workspace=workspace)
    print(y.shape)  # torch.Size([8, 64])
```

- **从 logits 开始**　`moe(x, logits, w13, w2, top_k=R, workspace=workspace)` 执行 Top-K 和专家前向。logits 支持 BF16 或 FP32。
- **已有路由**　`experts(x, w13, w2, topk_weights, topk_ids, workspace=workspace)` 直接计算专家输出。路由权重为 FP32，专家 ID 为 INT32 或 INT64。
- **复用缓冲区**　两个接口都返回 `[M, H]`，可传入独立的 `out`。输入与权重必须连续、处于同一 CUDA 设备，并使用相同的激活与权重 dtype。workspace 可在同一 CUDA stream 上顺序复用。

输入激活为 `[M, H]`，logits 为 `[M, E]`，专家权重 `w13` 和 `w2` 分别为 `[E, 2I, H]`、`[E, H, I]`。`w13` 的输出通道先 gate 后 up。路由张量均为 `[M, R]`，其中 `E` 是专家数，`I` 是专家中间维度，`R` 是每个 token 选择的专家数。

## vLLM 集成

在兼容的 vLLM 0.20.0 环境内安装本项目。插件适用于单卡、未量化的 Qwen3.5 MoE。三种模式保留原生 Router Linear、共享专家、输出合并与调度。

| `MOE_KERNEL_MODE` | Softmax Top-K | 路由专家 |
| :---: | :---: | :---: |
| `native`（默认） | vLLM | vLLM |
| `experts` | vLLM | MoE Kernel |
| `full` | MoE Kernel | MoE Kernel |

```bash
MODEL=/path/to/Qwen3.5-35B-A3B
MOE_KERNEL_MODE=experts MOE_KERNEL_MAX_TOKENS=2048 \
  vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --language-model-only \
    --enforce-eager --moe-backend triton --dtype bfloat16 \
    --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048
```

`MOE_KERNEL_MAX_TOKENS` 须覆盖调度器的批次 token 预算。切换模式要重启服务。环境要求、支持范围和短请求验证见 [vLLM 插件指南](integrations/vllm/README.md)。

## 实现

![Qwen3.5-35B-A3B 文本解码器与 MoE，256 个路由专家中每 token 选择 8 个，同时经过 1 个共享专家](qwen3.5-35b-a3b-moe-architecture.svg)

图依据 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)和[模型配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)绘制。

| 组件 | 实现 |
| --- | --- |
| [Top-K](moe_kernel/router.py) | 在一次 launch 内将 logits 转为 FP32、选择专家、计算路由权重 |
| [路由对齐](moe_kernel/align.py) | 按专家重排路由索引并补齐到 GEMM 行块，保留原始 hidden states 布局 |
| [Grouped GEMM](moe_kernel/gemm.py) | 间接寻址读取输入，FP32 累加。第二次 GEMM 写回前乘路由权重 |
| [SwiGLU](moe_kernel/activation.py) 与[输出归约](moe_kernel/reduce.py) | 逐路由激活，再沿 Top-K 维归约到原 token 顺序 |
| [派发](moe_kernel/api.py)与 [workspace](moe_kernel/workspace.py) | 按批次形状选路径，复用索引与中间缓冲区 |

常规专家路径发射 9 次 kernel。单 token 且 Top-K ≤ 8 时直接寻址专家，发射 4 次。目标 BF16 形状在 Blackwell SM 12.0、默认 BM=16 时，2–8 个 token 用[单次对齐](moe_kernel/align_short.py)，专家路径共发射 5 次。独立算子支持 CUDA Graph 重放，vLLM 插件使用 eager 模式。

## 复现评测

| 层级 | 入口 | 适合回答的问题 |
| --- | --- | --- |
| 合成微基准 | [`bench_moe`](benchmarks/bench_moe.py) | 不同 token 数和路由分布下的 CUDA Event 延迟 |
| 真实单层回放 | [`capture_layer`](benchmarks/capture_layer.py) · [`replay_layer`](benchmarks/replay_layer.py) | 固定真实输入时，路由和专家输出是否一致 |
| 整模型服务 | [`bench_serve`](benchmarks/bench_serve.py) | 替换算子对吞吐和延迟的影响 |

已有 CUDA 环境时可先运行独立专家微基准。对照原生 vLLM 时将 `--backend custom` 换为 `--backend both`。

```bash
python -m benchmarks.bench_moe --backend custom --boundary experts --tokens 1 8 128
```

模型采集、回放和服务压测的完整命令见 [评测指南](benchmarks/README.md)。逐阶段的数值语义见 [PyTorch reference](moe_kernel/reference.py)。
