# Single-GPU Qwen3.5 MoE kernels for vLLM 0.20.0

The optional vLLM general plugin replaces only the routed computation. The native
Router Linear, shared expert and output combination are retained.

- native: load the plugin entry point without changing vLLM.
- experts: keep vLLM Top-k and run the Triton expert kernels.
- full: replace routed Top-k and expert computation with Triton kernels.

Supported integration boundary: Qwen3.5 MoE, single CUDA GPU, unquantized BF16
or FP16 experts, no bias, standard softmax Top-k and SwiGLU. Use eager mode and
explicitly select the Triton vLLM backend to retain the expected gate/up weight
order. EP, EPLB, LoRA, MTP, pipeline parallelism and CUDA Graph are outside the
integration's supported scope.

Install into an existing matching vLLM v0.20.0 environment, keeping its PyTorch
and CUDA dependencies:

    python -m pip install --no-deps --no-build-isolation .
    python -c "from importlib.metadata import version; print(version('vllm'))"

Start the same command for each mode by changing MOE_KERNEL_MODE. The following
requires a GPU and an existing local checkpoint:

    MOE_KERNEL_MODE=experts MOE_KERNEL_MAX_TOKENS=2048 \
    vllm serve /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 1 \
      --language-model-only --enforce-eager --moe-backend triton \
      --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048

MOE_KERNEL_MAX_TOKENS must cover the scheduler's token budget. The process logs
the custom mode when it first allocates the workspace. Compare native, experts,
and full using the same model revision and launch arguments. A successful import
on a server without a GPU does not establish numerical or performance parity.
