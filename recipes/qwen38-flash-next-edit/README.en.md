# Qwen3.8-Flash-Next · Edit · single-node llama.cpp · Fireworks recipe (1× DGX Spark)

Serve **Qwen3.8-Flash-Next** on **one** DGX Spark (GB10, 128 GiB unified memory) with the
**llama.cpp coding-edit lane** (the containerized form of the upstream `edit` setup) — best
for coding agents that rewrite files you hand them, at the **full 262,144-token context**.

## Model

- Base model: `unsloth/Qwen3.8-Flash-Next-GGUF` **UD-Q4_K_XL** shards (~104 GiB) +
  `mmproj-F16.gguf` (~0.9 GiB vision projector, optional)
- Same key trick as the vLLM recipe: the 51.2B n-gram/PLE lookup table stays on the
  **NVMe page cache** (`-lm mmap` + `-ot per_layer_token_embd=CPU`), not on the GPU
- Speculation: **ngram-mod** context-copying (verifies 60-token spans in one pass —
  **exact**, byte-identical output); **no MTP** (the GGUF converter drops the trained draft
  head)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0`
- API port defaults to `8000`
- Versus the vLLM recipe: no NVFP4 checkpoint to download, and a lower-bit quant is
  available for disk; for chat/reasoning/long documents use the
  [`qwen38-flash-next-vllm`](../qwen38-flash-next-vllm/README.en.md) recipe (steadier, ~5×
  faster prefill; they share the GPU — only one runs at a time)

## Speed

Measured (upstream):

- **Judge speed by task, not by a single number**: rewriting a file you handed it (answer
  already in the prompt) is **88 tok/s**; writing something new (nothing to copy) is
  **~28 tok/s**
- Speculation is exact: output is byte-identical to disabling it — no quality-for-speed trade
- `--parallel 2` gives ~1.24–1.30× under load, free at one caller; thinking tokens dominate
  (as in the vLLM recipe — request-level thinking-off cuts the same answer from ~55 s to ~15 s)

## Hardware requirements

- **1** DGX Spark node (fixed single node · GB10, 128 GiB unified memory)
- Context is split across slots: with two slots a too-long prompt returns 400 (not
  truncated) — for very long documents set `PARALLEL=1` or use the vLLM recipe
- **Disk**: UD-Q4_K_XL ~104 GiB + image ~4–6 GiB

## Upstream references

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  (MIT): the edit lane's container Dockerfile, serving config and measurements
  (`recipes/llamacpp-edit/`)
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
  UD-Q4_K_XL quants and mmproj
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): the engine

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
