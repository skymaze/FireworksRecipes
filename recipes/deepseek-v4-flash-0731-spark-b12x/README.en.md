# DeepSeek-V4-Flash-0731 · Spark-vLLM b12x · Fireworks recipe (2× DGX Spark)

Serve **DeepSeek-V4-Flash-0731** at **TP=2** on **2** DGX Spark nodes (head + 1 worker
over RoCE) from Fireworks, at 1M context.

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-0731` (distributed by Fireworks, loaded offline)
- Backends/quant: **B12X MLA SPARSE** attention + **b12x** MoE/linear · **FP8 KV** · dspark
  speculation (k=5) · 1M context; instanttensor + AOT compile (fast first boot)
- Image: `eugr/spark-vllm-b12x:latest` (spark-vllm b12x prebuilt vLLM distribution image)
- API port defaults to `8000`; default thinking mode `high`

## Speed

No local measurements included (this lane is not hardware-validated yet).

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, RoCE
- `MAX_NUM_SEQS=8` (source command value); `GPU_MEMORY_UTILIZATION` defaults to 0.85
  (existing recipes validated 0.80 stable; 0.90 fails to boot)
- `LOAD_FORMAT=instanttensor` requires an instanttensor layout in the node cache
  (switch to auto/safetensors for standard HF distribution)

## Upstream references

- `eugr/spark-vllm-b12x:latest`: distribution image (from the source docker run)
- [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
