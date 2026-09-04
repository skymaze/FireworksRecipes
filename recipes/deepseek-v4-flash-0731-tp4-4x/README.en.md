# DeepSeek-V4-Flash-0731 · TP=4 · Fireworks recipe (4× DGX Spark)

Serve **DeepSeek-V4-Flash-0731** at **TP=4** on **4** DGX Spark nodes (head + 3 workers
over RoCE) from Fireworks, with defaults pinned by real-hardware validation on an agentic
workload (1M context).

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks,
  loaded offline)
- Quant/speculation: NVFP4 DS-MLA · FlashInfer b12x + dspark speculation (k=5) ·
  **1M context**
- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (Anemll prebuilt vLLM distribution image)
- API port defaults to `8888`

## Speed

No local measurements included (see the upstream benchmark in References).

## Hardware requirements

- **4** DGX Spark nodes (fixed 4 nodes · TP=4), one GB10 GPU each, RoCE
- Hardware-validated defaults: `GPU_MEMORY_UTILIZATION=0.80`, `DEFAULT_THINKING=max`,
  1M context
- `MAX_NUM_SEQS=4` (`max_cudagraph_capture_size` computed as 4×6)

## Upstream references

- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
