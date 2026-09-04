# DeepSeek-V4-Flash DSpark · Fireworks recipe (2× DGX Spark)

Serve DeepSeek-V4-Flash at **TP=2** on **2** DGX Spark nodes (head + 1 worker over RoCE)
from Fireworks, at 1M context.

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks,
  loaded offline)
- Quant/speculation: NVFP4 DS-MLA · FlashInfer b12x + dspark speculation (k=5) ·
  **1M context**
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix6`
  (Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + the Mia fail-closed hotfix chain)
- Served name: `deepseek-v4-flash-0731`; API port defaults to `8888`
- Default thinking `low` (overridable per request — off/low/high/max)

## Speed

No local measurements included (see the upstream benchmark in References).

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, RoCE
- `GPU_MEMORY_UTILIZATION` defaults to 0.835; drop to ~0.80 when 1M context is
  memory-tight

## Upstream references

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
