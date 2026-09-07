# DeepSeek-V4-Flash DSpark · Fireworks recipe (2× DGX Spark)

Serve DeepSeek-V4-Flash at **TP=2** on **2** DGX Spark nodes (head + 1 worker over RoCE)
from Fireworks, at 1M context.

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks,
  loaded offline)
- Quant/speculation: NVFP4 DS-MLA · FlashInfer b12x + dspark speculation (k=5) ·
  **1M context**
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix7`
  (Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + Mia fail-closed hotfix chain, upstream snapshot
  `957890ac…` 2026-09-06)
- Served name: `deepseek-v4-flash-0731`; API port defaults to `8888`
- Default thinking `low` (overridable per request — off/low/high/max)

## New opt-in hotfixes in hotfix7 (default off)

The image carries the full upstream 09-06 patch tree; toggles default `0`, turn them on via env
(the recipe exposes four of them):

- `DSPARK_ENABLE_DSPARK_SWA_PREFIX`=1: recompute the draft sliding window on prefix-cache hits
  (Anemll #2 port; repeated identical prompts otherwise degenerate to a truncated response) —
  **correctness fix, recommended on**
- `DSPARK_ENABLE_DSML_RECOVERY`=1: a bare `<invoke name=...>` becomes a provisional tool call,
  validated against the request's declared tools (vllm#52645)
- `DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED`=1: strict named/required tool_choice contract
- `DSPARK_ENABLE_C128A_PREFILL_CACHE`=1: reuse the C128A prefill index conversion (SM120; no
  end-to-end speedup implied)
- The rest (`DSPARK_ENABLE_ROPE_SWA_FIX` / `DSPARK_ENABLE_DSPARK_BLOCK_K` /
  `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN` / `DSPARK_ENABLE_MXFP4_INDEXER_CACHE`, the latter
  requiring `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`) are injectable via env

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
