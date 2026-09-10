# DeepSeek-V4-Flash-Vision-Exp · Spark-vLLM b12x · Fireworks recipe (2× DGX Spark)

Serve **DeepSeek-V4-Flash-Vision-Exp** at **TP=2** on **2** DGX Spark nodes (head + 1 worker
over RoCE) from Fireworks, at 1M context, with native image input.

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (distributed by Fireworks, loaded offline)
- Backends/quant: **B12X** attention + **b12x** MoE/linear (sparse indexer) · **FP8 KV** ·
  dspark speculation (k=6) · 1M context; instanttensor + AOT compile (fast first boot)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0`
  (baked: `eugr/spark-vllm-b12x` nightly-20260909 base with the
  `instanttensor-hybrid-draft-loader` patch applied at build time, see
  [`docker/spark-vllm-b12x`](../../docker/spark-vllm-b12x/README.md);
  corresponds to the upstream `--exp-b12x` vision build)
- API port defaults to `8000`; default thinking mode `high` (aligned with upstream
  `thinking=true / effort=high`)

## Speed

No local measurements included (this lane is not hardware-validated; parameters follow the
upstream vision-exp recipe added 2026-09-08, i.e. the new b12x branch from 2026-09-03:
`--attention-backend B12X`, capture 48, MEGA_AOT_ARTIFACT 1).

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, RoCE
- `MAX_NUM_SEQS=8` (source command value); `GPU_MEMORY_UTILIZATION` defaults to 0.85
  (existing recipes validated 0.80 stable; 0.90 fails to boot)
- `LOAD_FORMAT=instanttensor` requires an instanttensor layout in the node cache
  (switch to auto/safetensors for standard HF distribution)

## Deltas from upstream

- The upstream `mods/instanttensor-hybrid-draft-loader` (target keeps InstantTensor, dspark
  draft switches to lazy safetensors) is **baked into the image** (patched at build time,
  `INSTANTTENSOR_DRAFT_LOADER=auto`), so the upstream runtime `--apply-mod` step is not
  needed; the base is pinned to the 09-09 nightly and the bake only succeeds when the base
  image's vLLM source matches (fail-fast)
- Upstream uses empty `--reasoning-config` delimiters; fixed here to `" thinking"` /
  `" response"` per the verified dspark recipes (otherwise reasoning sections do not split)
- `--max-model-len` pinned to 1048576 instead of upstream `auto`; multi-node coordination
  uses Fireworks auto variables (`--nnodes/--node-rank/--master-addr/--headless`, host shell
  vars upstream)
- Inherits the existing b12x recipe's RoCE/NCCL integration layer and ulimit fix (nofile=1048576)

## Upstream references

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT):
  `recipes/deepseek-v4-flash-vision-exp.yaml`, `recipes/deepseek-v4-flash-0731.yaml`
- `eugr/spark-vllm-b12x:latest`: distribution image (`--exp-b12x` build)
- [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
