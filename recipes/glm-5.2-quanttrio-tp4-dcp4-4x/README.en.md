# GLM-5.2 QuantTrio · TP=4 · DCP4 · Fireworks recipe (4× DGX Spark)

Serve **GLM-5.2 QuantTrio** (Int4-Int8Mix, unpruned, 256 experts) at **TP=4 + DCP4** on
**4** DGX Spark nodes (head + 3 workers, dual-rail RoCEv2), at 315,968-token context.

## Model

- Base model: `QuantTrio/GLM-5.2-Int4-Int8Mix` (measured 378G = 124 weight + 4 MTP shards)
- Quant/speculation: Int4-Int8Mix · KV `nvfp4_ds_mla` · **MTP k=2** · attention
  `B12X_MLA_SPARSE`
- Parallelism: **TP=4 + DCP4** (`--dcp-comm-backend a2a`, `mp` backend)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit`
  (AEON `v0.27.1` + `b12x@334a2d75` + spark-kit production overlays)
- API port defaults to `8210`; defaults: gmu 0.90, `MAX_NUM_SEQS=16`

## Speed

Validated (2026-08-18, real 4× DGX Spark GB10, deployed end-to-end from this recipe):

- ~**9.5 min** to a serving API; **TTFT ≈ 0.64 s** (~0.72 s under 4-way concurrency)
- **~15–30 tok/s** single stream (including reasoning tokens); cold prefill ~580–618 tok/s
- **Prefix cache ≈100×**: turn 2+ runs ~1.3/1.7 s at 16K/100K context
- KV pool 683,360 tokens; the v0.27 4-node idle-stall (vllm #51921) did not reproduce

## Hardware requirements

- **4** DGX Spark nodes (fixed 4 nodes · TP=4 + DCP=4), one GB10 GPU each, dual-rail RoCEv2
- gmu 0.90 protects the KV pool (0.87 evicts deep sessions); keep
  `--max-num-batched-tokens` at 4096 (8192 fails to boot at DCP4); keep `MAX_MODEL_LEN`
  315,968 (64-aligned, otherwise it crashes under concurrency)
- Weights+compile ≈ 101.9 GiB of the 121.69 GiB unified memory per box

## Upstream references

- [joesinvestments/glm52-spark-kit](https://github.com/joesinvestments/glm52-spark-kit):
  the production DCP4 platform image and launcher (`platform/Dockerfile`,
  `launch/launch_gx10.sh`, `docs/RECOMMENDATION.md`)
- [lukealonso/b12x](https://github.com/lukealonso/b12x) · [vllm-project/vllm](https://github.com/vllm-project/vllm)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
