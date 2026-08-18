# GLM-5.2 QuantTrio · TP=4 · DCP4 · Fireworks recipe (4× DGX Spark)

Serves **GLM-5.2 QuantTrio** (Int4-Int8Mix, unpruned, 256 experts) at **TP=4 + DCP4** on
**4** DGX Spark nodes (head + 3 workers over dual-rail RoCEv2):

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit`
  = AEON `v0.27.1` (sm_121a vLLM rebuild) + `b12x@334a2d75` + spark-kit production overlays
  (baked into the image, no runtime mounts)
- Parallelism: TP=4 + **DCP4** + `--dcp-comm-backend a2a`, `mp` backend, attention
  `B12X_MLA_SPARSE`
- Speculation: **MTP k=2** (probabilistic + `quantization: compressed-tensors` +
  `draft_tensor_parallel_size 1`)
- KV/shape: `--kv-cache-dtype nvfp4_ds_mla`, **315,968 context**, `--max-num-seqs 16`,
  `--max-num-batched-tokens 4096`, gmu 0.90, `FULL` capture ladder [3,6,9,12]
- API port defaults to `8210`; reasoning/tool parsers `glm45` / `glm47`

> Mirrors the author's **current production stack**: `joesinvestments/glm52-spark-kit`'s
> `platform/Dockerfile` + `launch/launch_gx10.sh` (RECOMMENDATION 2026-08-17: DCP4 stays
> production — 10-25% slower than DCP1 per-stream but holds three resident 316K sessions,
> and deep-session turns all land on the prefix cache). This recipe's image is built from
> that production V1 profile.

## Quick start (before publishing)

- Cluster: 4 nodes (head + 3 workers), dual-rail RoCEv2 configured and tested.
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit` pushed
  (built from the spark-kit production config; Fireworks pulls and distributes it).
- Model: `QuantTrio/GLM-5.2-Int4-Int8Mix` (HF hub layout `models--QuantTrio--GLM-5.2-Int4-Int8Mix`,
  378G = 124 weight + 4 MTP shards measured) distributed to the node cache; resolved offline by
  repo id inside the container (`HF_HOME=/cache/huggingface`, `HF_HUB_OFFLINE=1`).
- **NCCL**: no LD_PRELOAD by default — the image's bundled NCCL is verified end-to-end; a
  cache-distributed `nccl-2.30.4` is needed only when you deliberately set `NCCL_LD_PRELOAD`
  for wedge research.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM52_IMAGE` | `.../aixn-public/glm52-dcp4:v0.27.1-spark-kit` | Aliyun ACR platform image |
| `GLM52_MODEL_PATH` | `QuantTrio/GLM-5.2-Int4-Int8Mix` | Model (resolved offline via HF hub; or an absolute snapshot path in the cache) |
| `SERVED_MODEL_NAME` | `glm-5.2-quanttrio` | Served model name |
| `VLLM_PORT` | `8210` | API port |
| `MAX_MODEL_LEN` | `315968` | Production (64-aligned block-table seam; 316000/316K crashes) |
| `MAX_NUM_SEQS` | `16` | Production; 32 measured identical |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | **Do not use 8192** (3/3 boot failures at DCP4) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Protect the KV pool; 0.87 evicts deep sessions |
| `CPU_DIST_TIMEOUT` | `1800` | `--cpu-distributed-timeout-seconds` |
| `NCCL_LD_PRELOAD` | (empty) | Empty = image's bundled NCCL; set only for wedge research (nccl-2.30.4 path) |
| `MASTER_PORT` | `29501` | Distributed master port |

`NODES_TOTAL` (fixed 4), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks. The `hca` auto key returns comma-separated dual-rail per node
(`rocep1s0f0,roceP2p1s0f0`); `NCCL_IB_GID_INDEX` is auto-resolved (the source launcher also
resolves it dynamically each boot — don't hardcode).

## Validated on 4× DGX Spark GB10 (2026-08-18)

Deployed end to end on a real 4-node fleet (head + 3 workers) straight from this recipe:

- ~**9.5 min** to a serving API (weight load + CUDA graph warmup); all 4 containers stable
  (restart-count 0); image pulled from ACR directly onto the nodes.
- Mean **TTFT ≈ 0.64 s** once ready (/metrics); ~0.72 s under 4-way concurrency;
  ~15-30 tok/s single stream (including reasoning tokens).
- **The v0.27 4-node idle-stall (vllm #51921) did NOT reproduce**: requests still served
  immediately after 90 s idle. This stack (b12x + spark-kit overlays) covers the
  `shm_broadcast` / KV broadcast path.
- KV pool 683,360 tokens (weights+compile ≈101.9 GiB / 121.69 GiB, gmu 0.90, production profile).
- **GLM-5.2 reasoning is on by default and verbose**: it can exhaust `max_tokens` (`finish=length`,
  empty content, which is expected); the request-level `thinking:disabled` knob is not honored by
  this build — give `max_tokens` headroom in production.
- No nccl-2.30.4 was distributed and no `LD_PRELOAD` was set; the image's bundled NCCL was
  verified end to end (boot / multi-node NCCL / inference).

## Differences from the source launcher / integration

- **Overlays baked instead of runtime mounts**: production `launch_gx10.sh` mounts spark-kit
  overlays into site-packages; this image has the same set copied in (`torch_utils.py` /
  `kv_cache_interface.py` use the 3-way merged versions), so zero mounts at runtime.
- **Host variables** (`$NODES/$SSH_HOSTS/$WEIGHTS_DIR/$OVERLAY_DIR`) → Fireworks auto-filled
  variables (head_roce_ip / netdev / hca / gid_index / node_rank / headless).
- **`--master-port 29501` and `--cpu-distributed-timeout-seconds 1800` kept**.
- Added the Fireworks integration layer (host networking, `${HEADLESS:+--headless}`, HF
  offline load, `VLLM_HOST_IP`, persisted compile-cache dirs).

## Known issues & deployment notes (from the source repo's measurements)

- **Do not change**: keep `--all2all-backend` default (DeepEP measured 37-85% slower), no
  `--enable-dbo` (inherits DeepEP), gmu 0.90 (0.87 evicts deep sessions), `--max-num-seqs ≤16`,
  `--max-num-batched-tokens` at 4096 (8192 fails to boot at DCP4).
- **The cudagraph ladder [3,6,9,12] is `1+k` multiples**: a gap pads batches, and the
  non-uniform batch hits a broken sparse-MLA indexer branch and crashes — don't introduce
  gaps in the capture sizes.
- **Context-length alignment**: `MAX_MODEL_LEN` not a multiple of 64 opens the MTP-overhang
  seam (crash on the first request under concurrency); production 315,968 is verified stable.
- **The prefix cache is worth ~100x**: cold prefill ~580-618 tok/s, but turn 2+ is ~1.3/1.7s
  at 16K/100K context; use streaming clients for deep sessions and protect prefix-cache hits
  (GLM's template drops prior-turn reasoning; send
  `{"chat_template_kwargs":{"clear_thinking":false}}` when needed).
- **NCCL wedge**: the source keeps the full-speed config with an external watchdog
  (CROSS_NIC=1 default; the mitigations cost more than the disease). Compare `NCCL_CROSS_NIC=0`
  only if you actually hit the freeze.
- Single `mp` backend, one GPU per GB10 node; only **4 nodes (TP=4 + DCP=4)**.

## References

This project is **Apache-2.0** (see [`LICENSE`](../../LICENSE) at the repo root); third-party
component sources and derivations are listed in the repo-root
[`NOTICE.md`](../../NOTICE.md).

- [joesinvestments/glm52-spark-kit](https://github.com/joesinvestments/glm52-spark-kit): the
  production DCP4 platform image and launcher (`platform/Dockerfile`,
  `launch/launch_gx10.sh`, `docs/RECOMMENDATION.md`)
- [lukealonso/b12x](https://github.com/lukealonso/b12x): sparse MLA / MoE kernels
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine
