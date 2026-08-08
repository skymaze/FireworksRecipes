# DeepSeek-V4-Flash-0731 · Fireworks recipe doc

> This is the README (doc) attached to this catalog entry; it syncs from this git repo and
> renders inside the Fireworks Recipe Store. To get going fast: in Fireworks, open this
> recipe in the Recipe Store → **Install & run**.

## What it is

Serves **DeepSeek-V4-Flash-0731** on **exactly 2** DGX Spark nodes with **Fireworks**:

- Image: `fireworks-models/deepseek-v4-flash-0731:0.3.1` (mainline vLLM v0.26.0 +
  an Anemll-style GB10 overlay; the hybrid-draft-loader patch and tuned ENV are baked at
  build time)
- Topology: **2 nodes · TP=2** (head + 1 worker, clustered over RoCE)
- Loading: InstantTensor + dspark speculative decoding (MTP)
- KV cache: `nvfp4_ds_mla` (DS-MLA 4-bit, validated at **1M context**)
- MoE: `auto` (DeepGEMM) — on real hardware the b12x prefill path is unusable under the
  v0.26.0 integration (~88 tok/s) while `auto` reaches 2200+ tok/s

The weights are **not baked into the image** (~167 GB) and are distributed by Fireworks
model management to each node's HF cache; the image loads offline (`HF_HUB_OFFLINE=1`).

## Quick start

1. **Cluster**: add **exactly 2** deployed-Agent DGX Spark nodes to a cluster (head + 1
   worker) and configure/test the RoCE high-speed network (this recipe assumes a fixed
   2-node topology).
2. **Download the model** (optional; the publish wizard can handle it): control plane
   downloads `deepseek-ai/DeepSeek-V4-Flash-0731` → sends to head → RoCE-syncs to workers.
3. **Pull the image**: `fireworks-models/deepseek-v4-flash-0731:0.3.1` (one-click
   distribution too).
4. **Publish**: install this recipe in the Recipe Store → **Install & run** opens the
   publish wizard → pick the cluster → publish.

> The wizard locks the node count at **exactly 2** (TP/model params are tuned for it; no
> more vague "2 or more"). Model/image pickers are pre-filled from your local lists and
> prompt you to distribute before publishing.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Model downloaded via Fireworks |
| `VLLM_IMAGE` | `fireworks-models/deepseek-v4-flash-0731:0.3.1` | Dedicated image |
| `TENSOR_PARALLEL_SIZE` | `2` | Fixed 2 (1 GPU per GB10 node, TP = node count = 2) |
| `MAX_MODEL_LEN` | `1048576` | 1M context (needs KV `nvfp4_ds_mla`) |
| `GPU_MEMORY_UTILIZATION` | `0.88` | 1M context needs ≥0.88 (DeepGEMM stack) |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | Recommended; `fp8_ds_mla` works |
| `NUM_SPECULATIVE_TOKENS` | `5` | DSpark speculative tokens |
| `MASTER_PORT` | `25000` | Distributed master port (head rank 0 listens) |
| `DEFAULT_THINKING` | `off` | Thinking mode (off/low/high/max); leave empty if the image is not integrated |

`MASTER_ADDR`, `NODES_TOTAL`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*` are
auto-filled (head's RoCE IP, per-node role/rank, RoCE NIC/GID).

## Known issues & tips

- `GPU_MEMORY_UTILIZATION` at 0.80 only works on the b12x stack; DeepGEMM (1M ctx) needs ≥0.88.
- This recipe is tuned for a **fixed 2-node · TP=2** topology; for single-node or other
  topologies pick/await a matching recipe or author your own.
- The model is first distributed by Fireworks; the first publish loads it (time depends on
  storage/network).

## Changelog

- **0.3.1**: aligned to Fireworks' new variable model (`MASTER_ADDR=head_roce_ip` per-task
  head, `MASTER_PORT` moved to a recipe user variable); build refinements.
