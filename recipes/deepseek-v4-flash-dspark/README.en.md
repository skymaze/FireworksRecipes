# DeepSeek-V4-Flash 2x DGX Spark (DSpark) · Fireworks recipe doc

> Migrated from Fireworks' former built-in seed recipe (MiaAI-Lab reference route); it is
> now a standard recipe in this source. Install it from the Fireworks Recipe Store and
> publish — it updates with this repo, **not** with Fireworks releases.

## What it is

Serves DeepSeek-V4-Flash on **exactly 2** DGX Spark nodes (head + 1 worker over the RoCE
high-speed network) with **Fireworks**:

- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (Anemll prebuilt vLLM distribution image)
- Topology: **fixed 2 nodes · TP=2** (NVFP4 DS-MLA + FlashInfer b12x + DSpark speculative)
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks model
  management to node HF caches and loaded offline)

> Difference from the "dedicated image" recipe (`deepseek-v4-flash-0731`): this one uses the
> Anemll distribution image for quickly trying the DSpark speculative path; the dedicated
> recipe builds vLLM v0.26.0 with a GB10-targeted overlay and has better prefill
> (`MoE=auto/DeepGEMM`, 2200+ tok/s). Pick whichever fits.

## Quick start

1. Cluster **exactly 2** nodes (head + worker) and configure/test RoCE.
2. Download `deepseek-ai/DeepSeek-V4-Flash-0731` (wizard can send it to nodes).
3. Pull `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (one-click distribution).
4. Recipe Store: install this recipe → **Install & run** → pick cluster → publish.

> The wizard locks the node count at exactly 2 (fixed topology; TP/distributed params are
> tuned for it — no vague "2 or more").

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll image |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `VLLM_PORT` | `8888` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `GPU_MEMORY_UTILIZATION` | `0.80` | Memory utilization (b12x stack) |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens (≥ checkpoint dspark_block_size=5) |
| `DEFAULT_THINKING` | `off` | off/low/high/max thinking mode |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR` (head's RoCE IP), `NODE_RANK`, `HEADLESS`,
`VLLM_HOST_IP`, `NCCL_*` are auto-filled.

## Known issues

- k<5 silently truncates dspark draft blocks (lower throughput); don't lower `MTP_NUM_TOKENS`.
- 1M context is memory-hungry on the b12x stack; keep `GPU_MEMORY_UTILIZATION` ≥0.80.
- Exactly 2 nodes (TP=2) only; pick a matching recipe or author your own for other topologies.

## Changelog

- **1.0.0**: migrated from Fireworks' built-in seed to this source; fixed 2-node topology;
  variable model aligned (`MASTER_ADDR=head_roce_ip`, `MASTER_PORT` as user variable 25000).
