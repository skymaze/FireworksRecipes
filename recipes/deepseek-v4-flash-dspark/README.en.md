# DeepSeek-V4-Flash DSpark · Fireworks recipe

Serves DeepSeek-V4-Flash on **exactly 2** DGX Spark nodes (head + 1 worker over RoCE):

- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (Anemll prebuilt vLLM distribution image)
- Topology: **fixed 2 nodes · TP=2**; FlashInfer b12x + dspark speculation · NVFP4 DS-MLA ·
  1M context
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks, loaded
  offline)

> Difference from the "dedicated image" recipe `deepseek-v4-flash-0731`: this one uses the
> Anemll distribution image to quickly try the dspark speculative path; the dedicated recipe
> builds vLLM v0.26.0 with a GB10-targeted overlay and has better prefill (MoE=auto,
> 2200+ tok/s). Pick whichever fits.

## Quick start

Before publishing from Fireworks:

- Cluster: **exactly 2** nodes (head + 1 worker), RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` distributed to the nodes.
- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` pulled.

> The node count is locked at exactly 2; TP/distributed parameters are tuned for it.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll image |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `VLLM_PORT` | `8888` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `GPU_MEMORY_UTILIZATION` | `0.80` | Memory utilization (b12x stack) |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens |
| `DEFAULT_THINKING` | `off` | Thinking mode off/low/high/max |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks.

## Known issues

- Don't lower `MTP_NUM_TOKENS` below 5: k<5 silently truncates dspark draft blocks and
  lowers throughput.
- 1M context is memory-hungry on the b12x stack; keep `GPU_MEMORY_UTILIZATION` ≥0.80.
- Exactly 2 nodes (TP=2) only; pick a matching recipe or author your own for other
  topologies.

## References

References (full attribution in the repo-root `NOTICE.md`):

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) (DSpark recipe route reference)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
