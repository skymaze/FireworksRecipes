# DeepSeek-V4-Flash DSpark · Fireworks recipe

Serves DeepSeek-V4-Flash on **exactly 2** DGX Spark nodes (head + 1 worker over RoCE):

- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (Anemll prebuilt vLLM distribution image)
- Topology: **fixed 2 nodes · TP=2**; FlashInfer b12x + dspark speculation · NVFP4 DS-MLA ·
  1M context
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks, loaded
  offline)
- Served name: `deepseek-v4-flash-0731` (as of v1.1.0, matching the Mia upstream default)

> This recipe uses the Anemll prebuilt distribution image out of the box to try the
> FlashInfer b12x + dspark speculative path at 1M context (NVFP4 DS-MLA). The repo also
> ships a 4-node TP=4 DSpark recipe — pick by topology.

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
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | Served model name |
| `VLLM_PORT` | `8888` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `MAX_NUM_SEQS` | `6` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `GPU_MEMORY_UTILIZATION` | `0.835` | Text memory utilization (matches the Mia recipe) |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Long-prefill chunk threshold (#27, keeps decode fed) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache checkpoint spacing (#26) |
| `DEFAULT_THINKING` | `max` | Thinking mode off/low/high/max (matches the Mia recipe) |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks.

## Synced with the Mia upstream (v1.1.0 · 2026-08-25)

- **Defaults aligned**: `DEFAULT_THINKING=max`, `SERVED_MODEL_NAME=deepseek-v4-flash-0731`,
  `GPU_MEMORY_UTILIZATION=0.835` (text util).
- **JIT compile caches persisted** (Mia #65/#117): `TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR` /
  `B12X_CUTE_COMPILE_CACHE_DIR` land on the mounted HF volume, so a container recreate no
  longer re-JITs mid-serve (avoiding a TP-pair-desync hazard).
- **Long-prefill chunking** (#27): `--long-prefill-token-threshold 1024` keeps long
  prefills from starving decode lanes.
- **Prefix-cache retention** (#26): `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` sparsifies
  SWA checkpoints.

> Note: the batch of patch-style hotfixes the Mia recipe applies from its `patches/`
> directory at container entrypoint (#27 inflight cap, #43 decode fairness, boot-shape
> warmup, #133 Triton specialization, …) are **not** packaged here — this recipe is a
> self-contained compose that just runs `vllm serve`, and carries only the deployment
> knobs the stock runtime consumes directly. For the full hotfix chain use the upstream
> repo's `start-*.sh`.

## Known issues

- Don't lower `MTP_NUM_TOKENS` below 5: k<5 silently truncates dspark draft blocks and
  lowers throughput.
- 1M context is memory-hungry on the b12x stack; `GPU_MEMORY_UTILIZATION` defaults to
  0.835 — lower toward ~0.80 if graph capture OOMs.
- `DEFAULT_THINKING=max` reasoning can be very long (tens of thousands of chars on a
  moderate prompt in live measurement); size `max_tokens` accordingly, or use request-level
  `low`/`off`.
- Concurrent long prefills still queue (#27 semantics): this recipe chunk-limits at 1024 so
  decode is not starved, but multiple huge cold prefills are not served in parallel.
- Exactly 2 nodes (TP=2) only; pick a matching recipe or author your own for other
  topologies.

## References

References (full attribution in the repo-root `NOTICE.md`):

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) (DSpark recipe route reference; v1.1.0 aligned with its 2026-08-25 state)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
