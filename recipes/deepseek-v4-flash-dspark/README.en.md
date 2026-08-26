# DeepSeek-V4-Flash DSpark · Fireworks recipe

Serves DeepSeek-V4-Flash on **exactly 2** DGX Spark nodes (head + 1 worker over RoCE):

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix`
  (A fully self-contained image: the Mia **fail-closed hotfix chain** baked on top of
  Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, applied at container start before `vllm serve`)
- Topology: **fixed 2 nodes · TP=2**; FlashInfer b12x + dspark speculation · NVFP4 DS-MLA ·
  1M context
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks, loaded
  offline; `DSPARK_REVISION` defaults to empty = entrypoint auto-resolves the local cached
  snapshot's commit sha, so offline load is safe)
- Served name: `deepseek-v4-flash-0731`

> This recipe uses a distribution image with the full Mia hotfix chain baked in
> (`/opt/dspark/entrypoint.sh` applies the fail-closed chain at every container start,
> mirroring the Mia repo's `start-*.sh`). The repo also ships a 4-node TP=4 DSpark recipe —
> pick by topology.

## Quick start

Before publishing from Fireworks:

- Cluster: **exactly 2** nodes (head + 1 worker), RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` distributed to the nodes (including
  `encoding/encoding_dsv4.py`).
- Image: ACR hotfix image pullable.

> The node count is locked at exactly 2; TP/distributed parameters are tuned for it.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `…/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix` | Mia hotfix-baked image |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `DSPARK_REVISION` | empty | Empty=auto-use cached snapshot sha (offline-safe); explicit pin must match the snapshot |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | Served model name |
| `VLLM_PORT` | `8888` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `MAX_NUM_SEQS` | `6` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `GPU_MEMORY_UTILIZATION` | `0.835` | Text memory utilization |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Long-prefill chunk threshold (#27) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache checkpoint spacing (#26) |
| `DEFAULT_THINKING` | `max` | Thinking mode off/low/high/max |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `2` | #27: concurrent chunked prefills (1-3) |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | 1 = enable GPU `thinking_token_budget` |
| `DSPARK_API_KEYS` | empty | Space-separated multi-key auth (empty = no auth) |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks. Additional hotfix switches (`DSPARK_SKIP_HOTFIX`, …) are
exposed as recipe variables.

## Baked hotfix chain (v1.2.0 · snapshot 70a7cc4b, 2026-08-25)

Applied fail-closed at every start:

- **Encoder** (#52 reasoning-effort map + #21): copies `encoding_dsv4.py` from the HF
  snapshot then patches (missing file warns, does not fail).
- **Python**: #55 tool-call truncation, #109 empty-encoder output, #27 partial-prefill
  concurrency, #43 decode fairness, #26 SWA prefix-cache, #133 Triton specialization,
  suppress-stops-in-reasoning.
- **Shell**: #22 nvfp4_ds_mla long context, #79 spin-wait, six v0.27 perf backports
  (#50312 / #49486 / #48407 / #48957 / #50298 / grammar-advance).
- **Opt-in (default off)**: `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`,
  `DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`, `DSPARK_API_KEYS` (multi-key auth + log redaction).

Triton / TileLang / B12X-CuTeDSL JIT compile caches persist on the HF volume (no
mid-serve re-JIT on container recreate, avoiding a TP-pair-desync hazard).

## Known issues

- Don't lower `MTP_NUM_TOKENS` below 5: k<5 silently truncates dspark draft blocks and
  lowers throughput.
- 1M context is memory-hungry on the b12x stack; `GPU_MEMORY_UTILIZATION` defaults to
  0.835 — lower toward ~0.80 if graph capture OOMs.
- `DEFAULT_THINKING=max` reasoning can be very long; size `max_tokens` accordingly, or use
  request-level `low`/`off`.
- Concurrent long prefills still queue (#27 semantics); the 1024 threshold keeps decode
  fed, but multiple huge cold prefills are not served in parallel.
- Exactly 2 nodes (TP=2) only; pick a matching recipe or author your own for other
  topologies.

## Image build source

Build context lives outside this repo at `FireworksProject/dspark-image-build/`
(Dockerfile + entrypoint.sh + patches/); base image Anemll `0.1.1`, snapshot
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark @ 70a7cc4b`.

## References

References (full attribution in the repo-root `NOTICE.md`):

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
