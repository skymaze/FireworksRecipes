# DeepSeek-V4-Flash DSpark · Fireworks recipe

Serves DeepSeek-V4-Flash on **exactly 2** DGX Spark nodes (head + 1 worker over RoCE):

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix2`
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
| `DSPARK_VLLM_IMAGE` | `…/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix2` | Mia hotfix-baked image |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `DSPARK_REVISION` | empty | Empty=auto-use cached snapshot sha (offline-safe); explicit pin must match the snapshot |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | Served model name |
| `VLLM_PORT` | `8888` | API port (serve `--port` only; decoupled from vLLM internal ports) |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `MAX_NUM_SEQS` | `6` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `GPU_MEMORY_UTILIZATION` | `0.835` | Text memory utilization |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Long-prefill chunk threshold (#27) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache checkpoint spacing (#26) |
| `DEFAULT_THINKING` | `max` | Thinking mode off/low/high/max |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `1` | #27: concurrent chunked prefills (1-3). #154 measured `2` widening the mixed-traffic fairness spread (3.72-5.14x vs 1.68-2.04x at `1`); `1` is the safe default, `2-3` are explicit operator opt-ins |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | 1 = enable GPU `thinking_token_budget` |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | `0` | 1 = apply the upstream vLLM #52805 XGrammar termination backport (#136, source-locked fail-closed) |
| `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` | `0` | 1 = accept type-less singleton assistant `output_text` replay (#138, minimal coercion only) |
| `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` | `0` | 1 = chunk sparse-MLA decode to fixed 64-row views (#141 stochastic-stall workaround, not a root-cause fix) |
| `DSPARK_API_KEYS` | empty | Space-separated multi-key auth (empty = no auth) |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks. Additional hotfix switches (`DSPARK_SKIP_HOTFIX`, …) are
exposed as recipe variables.

## Baked hotfix chain (v1.3.0 · snapshot 0107cef, 2026-08-29)

Applied fail-closed at every start:

- **Encoder** (#52 reasoning-effort map + #21): copies `encoding_dsv4.py` from the HF
  snapshot then patches (missing file warns, does not fail).
- **Python**: #55 tool-call truncation, #109 empty-encoder output, #27 partial-prefill
  concurrency, #43 decode fairness, #26 SWA prefix-cache, #133 Triton specialization,
  suppress-stops-in-reasoning.
- **Shell**: #22 nvfp4_ds_mla long context, #79 spin-wait, six v0.27 perf backports
  (#50312 / #49486 / #48407 / #48957 / #50298 / grammar-advance).
- **Opt-in (default off)**: `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`,
  `DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`, `DSPARK_API_KEYS` (multi-key auth + log redaction),
  plus the 2026-08-29 upstream additions `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX`
  (vLLM #52805 termination backport), `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT`
  (Responses history replay), `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` (64-row chunking).

> **2026-08-29 upstream sync**: the `v0.1.1-hotfix2` image is baked at `0107cef`; over
> the previous bake it adds the #136/#138/#141 opt-in patches plus the assistant-final
> branch (v1.3.0, all default off). #154 also reverted `DSPARK_MAX_INFLIGHT_PREFILLS`
> to a safe `1` default (2 widens mixed-traffic fairness spread).

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
- Concurrency stability (upstream #141/#143): sparse-MLA decode stalls are
  **stochastic per burst**; **no tested `MAX_NUM_SEQS` value is established as generally
  safe** — changing it alters incidence, it is not a fix. The easiest-to-miss failure
  signal is not a restart but silent stream truncation with a missing `finish_reason`.
  The shipped N=6/k=5 profile's 36 verify rows sit below the ~64-row boundary
  (consistent with the hypothesis but not a verified safety property); the engine also
  clamps CUDA-graph capture rows to 24 (the requested 36 never took effect). Read
  upstream #141 evidence and the #151 opt-in workaround before raising admission.
- Exactly 2 nodes (TP=2) only; pick a matching recipe or author your own for other
  topologies.
- **Mind the image version**: this recipe pins `v0.1.1-hotfix2` (0731 snapshot,
  the hardware-verified baseline). The registry also hosts `v0.1.1-hotfix4`
  (upstream 2026-08-31 vision-exp bake, native Vision-Exp image support and the
  MTP=6 pin) — but that image is **not hardware-verified against the 0731
  checkpoint**. Keep hotfix2 for 0731; for multimodal use the
  `deepseek-v4-flash-vision-exp-dspark` recipe.

## References

References (full attribution in the repo-root `NOTICE.md`):

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
