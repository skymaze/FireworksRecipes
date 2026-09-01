# DeepSeek-V4-Flash-Vision-Exp DSpark · Fireworks Recipe

Serve DeepSeek-V4-Flash-**Vision-Exp** with Fireworks on **exactly 2** DGX Spark
nodes (head + 1 worker, RoCE):

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix3`
  (Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` baked with the **full Mia
  fail-closed hotfix chain + Vision-Exp native image support**, applied at
  startup before `vllm serve`)
- Topology: **fixed 2 nodes · TP=2**, FlashInfer b12x + dspark speculation k=6 ·
  NVFP4 DS-MLA · 1M context
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (DeepSeek's first experimental
  multimodal model of the V4 family, ~305B total params incl. ViT+Aligner, MIT;
  ~167GB, distributed by Fireworks and loaded offline; `DSPARK_REVISION` left
  empty = entrypoint auto-resolves the newest cached snapshot's commit sha,
  offline-safe)
- Multimodality: native **image input** (OpenAI `image_url`, JPEG/PNG/GIF/WebP;
  GIF decoded as a still frame; up to `LIMIT_MM_PER_PROMPT`, default 8, per
  request; **images in `user` messages only** — system/assistant image parts
  return HTTP 400; **no video encoder in the official weights**)
- Served name: `deepseek-v4-flash-vision-exp`

> This recipe uses the image pre-baked with the Mia hotfix chain (the in-image
> entrypoint applies patches in fail-closed order on every container start,
> matching the Mia repo `start-*.sh`). Vision support comes from upstream
> 2026-08-31 `feat/vision-exp` (PR #164): the `hotfix-dsv4-vision-exp.py`
> startup hotfix + `patches/vision_exp/` build the image tower, map
> `vision.*`/`aligner.*` weights and register a vLLM multimodal processor, and
> also carry the #165 fix (a literal `<image>` substring in system/assistant
> text is no longer mistaken for an image). Upstream removed the old Qwen3-VL
> sidecar / MCP path. The repo also has a 4-node TP=4 DSpark recipe (still on
> the 0731 text checkpoint); choose per topology.

## Quick start

Before publishing:

- Cluster: **exactly 2** nodes (head + 1 worker) with RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` distributed to nodes
  (incl. `encoding/encoding_dsv4.py`; the Vision-Exp ViT+Aligner weights take
  more VRAM than 0731, shrinking the KV pool).
- Image: ACR hotfix image `v0.1.1-hotfix3` (upstream snapshot `d58c877`,
  2026-08-31) pullable.

> Node count is locked to **exactly 2**; TP/distributed knobs are tuned for it.

Image request example:

```bash
curl -s http://<head-ip>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-vision-exp","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
    {"type":"text","text":"What is in this picture?"}]}]}'
```

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `…/dspark-vllm-gx10-mia:v0.1.1-hotfix3` | Mia hotfix-baked image (incl. Vision-Exp image support) |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | Downloaded model |
| `DSPARK_REVISION` | empty | Empty = auto-resolve local snapshot sha (offline-safe); upstream pin is `86f746b3…` |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-vision-exp` | Served model name |
| `VLLM_PORT` | `8888` | API port (serve `--port` only, decoupled from the VLLM_PORT env) |
| `MAX_MODEL_LEN` | `1048576` | 1M context |
| `MAX_NUM_SEQS` | `6` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `LIMIT_MM_PER_PROMPT` | `8` | Images per request (the `image=N` shorthand is converted to Anemll's JSON by the entrypoint; no video) |
| `GPU_MEMORY_UTILIZATION` | `0.835` | Text VRAM utilization (**do not set by hand** — Mia exports it from the TEXT tier; drop to ~0.80 if KV is tight) |
| `MTP_NUM_TOKENS` | `6` | dspark spec tokens (**do not go below 6**: Vision-Exp `num_nextn_predict_layers=3`, k must be ≥5 and divisible by 3; k=5 is rejected by Anemll) |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Long-prefill chunk threshold (#27) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache checkpoint spacing (#26) |
| `DEFAULT_THINKING` | `max` | off/low/high/max |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `1` | #27: concurrent chunked prefills (1-3). Upstream #154 measured 2 widening the mixed-traffic fairness spread (3.72-5.14x vs 1.68-2.04x at 1); 1 is the safe default, 2-3 are explicit operator opt-ins |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | 1 = enable GPU `thinking_token_budget` |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | `0` | 1 = apply the upstream vLLM #52805 XGrammar termination backport (#136, source-locked fail-closed) |
| `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` | `0` | 1 = accept type-less assistant `output_text` singleton replay (#138, minimal coercion) |
| `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` | `0` | 1 = chunk sparse-MLA decode to fixed 64-row views (#141 stochastic stall workaround, not a root-cause fix) |
| `DSPARK_API_KEYS` | empty | Space-separated multi-key auth (empty = no auth) |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`,
`NCCL_*` are auto-filled by Fireworks.

Remaining hotfix toggles (`DSPARK_SKIP_HOTFIX` etc.) live in the recipe variables.

## KV pool & concurrency

Vision-Exp weights are larger than 0731 (~305B vs 284B total per the model card;
the ViT+Aligner is resident), so the KV pool shrinks. Upstream reference
boot log on the same profile (util 0.83):

```text
Available KV cache memory: 17.04 GiB
GPU KV cache size: 2,331,430 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.22x
```

`MAX_MODEL_LEN` / `MAX_NUM_SEQS` are **ceilings, not reservations**; the real
constraint is `sum(live tokens) <= KV pool`. Six normal agent turns fit; six
simultaneous full-1M requests do not (they queue).

Upstream measured on the Anemll 1M/6 tier (results/RESULTS-2026-08-14.md):

| Workload | Reference |
| --- | --- |
| One chat (any prompt through 128K) | ~62-83 decode tok/s after first token |
| **Six short chats** (hundreds of tokens) | **~160-190 tok/s aggregate** (~30-37 per stream) |
| Six cold 32K-128K prompts at once | Prefills queue (#27), ~8 tok/s decode floor |

## Hotfix chain (v1.4.0 · snapshot d58c877, 2026-08-31)

Applied on every container start in fail-closed order:

- **Vision-Exp image support** (`hotfix-dsv4-vision-exp.py` + `patches/vision_exp/`,
  unconditional fail-closed): build the ViT+Aligner image tower, map
  `vision.*`/`aligner.*`/`bias_vl` weights (incl. the MoE 0-2 layer
  `ffn.gate.bias_vl` remap), register the multimodal processor, restrict images
  to `user` messages (#165 fix: system/assistant messages that merely *mention*
  `<image>` are no longer rejected).
- **Encoding hotfix** (#52 reasoning-effort mapping + #21): patches after
  copying `encoding_dsv4.py` from the HF snapshot (missing file warns, doesn't fail).
- **Python**: #55 tool-call truncation, #109 empty encoder output, #27 partial-
  prefill concurrency, #43 decode fairness, #26 SWA prefix cache, #133 Triton
  specialization, suppress-stops-in-reasoning.
- **Shell**: #22 nvfp4_ds_mla long context, #79 spin-wait, six v0.27 perf
  backports (#50312 / #49486 / #48407 / #48957 / #50298 / grammar-advance).
- **Optional (default off)**: `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`,
  `DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`,
  `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` (vLLM #52805 termination backport),
  `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` (Responses history replay),
  `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` (64-row chunking),
  `DSPARK_API_KEYS` (multi-key auth + log redaction).

> **2026-08-31 upstream sync (v1.4.0)**: retargeted to `DeepSeek-V4-Flash-Vision-Exp`
> (upstream PR #164 `feat/vision-exp`, snapshot `de230b45bc49…`): model/served name/image
> tag all switched, `MTP_NUM_TOKENS` default 5 → 6 (Vision-Exp `n_predict=3`), new
> `LIMIT_MM_PER_PROMPT` (images per request). Vs `v0.1.1-hotfix2` (snapshot
> `0107cef`), the image gains: native Vision-Exp image support (#164), the paired
> `<image>` tag role check (#165), and the tool-result text scan skip (#167).
> **`v0.1.1-hotfix3` is baked and pushed to ACR** (manifest `sha256:e9c9dca7…`,
> single-arch arm64, 2026-09-01). In-container dry-run verified: full hotfix chain
> order, the vision trio (model/encoding/dspark) APPLIED, `--limit-mm-per-prompt
> {"image":N}` conversion, the MTP-6 capture size 42, and fail-closed refusal
> without the checkpoint encoding. **The checkpoint distribution still needs
> hardware validation** — confirm the Model page has distributed
> `DeepSeek-V4-Flash-Vision-Exp` to both nodes before publishing.

Triton / TileLang / B12X-CuTeDSL JIT compile caches are persisted to the HF
volume (container reinstantiation doesn't re-JIT and TP stays in sync).

## Known issues

- Do not set `MTP_NUM_TOKENS` below 6: Vision-Exp `num_nextn_predict_layers=3`,
  k must be ≥ `dspark_block_size`(5) and divisible by 3; k=5 is rejected outright.
- Images in **user messages only**: `image`/`image_url` parts in system/assistant
  messages return HTTP 400 (matches official Chat Completions); plain text that
  mentions `<image>` is fine (#165 fixed, effective in hotfix3).
- **No video**: the official weights ship no video encoder; GIF is decoded as a
  still RGB frame.
- The b12x stack at 1M context is VRAM-tight, and Vision-Exp weights take more
  VRAM than 0731 with a smaller KV pool; `GPU_MEMORY_UTILIZATION` defaults to
  0.835 — lower to ~0.80 on pressure or graph-capture OOM.
- `DEFAULT_THINKING=max` reasoning can be long (measured ~12.5k tokens / tens of
  thousands of chars on a moderate prompt); budget `max_tokens` in the tens of
  thousands or use `low`/`off` per request.
- Concurrent long prefills still queue (#27 semantics): 1024-threshold chunking
  keeps decode fed, but multiple huge cold prefills cannot run in parallel.
- Concurrency stability (upstream #141/#143): sparse-MLA decode stalls are
  **per-burst stochastic**; **no `MAX_NUM_SEQS` value is proven generally safe** —
  changing it shifts probability, it is not a fix. The earliest silent failure
  signal is a truncated stream missing `finish_reason`. Default N=6/k=6 give 42
  verify rows (6×7); the engine may clamp CUDA-graph capture rows to ~32. Read
  upstream #141 evidence and the #151 opt-in workaround before raising it.
- Exactly 2 nodes (TP=2) only; other topologies need another recipe.
- **Dev-branch recipe**: v1.4.0 depends on the `v0.1.1-hotfix3` image (pushed to
  ACR on 2026-09-01) and the `DeepSeek-V4-Flash-Vision-Exp` checkpoint
  distribution (**not hardware-validated yet**); confirm the Model page has
  distributed the checkpoint to both nodes before publishing.

## References

Full attribution in root `NOTICE.md`:

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  (vision-exp · snapshot `d58c877`, 2026-08-31)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
- Upstream checkpoint/encoder doc: [docs/DEEPSEEK_V4_FLASH_0731.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)
- Upstream benches: [results/RESULTS-2026-08-14.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/results/RESULTS-2026-08-14.md)
