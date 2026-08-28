# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 900K · Fireworks recipe (2× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) on **2×** DGX Spark (head + 1 worker, direct CX7 RoCEv2) on the
**EXL3/TR3 lane** — a different image and lane from the NVFP4 (marlin) recipes in this repo:

- Weights: **[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)**
  (uniform-K4 EXL3/TR3 routed-experts, 4 bpw, ~164 GiB, 120 shards) — 4bpw matches official FP8
  KLD (~1.00×) at only **54%** of the bytes;
- KV: **fp8 · packed `fp8_ds_mla`** (NoPE MLA zero-padded into the only sparse-MLA backend on SM12x);
- Speculation: **DFlash2 k=7** ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
  ~2.3 GiB BF16, drafter stays on rank 0);
- Context: **900,000**; Vision on by default (image×4 / video×1).

> **Do not** use `--moe-backend marlin` / NVFP4 weights / `kv-cache-dtype nvfp4|bf16` /
> `attention_backend=TRITON_ATTN` (causal-in-block on this image — collapses later-position
> acceptance). Do not pull `glm53-flash-sm121:v8` either — that is the older NVFP4/Ray kernel.

## Measured (upstream 2026-08-28, sparkDash decode bench)

DFlash2 k=7 · Structured/Code, same high-accept regime · temp 0 · thinking off · 400 tokens ·
CUDA graphs · fused EXL3 MoE:

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

Lab `tests/bench_decode.py` (C1, median 5×400): Structured **61.7** tok/s (0.918 accept /
6.43 per step); Prose 26.9 (0.332/2.33); long-context mixed (~60–100k KV) 24–27; MTP k=2
baseline ~24.6. KV pool **982,612** tokens (~15.67 GiB fp8 MLA) at util 0.87 — **1.09×** a full
900k request.

## Quick start (before publishing)

- Cluster: exactly **2** nodes (head + 1 worker), direct CX7 cabling (NCCL cannot use
  10.0.0.x loopback aliases).
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (public GHCR, no login).
- Models: the **main model + DFlash2 drafter** are both distributed by Fireworks via
  `picker=model` vars to each node's HF cache (`HF_HOME=/root/.cache/huggingface`, resolved
  offline by repo id).
- **NCCL**: HCA / interface / GID index auto-filled per node by Fireworks auto keys.

## Main variables

| Variable | Default | Notes |
|---|---|---|
| `GLM53EXL3_IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | overlay image (NoPE-MLA SM121 + exllamav3_ext + DFlash2/EAGLE3/video baked) |
| `GLM53EXL3_MODEL_PATH` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | **EXL3 main model** (do not swap in NVFP4) |
| `GLM53EXL3_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (must be distributed to the node cache) |
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | served model id |
| `VLLM_PORT` | `8888` | API port |
| `MAX_MODEL_LEN` | `900000` | **upstream production tier** (native 1M does not allocate) |
| `MAX_NUM_SEQS` | `4` | decode batch (upstream pin) |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | **upstream pin** (8192 blows the GB10 indexer smem) |
| `GPU_MEMORY_UTILIZATION` | `0.87` | 982,612-token pool |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; bf16/nvfp4 have no sparse kernel |
| `GLM53EXL3_DFLASH_TOKENS` | `7` | DFlash2 tokens (trained block 8) |
| `GLM53EXL3_DFLASH_DRAFT_TP` | `1` | drafter stays on rank 0 (no CX7 per draft step) |
| `LANGUAGE_MODEL_ONLY` | `0` | 0=load vision tower (default); 1=language-only |
| `SKIP_MM_PROFILING` | `1` | keep 1 (MM dummy profile OOMs the UMA) |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template.jinja` | in-image MM template |
| `MASTER_PORT` | `29521` | distributed master port |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks.

## Publish notes

- Task name = docker compose project name: only **lowercase letters / digits / `-` / `_`, no dots**
  (node Docker Compose v5 hard limit). A dotted task name fails to publish (502). Suggested:
  `glm53-exl3-tp2`.
- **Thinking defaults ON**: disable per request with a top-level
  `"chat_template_kwargs": {"enable_thinking": false}` (`extra_body` is an SDK option — do not
  send a nested object over raw HTTP).
- Vision goes through the MM template (default); `usage.prompt_tokens_details.cached_tokens`
  verifies prefix-cache hits (`--enable-prefix-caching` + `--enable-prompt-tokens-details` on;
  the OpenAI API is stateless — only block-aligned prefixes hit).

## Deploy notes (from upstream on-hardware lessons)

- **EXL3 ≠ NVFP4**: `--quantization exl3` is fixed; weights, KV and image must match or it fails.
- Draft KV is `auto`/bf16, TP=1 (dense DFlash2 cannot use the target's `fp8_ds_mla`); the target
  stays `fp8`.
- Cold start is slow (weight load + warmup); healthcheck `start_period` is 900s. CUDA graphs are
  on (capture `1 2 4 8 16 24 32`) — do not `--enforce-eager`.
- NCCL must use the direct CX7 ports (HCA/IF auto-filled per node); without them
  `ncclCommInitRank` hangs.
- Upstream pulls on the head and ships via `docker save | ssh docker load`; with Fireworks each
  node pulls the public image directly.

## References

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):
  parameter-level reference (.env / start.sh / overlay / Dockerfile)
- [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw):
  EXL3/TR3 uniform-K4 4bpw weights (ShapleyMCG License 1.0)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft
  model (CC BY-NC-ND 4.0)
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): base model
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3): EXL3 format/kernels
