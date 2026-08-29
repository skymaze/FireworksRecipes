# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 1M · Fireworks recipe (2× DGX Spark)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B MoE,
`glm5_next`) on the **EXL3/TR3 lane** with **2×** DGX Spark (head + 1 worker, direct CX7) —
a different image and lane from the NVFP4 (marlin) recipes in this repo:

- Weights: **[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
  (public mirror of the [brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  snapshot `5ab363a8…`; 4bpw matches official FP8 KLD ~1.00× at only **54%** of the bytes;
  falls back to brandonmusic if the mirror is incomplete);
- KV: **fp8 · packed `fp8_ds_mla`**;
- Speculation: **DFlash2 k=7** ([incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
  drafter stays on rank 0); the draft shares KV pages with MLA via **padded slot-share**;
- Context: **1,000,000** (padded slot-share is why 1M allocates); Vision on by default (image×4 / video×1).

> Do **not** use `--moe-backend marlin` / NVFP4 weights / `kv-cache-dtype nvfp4|bf16` /
> `attention_backend=TRITON_ATTN`; do not pull `glm53-flash-sm121:v8` (old NVFP4/Ray kernel).

## Measured (upstream 2026-08-28, sparkDash decode bench)

DFlash2 k=7 · Structured/Code high-accept regime · temp 0 · thinking off · 400 tokens · CUDA graphs · fused EXL3 MoE:

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

Lab (C1, median 5×400): Structured **61.7** tok/s (0.918 accept); Prose 26.9; long-context
(~60–100k KV) 24–27; MTP k=2 baseline ~24.6. Upstream 1M serve (util 0.87, same pool 1,754,237
/ **1.75×** / 690 blocks / **18.67 GiB**). KV headroom drifts per boot — this kit once left only
11.77 GiB at 0.87 (< the 14.61 GiB a 1M seq needs); first verify the node `:exl3` is the same build
as upstream, then raise util (≥0.90) only if 0.87 still falls short. Prefix
cache is block-aligned (3584-token): a ~7.7k follow-up hits 93%, TTFT 9.7 s → 1.17 s.

## Quick start

- Cluster: exactly **2** nodes (head + 1 worker), direct CX7 cabling (NCCL cannot use 10.0.0.x
  loopback aliases).
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (public GHCR, no login; the nodes do not
  pull — the image is distributed by the cluster mirror and used when present locally).
  If the KV pool is unexpectedly small, verify the `:exl3` on the nodes is the same build as upstream.
- Models: the **main model + DFlash2 drafter** are distributed by Fireworks via `picker=model` to
  each node's HF cache (`HF_HOME=/root/.cache/huggingface`, resolved offline by repo id).
- **NCCL**: HCA / interface / GID index auto-filled per node by Fireworks auto keys.
- The first request after a cold start triggers one JIT compile (upstream has a boot-warmup hook,
  Fireworks does not) — expect one slower call; Triton/TileLang caches are persisted.

## Main variables

| Variable | Default | Notes |
|---|---|---|
| `GLM53EXL3_IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | overlay image |
| `GLM53EXL3_MODEL_PATH` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | **EXL3 main model** (no NVFP4) |
| `GLM53EXL3_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (must reach the node cache) |
| `MAX_MODEL_LEN` | `1000000` | upstream production 1M (do not drop to 256k) |
| `GPU_MEMORY_UTILIZATION` | `0.87` | upstream-validated (pool ~18.67 GiB); if your boot is <14.61 GiB verify the image build first, then raise to ≥0.90 |
| `MAX_NUM_SEQS` | `4` | decode batch (upstream pin) |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | upstream pin (8192 blows the GB10 indexer smem) |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not bf16/nvfp4 |
| `GLM53EXL3_DFLASH_TOKENS` | `7` | DFlash2 tokens (trained block 8) |
| `GLM53EXL3_DFLASH_DRAFT_TP` | `1` | drafter stays on rank 0 |
| `EXL3_FUSED_MOE` | `1` | fused `exl3_moe` per layer; 0 = per-expert loop |
| `GLM53_MIXED_PREFILL_CHUNK` | `skip` | no peer prefill in a decode step (upstream pin) |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | client stops dormant while thinking |
| `LANGUAGE_MODEL_ONLY` | `0` | 0=load vision tower; 1=language-only (faster) |
| `LIMIT_MM` | `{"image":4,"video":1}` | per-request MM limit |
| `SKIP_MM_PROFILING` | `1` | keep 1 (profiling OOMs the UMA) |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template.jinja` | in-image MM template |
| `MASTER_PORT` | `29521` | distributed master port |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*` are
auto-filled by Fireworks; `SERVED_MODEL_NAME` (default `GLM-5.3-Flash-EXL3`) and `VLLM_PORT`
(default 8888) are also adjustable.

## Publish notes

- Task name = compose project name: lowercase letters / digits / `-` / `_` only, **no dots**
  (publishing fails with 502 otherwise). Suggested: `glm53-exl3-tp2`.
- **Thinking defaults ON**: disable with a top-level
  `"chat_template_kwargs": {"enable_thinking": false}` (`extra_body` is an SDK option — do not
  send a nested object over raw HTTP).
- `usage.prompt_tokens_details.cached_tokens` verifies prefix-cache hits (`--enable-prefix-caching` +
  `--enable-prompt-tokens-details` on; the OpenAI API is stateless — only block-aligned prefixes hit).
- **EXL3 ≠ NVFP4**: `--quantization exl3` is fixed; weights, KV and image must match. Draft KV is
  `auto`/bf16, TP=1 (dense DFlash2 cannot use the target's `fp8_ds_mla`); the target stays `fp8`.
- **KV headroom varies per kit**: if the boot log's `Available KV cache memory` is below 14.61 GiB
  (the 1M requirement), raise `GPU_MEMORY_UTILIZATION` (≥0.90); too high loses headroom over the
  MM / long-prefill activation peak.
- Cold start is slow; healthcheck `start_period` is 900s. CUDA graphs are on — do not `--enforce-eager`.
- The container runs the in-image runtime overlay patches on start (incl. disabling GB10 `persistent_topk`, which otherwise oversubscribes smem during CUDA-graph capture) — same as upstream start.sh.
- NCCL must use the direct CX7 ports (auto-filled per node), or `ncclCommInitRank` hangs.

## References

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):
  parameter-level reference (.env / start.sh / overlay / Dockerfile; 2026-08-29: 1M context +
  padded slot-share + hybrid prefix hit)
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw):
  EXL3/TR3 4bpw weights mirror (ShapleyMCG License 1.0) · [original](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft (CC BY-NC-ND 4.0)
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)
