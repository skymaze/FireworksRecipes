# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks recipe (2× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) on **2** DGX Spark with **fp8 KV + DFlash2 block-diffusion
speculative decoding** — the config upstream marks as **"the one to copy"** (the
proven-&-reproducible tier of the sibling repo
`GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`).

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2`
  (same as the TP4 DFlash2 recipe: sm121-v8 + the 4-patch DFlash2 overlay)
- Two models (both distributed by repo id into the node HF cache, resolved offline):
  - main `LibertAIDAI/GLM-5.3-Flash-NVFP4` (120 shards ~182 GiB)
  - **DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2`** (2.34 GB)
- Topology: **2 nodes · TP=2** (`--tensor-parallel-size 2`), `mp` backend
- Context: **262,144 (262K)**; KV: **fp8_e4m3 · `--kv-cache-memory 3221225472`
  (3 GiB → 310,292-token pool)**
- Speculation: `--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  (must be 7 = block_size−1)
- Measured (upstream 2026-08-28, warm): **single-stream 46.9 tok/s · 74.1% acceptance**;
  C1–C6 concurrency sweep **zero failures**: aggregate 35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) / 47.7

## Quick start (before publishing)

- Cluster: exactly **2** nodes (head + 1 worker), dual-rail RoCEv2 tested.
- Image: `…/glm53-flash-sm121:v8-dflash2` (ACR, already pushed; nothing to build).
- Models: both the main model and the drafter are distributed via the recipe's
  `picker=model` variables.
- NCCL: HCA / ifname / GID index filled per node by Fireworks auto keys.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v8-dflash2` | Platform image (8-layer patch + DFlash2 overlay baked in) |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | Main model |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (make sure it is distributed) |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | Served name |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `262144` | **Upstream-validated** (TP2 carries ~97 GiB weights/rank; do not use 1M) |
| `MAX_NUM_SEQS` | `6` | Matches upstream |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Pairs with the pinned `KV_CACHE_MEMORY` |
| `KV_CACHE_MEMORY` | `3221225472` | **3 GiB pin** (validated in the upstream concurrency sweep; 4.1 GiB OOM'd under 3x 20K prefills) |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **Must = block_size−1** |
| `CHAT_TEMPLATE` | (empty) | In-container template path; set to the mm template to enable Vision |
| `MASTER_PORT` | `29521` | Distributed master port |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks.

## Publish note (task / project naming)

The task name becomes the Docker Compose project name: only **lowercase letters, digits, `-`
and `_`, no dots** (node Docker Compose v5 hard limit). Use dot-free names like
`glm53-dflash2-tp2`.

## Deployment notes (upstream field experience)

- **Cache-hit visibility**: `--enable-prefix-caching` (on by default for hybrid models) +
  `--enable-prompt-tokens-details`; the API reports `usage.prompt_tokens_details.cached_tokens`
  so you can verify prefix-cache hits (deep-session/agentic ~100×; compare a second request
  with the same prefix).
**TP2 carries ~97 GiB weights/rank; GB10 free-memory headroom is the hard constraint**:
  do not raise `KV_CACHE_MEMORY` (upstream ladder: any KV > 4.14 GiB NVRM-OOMs on some boot;
  3 GiB is the concurrency-validated shipping pin). During boot, upstream relies on the
  `cache_flusher` sidecar against page-cache-full allocations — Fireworks does not run it, so
  staying on the conservative pin matters even more.
- `num_speculative_tokens` must be 7 (block_size−1).
- The drafter drafts text only (vision requests still work, just not speculated).
- Cold JIT warm-up needed; healthy signatures: `Using Eagle3 auxiliary layers … (6,15,25,34,43)`,
  acceptance 0.6–0.8 (≈0.15 means aux capture/mHC contraction is wrong, silent degrade).
- Boot discipline: tear down all ranks before relaunching any; verify `IMAGE` on every node.
- C1–C6 is upstream's TP2 measurement; re-benchmark on your hardware (no flusher here) and
  backfill your own numbers.

## References

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark):
  sibling repo (TP2 variant) — `docs/BENCH-C1-C6-DFLASH2.md` TP2 table, `docs/GB10-KV-MEMORY-LADDER.md`
- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark): main 4x repo
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft model
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4): NVFP4 quant main model
