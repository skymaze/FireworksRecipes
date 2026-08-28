# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks recipe (4× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) at **TP=4** on **4** DGX Spark with **Lane A (fp8 KV) +
DFlash2 block-diffusion speculative decoding** — the first working DFlash2 deployment on GB10:
single-stream **46.9 tok/s (2.15× over MTP-4)**.

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2`
  = upstream sm121-v8 build chain (day-0 + 8-layer patch, i.e. the Lane A image) + the
  **4-patch DFlash2 overlay**:
  1. `qwen3_dflash2.py` + `dflash2/` + registry/selection wiring (ports upstream PR #52816's
     DFlash2 drafter/selector into this vLLM tree, registers `DFlash2DraftModel`)
  2. GLM aux hidden-state capture (`SupportsEagle3` + 5 tap layers `(6,15,25,34,43)` +
     mHC contraction)
  3. drafter KV group **kept inside GLM's custom fast path** — the 5 `SlidingWindowSpec`
     layers slot-share the MLA tensors (~zero KV-pool cost; the generic path inflates a
     262K request ~13×)
  4. `patch_kv_page_lcm2.py` (documented no-op)
- Two models (both distributed by repo id into the node HF cache, resolved offline):
  - main `LibertAIDAI/GLM-5.3-Flash-NVFP4` (120 shards ~182 GiB, censored)
  - **DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2`** (single `model.safetensors` 2.34 GB,
    qwen3 arch 5 SWA layers, `block_size=8` / `selector_rank=256` /
    `target_layer_ids [5,14,24,33,42]`; sensitive main-model drops in unchanged)
- Speculation: `--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  (**must be 7 = block_size−1**)
- KV / shapes: **fp8_e4m3**, 24 GiB per rank, `--block-size 2304`, gmu 0.85,
  **1,048,576 context**, `--max-num-seqs 6`, port `8000`
- Measured (upstream 2026-08-28, TP2/262K, warm): **46.9 tok/s single-stream · 74.1% draft
  acceptance**; C1–C6 concurrency sweep with zero failures (C5 aggregate peak 56.2 tok/s);
  upstream states "the same overlay applies to this TP4 recipe"

> vs Lane A (MTP-4): DFlash2 is ~2.15× single-stream on identical hardware/context with
> ~zero KV-pool cost. TP4 MTP single-stream measured ~55 tok/s; TP4 DFlash2 numbers are not
> separately published upstream — this recipe follows "the same overlay applies to TP4" and
> suggests backfilling numbers after a real-machine run.

## Quick start (before publishing)

- Cluster: exactly **4** nodes (head + 3 workers), dual-rail RoCEv2 tested.
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2` (ACR, pushed).
- Models: **both** the main model and the drafter are distributed to the node HF cache via
  the recipe's `picker=model` variables (drafter is only 2.34 GB); containers resolve them
  offline by repo id with `HF_HUB_OFFLINE=1`.
- NCCL: HCA / ifname / GID index filled per node by Fireworks auto keys.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v8-dflash2` | Platform image (8-layer patch + DFlash2 overlay baked in) |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | Main model (offline HF-hub resolution; or absolute snapshot path) |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (second model — make sure it is distributed) |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | Served name (drop-in name) |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `1048576` | Model-native 1M; lower (e.g. 300000) for snappier multi-user, keep 64-aligned |
| `MAX_NUM_SEQS` | `6` | Same as Lane A |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Pairs with the pinned `KV_CACHE_MEMORY` |
| `KV_CACHE_MEMORY` | `25769803776` | Per-rank fp8 KV budget (24 GiB); the drafter costs ~0 KV, no cut here |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **Must = block_size−1** |
| `CHAT_TEMPLATE` | (empty) | In-container template path; set to the mm template to enable Vision |
| `MASTER_PORT` | `29521` | Distributed master port |

`NODES_TOTAL` (fixed 4), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks.

## Publish note (task / project naming)

The task name becomes the Docker Compose project name: only **lowercase letters, digits, `-`
and `_`, no dots** (node Docker Compose v5 hard limit). Dotted names such as `glm5.3-flash-nv`
fail publish with `invalid project name ...` (502). Use a dot-free name like
`glm53-flash-dflash2-4x`.

## Deployment notes (upstream field experience)

- **`num_speculative_tokens` must be block_size−1 = 7**: the drafter is trained for a block of
  8 and the last position is the target's own verified token; 8 drafts a position the model
  never learned.
- The drafter drafts **text only**: vision requests still work but are not speculated (log
  warns "does not support external multimodal embeddings").
- **Cold JIT warm-up**: the first request JITs `_prepare_dflash_inputs_kernel` and
  `mhc_pre_big_fuse_with_norm_tilelang`; a cold measurement reads ~10 tok/s low — warm first.
- Healthy signatures: `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)` and
  `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`; acceptance
  (`spec_decode_num_accepted_tokens_total ÷ num_draft_tokens_total`) ≈ 0.6–0.8 — near 0.15
  means the aux capture or mHC contraction is wrong (silent degrade, no crash).
- Do not change: `--block-size 2304`, `--moe-backend marlin`, `--kv-cache-dtype fp8_e4m3`,
  `--enforce-eager`, the default `KV_CACHE_MEMORY` (bigger slabs NVRM-OOM under concurrent
  prefills on GB10 — gate every bump behind a real long prefill).
- Boot discipline (hard-won rules): tear down all ranks before relaunching any; verify `IMAGE`
  matches on every node; capture `docker logs` before `rm -f`.
- At TP2 upstream tightened KV to 3 GiB for headroom; **do not copy that at TP4** (~50 GiB
  weights/rank + 24 GiB KV was validated; the drafter adds only 2.34 GB/rank).

## References

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
  `docs/DFLASH2-SPECULATIVE-DECODING.md` (method + nine-boot failure ladder),
  `docs/BENCH-C1-C6-DFLASH2.md`, `overlay-dflash2/` (reproducible 4-patch overlay)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft model
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4): NVFP4 quant main model
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine (DFlash2 upstream PR #52816)
