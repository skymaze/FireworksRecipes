# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks recipe (2× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) on **2** DGX Spark (head + 1 worker, dual-rail RoCEv2) with
**fp8 KV + DFlash2 block-diffusion speculative decoding** — the upstream sibling repo's
`GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` proven-&-reproducible tier (README: "This is the
config to copy").

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` (ACR,
  **the same image the TP4 DFlash2 recipe uses**): baked from the public upstream GHCR chain
  `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (day-0 + the **v1→v9 nine-layer
  patch stack** + the **DFlash2 overlay**), plus the **SM121 `sparse_attn_indexer_kpool`
  module** baked in (the same file upstream bind-mounts at launch — the platform cannot
  distribute host files) and the **mm chat template**.
- Main model (**default**):
  [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) —
  **compressed-tensors**, fixes the **intermittent corrupted-token IDs** of the ModelOpt
  builds (vLLM #54150); drop-in, zero flag changes, loads ~2× faster. ModelOpt builds remain
  usable but corrupted: censored `LibertAIDAI/GLM-5.3-Flash-NVFP4` (120 shards ~182 GiB),
  uncensored `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- Draft model: `incoai/GLM-5.3-Flash-DFlash2` (2.34 GB, qwen3 arch 5 SWA layers)
- Topology: **2 nodes · TP=2** (`--tensor-parallel-size 2`), `mp` backend
- Context: **262,144 (262K)** (TP2 carries ~97 GiB weights/rank — do not use 1M here; use the
  4-node recipe for 1M)
- KV: **fp8_e4m3 · let the profiler size the pool (no `--kv-cache-memory`) = 581,040-token
  pool** (upstream-verified)
- Speculation: `--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  (**must be 7 = block_size−1**)
- Measured (upstream 2026-08-28, warm): **46.9 tok/s single-stream · 74.1% draft acceptance**
  (structured output **54–61 tok/s**); C1–C6 concurrency sweep **zero failures**: aggregate
  35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) / 47.7; = **2.15×** over MTP-4 (21.8 tok/s)

> **The hard lesson on this TP2 line: never pin `--kv-cache-memory`.** With a pin, vLLM still
> runs its profile pass but **never subtracts the measured activation peak**
> (`--gpu-memory-utilization` becomes dead) — allocation, warmup and short generations all
> pass, then the **first long prompt has nowhere for its activations and the engine dies**
> (reproduced at four different pins upstream). The launcher's old 3 GiB pin (`3221225472`)
> is **stale residue**; this recipe defaults to a profiler-sized pool (DFlash2 + fp8 KV @262K =
> **581,040 tokens**, which survived a 28,818-token deep prompt, engine healthy after).

## Quick start (before publishing)

- Cluster: exactly **2** nodes (head + 1 worker), dual-rail RoCEv2 tested.
- Image: `…/glm53-flash-sm121:v11-dflash2` (ACR, pushed; same image as the TP4 DFlash2 recipe).
- Models: **both** the main model and the drafter are distributed to the node HF cache via
  `picker=model` variables (drafter only 2.34 GB); containers resolve them offline by repo id
  with `HF_HUB_OFFLINE=1`.
- NCCL: HCA / ifname / GID index filled per node by Fireworks auto keys.
- Boot order: Fireworks publishes workers first, head last.
- **Memory ritual (host side, every node)**: `vm.swappiness=0` (persist via `/etc/sysctl.d/`;
  swap may exist but **must never have swappiness** or the UVM driver livelocks and freezes
  shard loading), plus `sync; echo 3 > /proc/sys/vm/drop_caches` on both nodes before start.
- **Healthy signatures**: the log should show `Using Eagle3 auxiliary layers from config:
  (6, 15, 25, 34, 43)` and `Warming up spec-decode rejection sampler kernels (vocab=154880,
  num_spec=7, ...)`; the KV line should read **`GPU KV cache size: 581,040 tokens`** (yours
  will differ; the pool is the **minimum across ranks** and `Available KV cache memory` is
  logged by rank 0 only — read it on **every** node).

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v11-dflash2` | Platform image (same as TP4 DFlash2; nine-layer patch + DFlash2 overlay + SM121 indexer module baked in) |
| `GLM53_MODEL_PATH` | `RedHatAI/GLM-5.3-Flash-NVFP4` | Main model (compressed-tensors default; ModelOpt censored/uncensored drop-ins available but corrupted; or absolute snapshot path) |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (make sure it is distributed) |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | Served name |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `262144` | **Upstream-validated tier** (~97 GiB weights/rank at TP2 — do not use 1M) |
| `MAX_NUM_SEQS` | `6` | Same as upstream |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Upstream production value (0.78–0.80 starve the KV cache at 131K+); pairs with profiler sizing |
| `KV_CACHE_MEMORY` | (empty) | **Empty = do not pass the flag, profiler-sized pool (581,040 tokens) — upstream-recommended**; only fills in `--kv-cache-memory` when set (do not pin, see above) |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **Must = block_size−1**; K=7 is optimal, do not sweep |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template_mm.jinja` | In-image mm template (baked), Vision works out of the box (just not speculated); empty = text-only |
| `MASTER_PORT` | `29521` | Distributed master port |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks.

## Publish note (task / project naming)

The task name becomes the Docker Compose project name: only **lowercase letters, digits, `-`
and `_`, no dots** (node Docker Compose v5 hard limit). Dotted names fail publish with a 502.
Use a dot-free name like `glm53-dflash2-tp2`.

## Deployment notes (upstream field experience)

- **KV sizing discipline (the #1 TP2 lesson)**: let the profiler size the pool; do not pin —
  a pin skips the activation-peak deduction and the first long prompt NVRM-OOMs (reproduced at
  four different pins upstream; the launcher's 3 GiB pin is stale).
- **The drafter's hidden cost**: DFlash2 consumes ~4.8 GiB of KV headroom (far more than its
  2.2 GiB of weights) for ~+91 % decode speed / −40 % pool — pick per workload; without the
  drafter (MTP-4 v8 image) the pool is 965,166 tokens.
- **Reading KV numbers**: the headline `GPU KV cache size` = `max_concurrency ×
  max_model_len` and inflates with context; only `blocks × block_size` compares honestly
  across configs. `Available KV cache memory` is logged by rank 0 only, but the pool is built
  from the **minimum across ranks** — read that line on every rank. The TP worker profiles
  4–5 GiB less KV headroom than the head (unexplained upstream).
- **Do not run ModelOpt checkpoints in production** (token corruption, vLLM #54150); if you
  need uncensored, only the ModelOpt ablit build exists today (corrupted).
- `num_speculative_tokens` must be 7 (block_size−1); K=7 is optimal, do not sweep (the last
  position still earns ~0.94 conditional acceptance).
- The drafter drafts **text only** (vision requests work, just not speculated).
- **Cold JIT warm-up** (`_prepare_dflash_inputs_kernel`, `mhc_pre_big_fuse_with_norm_tilelang`);
  healthy signatures `Using Eagle3 auxiliary layers … (6,15,25,34,43)`; acceptance 0.6–0.8
  (~0.15 = aux capture / mHC contraction wrong, silent degrade).
- **`temperature: 0` is free throughput (+13–21 %)**; `enable_thinking: false` is also faster
  (+8 %) — note GLM emits untagged reasoning-prose into `content` with thinking off, which
  some agent harnesses mis-parse.
- Do not change: `--block-size 2304`, `--moe-backend marlin`, `--kv-cache-dtype fp8_e4m3`,
  `--enforce-eager`, the default `GPU_MEMORY_UTILIZATION` (0.78–0.80 starve KV).
- Memory discipline: `vm.swappiness=0` is **mandatory** and **does not survive a reboot**
  (persist it); with swap fully off the worker dies during MoE marlin repack, with default
  swappiness the UVM driver livelocks.
- Boot discipline: tear down all ranks before relaunching any; verify `IMAGE` matches on every
  node; capture `docker logs` before `rm -f`; probe `/health` for liveness, never `/v1/models`.
- **SM121 indexer module (disease 1)**: without it every decode past ~24K context hard-crashes;
  this recipe bakes it into the image (upstream bind-mounts the same file at launch).
- C1–C6 numbers are upstream TP2 measurements; re-validate on real hardware and backfill your
  own figures.

## References

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark):
  the sibling (TP2) repo — `launch-glm53-vllm-tp2-dflash2.sh`, `docs/BENCH-C1-C6-DFLASH2.md`,
  `docs/GB10-KV-MEMORY-LADDER.md`, `docs/SM121-CRASH-FORENSICS-2026-08-27.md`,
  `docs/KV-HUNT-672K-TP2-RECORD.md`, `docs/OPEN-PROBLEMS.md`
- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark): the 4x main repo (same image, TP4/1M)
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4): compressed-tensors main model (default)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft model
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine (`glm5_next` PR #53906, DFlash2 upstream PR #52816, token-corruption issue #54150)
