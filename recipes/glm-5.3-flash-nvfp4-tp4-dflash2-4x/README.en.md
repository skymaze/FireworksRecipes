# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks recipe (4× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) at **TP=4** on **4** DGX Spark (head + 3 workers, dual-rail
RoCEv2) — **the upstream current default config**: fp8 KV + **DFlash2 k=7 block-diffusion
speculative decoding**, **1M context, 3,895,606-token KV pool (3.72× a full 1M request)**:

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` (ACR)
  = baked from the public upstream chain `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
  (day-0 `glm53-flash-arm64-cu130` + the **v1→v9 nine-layer patch stack** + the **DFlash2
  overlay**): NoPE-MLA FA2 SM121 unlock, FlashInfer 0.6.18 nightly, NCCL 2.30.7 /
  cutlass-dsl 4.6.2 re-pins, PDL off, indexer top-k init, fp8 KV smem tile, InstantTensor
  (not enabled), DFlash2 drafter/aux capture/KV-group slot-share. It also bakes in the
  **SM121 `sparse_attn_indexer_kpool` module** (the same file upstream binds at launch — the
  platform cannot distribute host files, so it is baked) and the **mm chat template**.
- Main model (**default**):
  [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) —
  **compressed-tensors** quant, fixes the **intermittent corrupted-token IDs** of the
  ModelOpt builds (vLLM #54150); drop-in, zero flag changes, loads ~2× faster (11 large
  shards vs 120 small). ModelOpt builds remain usable but corrupted: censored
  `LibertAIDAI/GLM-5.3-Flash-NVFP4`, uncensored
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- Draft model: `incoai/GLM-5.3-Flash-DFlash2` (single `model.safetensors` 2.34 GB, qwen3
  arch 5 SWA layers, `block_size=8` / `selector_rank=256` / `target_layer_ids [5,14,24,33,42]`)
- Parallelism: **TP=4** (`--tensor-parallel-size 4`), `mp` backend, one GPU per node
- KV / shapes: **fp8_e4m3**, **24 GiB per rank** (`--kv-cache-memory 25769803776` =
  **3,895,606-token pool**), `--block-size 2304`, gmu 0.85, `--max-num-seqs 6`,
  **`--max-num-batched-tokens 8192`** (unset, vLLM derives 2048 and warns "suboptimal"),
  **1,048,576 context**, port `8000`
- Speculation: `--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  (**must be 7 = block_size−1**); the drafter layers **slot-share the MLA tensors —
  ~zero KV-pool cost**
- Measured (upstream gate suite, 2026-08-29): **54.5 tok/s single-stream** (n=1: 408 tokens
  in 7.5 s, code prompt, temp 0, thinking off; **quote the prompt or the number is
  meaningless** — acceptance is content-driven, ~0.70+ on structured/code vs ~0.33 on
  freeform prose), **4,141.8 tok/s prefill** (warmed, single sample; cold first-prefill
  ~467 tok/s because the kernels JIT); gate = 2× ~41K deep decodes (392/399 decoded tokens)
  + 3× concurrent 32,879-token prefills + vision + `/health` 200 throughout; residual head
  15 GiB, workers 19–20 GiB

> **The unconditional flusher is the whole trick behind this 24 GiB config.** For a week
> upstream believed in "phantom KV backing" above 16 GiB/rank — it was the **page cache**:
> a threshold-triggered flusher can sit below its threshold and still starve the NVRM
> allocator. An unconditional flusher (every node, started before the launcher, running the
> whole boot) took the same 24 GiB pin straight through the gate suite: **+54.8 % pool**
> (2,516,582 → 3,895,606 tokens). **Do not publish without it** (see Deployment notes).

## Quick start (before publishing)

- Cluster: exactly **4** nodes (head + 3 workers), dual-rail RoCEv2 tested.
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` (ACR,
  pushed; the upstream GHCR source is ~31 GiB — let Fireworks pull once and fan out, do not
  pull concurrently from four nodes).
- Models: **both** the main model and the drafter are distributed to the node HF cache via
  `picker=model` variables (drafter only 2.34 GB); containers resolve them offline by repo id
  with `HF_HUB_OFFLINE=1` (`HF_HOME=/cache/huggingface`).
- NCCL: HCA / ifname / GID index filled per node by Fireworks auto keys.
- Boot order: Fireworks publishes workers first, head last.
- **Memory ritual (host side, every node)**: `vm.swappiness=0` (persist via `/etc/sysctl.d/`),
  `swapoff -a && swapon -a`, `sync; echo 3 > /proc/sys/vm/drop_caches`, then
  `setsid nohup ./flusher-unconditional.sh > flusher.log 2>&1 &` (needs passwordless sudo) for
  the **whole boot**, and `pkill -f flusher-unconditional.sh` once serving.
- **Healthy signatures**: the log should show
  `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)` and
  `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`;
  the KV line should read **`GPU KV cache size: 3,895,606 tokens, Maximum concurrency for
  1,048,576 tokens per request: 3.72x`** (yours will differ — see "Measure your own ceiling").

> The upstream Quickstart lists two "must exist on every node" repo files that this recipe
> bakes into the image instead: **(c) the SM121 indexer module**
> (`docker/sparse_attn_indexer_kpool_sm121.py`) and **(d) the mm template**
> (`chat_template_mm.jinja`). The rest (main weights, drafter) are distributed by the platform.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v11-dflash2` | Platform image (nine-layer patch + DFlash2 overlay + SM121 indexer module baked in) |
| `GLM53_MODEL_PATH` | `RedHatAI/GLM-5.3-Flash-NVFP4` | Main model (compressed-tensors default; ModelOpt censored/uncensored drop-ins available but carry token corruption; or absolute snapshot path) |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter** (second model — make sure it is distributed) |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | Served name (drop-in name) |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `1048576` | Model-native 1M; lower (e.g. 300000) for snappier multi-user, keep 64-aligned |
| `MAX_NUM_SEQS` | `6` | Same as Lane A |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Pairs with the pinned `KV_CACHE_MEMORY` |
| `KV_CACHE_MEMORY` | `25769803776` | Per-rank fp8 KV budget (**24 GiB = 3,895,606-token pool**, the current upstream default, gate-passed); assumes the unconditional flusher; gate any bump behind a real long prefill |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Upstream pin; unset, vLLM derives 2048 and warns |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **Must = block_size−1** |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template_mm.jinja` | In-image mm template (baked), so Vision works out of the box; empty = text-only |
| `MASTER_PORT` | `29521` | Distributed master port |

`NODES_TOTAL` (fixed 4), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are auto-filled by Fireworks.

## Publish note (task / project naming)

The task name becomes the Docker Compose project name: only **lowercase letters, digits, `-`
and `_`, no dots** (node Docker Compose v5 hard limit). Dotted names such as `glm5.3-flash-nv`
fail publish with `invalid project name ...` (502). Use a dot-free name like
`glm53-flash-dflash2-4x`.

## Deployment notes (upstream field experience)

- **The unconditional flusher has no substitute**: a threshold-triggered flusher (drop only
  when `Cached > 40 GiB`) can sit below its threshold and still starve the NVRM allocator —
  which is why the same pin booted or OOM'd depending on the moment. Run it on every node,
  started before the launcher, for the whole boot. This is the prerequisite that makes the
  24 GiB default hold.
- **Measure your own ceiling — do not paste ours**: 24 GiB/rank is where the upstream fleet
  lands (head residual 15 GiB, workers 19–20 GiB; other operators run 28/32 GiB on the same
  hardware). The head rank is always the binding constraint (API server + engine core on top
  of its shard), and free memory at startup varies several GiB between identical nodes.
  Ladder up and gate every step: **a config that boots and answers a short prompt is not a
  config that works.**
- **Do not run ModelOpt checkpoints in production** (token corruption, vLLM #54150: a
  corrupted token inside a tool-call block desyncs the `glm47` parser and generation can
  spiral into a repetition lock; same-machine probe U+FFFD 4/9/8 vs RedHatAI 0/0/0). If you
  need uncensored, only the ModelOpt ablit build exists today (corrupted; wait for a
  compressed-tensors abliteration).
- **`num_speculative_tokens` must be block_size−1 = 7**: the drafter is trained for a block
  of 8 and the last position is the target's own verified token; 8 drafts a position the
  model never learned.
- The drafter drafts **text only**: vision requests still work but are not speculated (log
  warns "does not support external multimodal embeddings").
- **Cold JIT warm-up**: the first request JITs `_prepare_dflash_inputs_kernel` and
  `mhc_pre_big_fuse_with_norm_tilelang`; a cold measurement reads ~10 tok/s low — warm first.
- **Healthy signatures**: `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)` and
  `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`; acceptance
  (`spec_decode_num_accepted_tokens_total ÷ num_draft_tokens_total`) ≈ 0.6–0.8 — near 0.15
  means the aux capture or mHC contraction is wrong (silent degrade, no crash). A single-stream
  tok/s figure is a statement about the prompt — ~0.70+ on structured/code, ~0.33 on prose;
  quote the prompt with 54.5.
- **SM121 indexer module (disease 1)**: without it every decode past ~24K context hard-crashes
  (`persistent_topk` smem wall — SM121 has 99 KB/block, `FilteredTopK` needs 128 KB); this
  recipe bakes it into the image (upstream bind-mounts the same file at launch).
- Do not change: `--block-size 2304`, `--moe-backend marlin`, `--kv-cache-dtype fp8_e4m3`,
  `--enforce-eager`, the default `KV_CACHE_MEMORY` (bigger slabs NVRM-OOM under concurrent
  prefills on GB10 — gate every bump behind a real long prefill), the default
  `MAX_NUM_BATCHED_TOKENS` 8192.
- Boot discipline (hard-won rules): tear down **all** ranks before relaunching **any** (a fresh
  rank that rendezvouses with a dying one hangs); verify the **image sha256, not the tag**, on
  every node before every launch (matching tags prove nothing); capture `docker logs` before
  `docker rm -f`; gate with a long prompt **and** a long answer (≥100 tokens) and vary the
  prompt per run or the prefix cache turns the gate into a no-op.
- Self-healing: upstream `fleet_watchdog.sh` (3 consecutive failures → teardown → memory
  ritual → relaunch, workers-first); recovery is ~15 min, so tune thresholds before pointing
  it at a busy endpoint.

## References

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
  the source deployment for this recipe (upstream current default): `launch-glm53-tp4-24g.sh`
  (worker-first 3→2→1→head 0), `flusher-unconditional.sh`, `fleet_watchdog.sh`, `docker/`
  (v1→v9 patch stack), `overlay-dflash2/`, `docs/DFLASH2-SPECULATIVE-DECODING.md`,
  `docs/GB10-KV-MEMORY-LADDER.md`, `docs/SM121-CRASH-FORENSICS-2026-08-27.md` (the gate suite)
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4):
  compressed-tensors NVFP4 main model (default; fixes ModelOpt token corruption)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): DFlash2 draft model
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine (`glm5_next` PR #53906,
  DFlash2 upstream PR #52816, token-corruption issue #54150)
