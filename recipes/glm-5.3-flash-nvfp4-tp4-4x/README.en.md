# GLM-5.3-Flash NVFP4 · TP=4 · Lane A (fp8 KV) · Fireworks recipe (4× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE, `glm5_next`) at **TP=4** on **4** DGX Spark (head + 3 workers, dual-rail
RoCEv2) — the first **fp8 KV cache for a NoPE-MLA model on consumer Blackwell**. This recipe
is the upstream deployment's **Lane A — fp8 KV** (FlashInfer SM12x unlock, the
speed-oriented lane) with **native MTP k=4** speculation.

> **Upstream has marked MTP TP4 as superseded**: the current default is DFlash2 (faster,
> ~zero KV-pool cost); the only matched TP4 comparison ever run was at TP2 (DFlash2 46.9 vs
> MTP-4 21.8 tok/s = 2.15×). **For new deployments use the
> [DFlash2 recipe](../glm-5.3-flash-nvfp4-tp4-dflash2-4x/README.en.md) instead**; this recipe
> stays for existing MTP installs or people deliberately avoiding the drafter.

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` = vLLM day-0
  `glm53-flash-arm64-cu130` + the **8-layer patch stack** baked from the upstream chain
  (pushed to ACR with this recipe):
- Model (**default**): `RedHatAI/GLM-5.3-Flash-NVFP4` (**compressed-tensors**, drop-in with
  zero flag changes, loads ~2× faster: fixes the intermittent token corruption of the ModelOpt
  builds, vLLM #54150); ModelOpt builds remain usable but corrupted: censored
  `LibertAIDAI/GLM-5.3-Flash-NVFP4` (120 shards ~182 GiB), uncensored
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- Parallelism: **TP=4** (`--tensor-parallel-size 4`), `mp` backend, one GPU per node
- KV: **fp8_e4m3** (the current 512 B/token/layer NoPE record), **24 GiB per rank**
  (`--kv-cache-memory 25769803776` = 3,895,606-token pool, gate-passed; the 5.03M pool was
  the historical 32 GiB/rank comparison, superseded), `--block-size 2304`, gmu 0.85
- Speculation: **native MTP k=4**; tool calling on (`glm47`), thinking off by default
- Context: **1,048,576 (1M)**; API port `8000`
- Measured (upstream 2026-08-27, TP4, warmed): structured/agentic decode **~55 tok/s**,
  prefill ~3,530 tok/s · TTFT ~0.2 s
- **New publish prerequisite**: the 24 GiB KV tier only passes gates with the **unconditional
  flusher** on every node for the whole boot — see Deployment notes.

> Upstream now splits the TP4 deployment by KV dtype: **fp8 KV is the daily driver** (this
> recipe) vs **NVFP4 KV Lane B** (b12x path, 368 B/token/layer, 6,652,112-token pool at 32 GiB,
> 1.32×, but ~33 % slower decode; `--enforce-eager` required). Pick fp8 for everyday agentic
> production.

## Quick start (before publishing)

- Cluster: exactly **4** nodes (head + 3 workers), dual-rail RoCEv2 tested.
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` (Aliyun ACR;
  Fireworks pulls and distributes). The upstream author's `radixark/vllm-glm53-flash:sm121-v8`
  is only a local build-chain tag and was never pushed publicly, so this recipe bakes the image
  from the upstream chain and pushes it to ACR.
- Model (default): `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors, drop-in, loads ~2×
  faster, fixes ModelOpt token corruption); or the ModelOpt drop-ins:
  `LibertAIDAI/GLM-5.3-Flash-NVFP4` (HF hub layout `models--LibertAIDAI--GLM-5.3-Flash-NVFP4`,
  120 shards ~182 GiB, censored) and
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock` (uncensored, verified on the same
  launcher — but both ModelOpt builds carry token corruption), distributed to the node cache;
  in-container `HF_HUB_OFFLINE=1` resolves by repo id (`HF_HOME=/cache/huggingface`).
- NCCL: HCA / ifname / GID index are all filled per node by Fireworks auto keys. The upstream
  launcher hard-codes `gid_index=3`; this recipe resolves it per node (no drift across reboots).
- Boot order: Fireworks publishes workers first, head last (upstream hard-won rule #1).

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `GLM53_IMAGE` | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` | Platform image (upstream 8-layer patch stack baked in) |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | Model (offline HF-hub resolution; or absolute snapshot path) |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | Served name (drop-in name) |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `1048576` | Model-native 1M; lower (e.g. 300000) for snappier multi-user, keep 64-aligned |
| `MAX_NUM_SEQS` | `6` | Upstream launcher production value |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Pairs with the pinned `KV_CACHE_MEMORY` |
| `KV_CACHE_MEMORY` | `25769803776` | Per-rank fp8 KV budget (24 GiB = 3,895,606-token pool, the current upstream default, gate-passed); guards the GB10 UMA OOM line; gate any bump behind a real long prefill |
| `MTP_NUM_SPECULATIVE_TOKENS` | `4` | Native MTP k; k=3 is a candidate micro-tune |
| `CHAT_TEMPLATE` | (empty) | In-container template path; set to the mm template to enable Vision |
| `MASTER_PORT` | `29521` | Distributed master port |

`NODES_TOTAL` (fixed 4), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_IB_*`
are all auto-filled by Fireworks.

## Publish note (task / project naming)

The **task name** you type when creating the task in Fireworks becomes the Docker Compose
project name, passed verbatim to each node agent's `/api/compose/up`. The node-side
**Docker Compose v5 only allows project names made of lowercase alphanumerics, hyphens and
underscores — no dots (`.`)**. A task name containing a dot from `5.3` (e.g. `glm5.3-flash-nv`)
makes `compose up` fail immediately with `invalid project name ... must consist only of
lowercase alphanumeric characters, hyphens, and underscores` — surfaced by the control plane
as a **502 Bad Gateway** on `http://<node>:9000/api/compose/up`.

Use a **dot-free task name** when publishing, e.g. `glm53-flash-nv`, `glm53-flash-tp4-4x`;
do **not** use `glm5.3-*`.

## Deployment notes (from upstream field experience)

- **The unconditional flusher (prerequisite for the 24 GiB tier)**: 24 GiB/rank only passes
  gates with the **unconditional flusher** (`flusher-unconditional.sh`) running on every node,
  started before the launcher, for the whole boot (needs passwordless sudo). The week-long
  "phantom KV backing" misdiagnosis was threshold-triggered page-cache flushing starving the
  NVRM allocator. Before publishing, every node also needs the memory ritual:
  `vm.swappiness=0` (persist), `swapoff -a && swapon -a`, `sync; echo 3 > /proc/sys/vm/drop_caches`.
  Measure your own ceiling — the head rank is always the binding constraint.
- **Do not run ModelOpt checkpoints in production** (token corruption, vLLM #54150 — a
  corrupted token inside a tool-call block desyncs the parser and generation can spiral into a
  repetition lock); if you need uncensored, only the ModelOpt ablit build exists today (corrupted).
- **Cache-hit visibility**: `--enable-prefix-caching` (on by default for hybrid models) +
  `--enable-prompt-tokens-details`; the API reports `usage.prompt_tokens_details.cached_tokens`
  so you can verify prefix-cache hits (deep-session/agentic ~100×; compare a second request
  with the same prefix).
Do **not** change: `--block-size 2304` (DeepGEMM arch-12 kpool page rule; 2176 dies),
  `--moe-backend marlin`, `--kv-cache-dtype fp8_e4m3`, `--enforce-eager` (required by the
  b12x/fp8 path; it also caps single-stream decode), and the default `--kv-cache-memory`
  (bigger slabs NVRM-OOM under concurrent prefills on GB10 — gate every bump behind a real
  long prefill).
- Boot discipline (upstream hard-won rules): tear down all ranks before relaunching any; verify
  the `IMAGE` matches on every node before every launch; capture `docker logs` before `rm -f`.
- Context alignment: keep `MAX_MODEL_LEN` 64-aligned (kpool·64 / MLA alignment rules).
- Thinking is off by default (`--default-chat-template-kwargs '{"enable_thinking": false}'`);
  re-enable per request with `chat_template_kwargs: {"enable_thinking": true}`. With thinking
  on, `max_tokens` includes reasoning tokens.
- **Vision (image input)**: the checkpoint ships a text-only chat template, so image requests
  500 without the mm variant. To enable Vision, place `chat_template_mm.jinja` from the
  upstream repo into the node cache and set `CHAT_TEMPLATE` to its in-container path (e.g.
  `/cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/<hash>/chat_template_mm.jinja`).
- InstantTensor (v9) is **not** enabled: ranks die silently ~1 min after load in multi-node;
  this recipe stays on the v8 image / standard loader.

## References

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
  the source deployment (Lane A) with `launch-glm53-tp4-24g.sh` (renamed from
  `launch-glm53-vllm-tp4.sh`), `flusher-unconditional.sh`, the `docker/` patch stack,
  `docs/DEPLOY-REPORT.md`, `docs/GB10-KV-MEMORY-LADDER.md`, and
  `docs/SM121-CRASH-FORENSICS-2026-08-27.md`
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4): compressed-tensors NVFP4 main model (default)
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): base model
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine (`glm5_next` support PR #53906)
