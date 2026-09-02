# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks recipe (4× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE) at **TP=4** on **4** DGX Spark nodes (head + 3 workers, dual-rail RoCEv2)
— **the upstream current default config**: fp8 KV + **DFlash2 k=7 block-diffusion
speculation**, **1M context, 3,895,606-token KV pool (3.72× a full 1M request)**.

## Model

- Main model (default): `RedHatAI/GLM-5.3-Flash-NVFP4` (**compressed-tensors**, fixes the
  intermittent corrupted-token IDs of the ModelOpt builds, vLLM #54150; drop-in, zero flag
  changes, loads ~2× faster)
- Draft model: `incoai/GLM-5.3-Flash-DFlash2` (2.34 GB; k=7 must equal block_size−1; the
  drafter layers slot-share the MLA tensors — **~zero KV-pool cost**)
- KV/shapes: **fp8_e4m3 · 24 GiB per rank = 3,895,606-token pool** (`--block-size 2304`,
  gmu 0.85, `--max-num-seqs 6`, `--max-num-batched-tokens 8192`)
- Context: **1,048,576 (1M)**; API port `8000`
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`
  (ACR; nine-layer patch stack + DFlash2 overlay + SM121 indexer module baked in)

## Speed

Measured (upstream gate suite, 2026-08-29, warmed):

- **54.5 tok/s single-stream** (n=1: 408 tokens in 7.5 s, code prompt, temp 0, thinking
  off; **quote the prompt or the number is meaningless** — acceptance is content-driven,
  ~0.70+ on structured/code vs ~0.33 on prose)
- **4,141.8 tok/s prefill** (warmed; cold first prefill ~467 tok/s due to kernel JIT)
- Gate: 2× ~41K deep decodes + 3× concurrent 32,879-token prefills + vision + `/health`
  200 throughout; residual head 15 GiB, workers 19–20 GiB

## Hardware requirements

- **4** DGX Spark nodes (fixed 4 nodes · TP=4), one GB10 GPU each, dual-rail RoCEv2
- 24 GiB fp8 KV per rank; **the "unconditional flusher" is required on every node**
  (started before the launcher, running the whole boot, `flusher-unconditional.sh`,
  passwordless sudo) — it is the whole trick behind the 24 GiB tier
- Measure your own ceiling — the head rank is always the binding constraint (API server +
  engine core on top of its shard)

## Upstream references

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark):
  the source deployment (upstream current default): `launch-glm53-tp4-24g.sh`,
  `flusher-unconditional.sh`, `fleet_watchdog.sh`, the `docker/` patch stack,
  `overlay-dflash2/`, the gate-suite docs
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) ·
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) ·
  [vllm-project/vllm](https://github.com/vllm-project/vllm)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
