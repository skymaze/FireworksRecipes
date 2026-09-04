# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks recipe (2× DGX Spark)

Serve **GLM-5.3-Flash** ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
320B / A18B MoE) on **2** DGX Spark (head + 1 worker, dual-rail RoCEv2) with **fp8 KV +
DFlash2 block-diffusion speculation** at 262K context — the upstream sibling repo's
proven-&-reproducible tier ("This is the config to copy").

## Model

- Main model (default): `RedHatAI/GLM-5.3-Flash-NVFP4` (**compressed-tensors**; fixes the
  intermittent token corruption of the ModelOpt builds, vLLM #54150; drop-in, loads ~2×
  faster; ModelOpt builds remain usable but corrupted)
- Draft model: `incoai/GLM-5.3-Flash-DFlash2` (DFlash2 k=7 — must equal block_size−1)
- KV: **fp8_e4m3 · pinned 6 GiB (`--kv-cache-memory 6442450944`) = 678,661-token pool**
  (upstream 2026-09-02 launcher pin: 6 preemptions → 0 under load; ⚠️ never go above 6 GiB —
  the activation peak is not deducted and the first long prompt kills the engine; empty =
  profiler-sized, the old 581,040 tier)
- Speculation findings (upstream 09-02): k=5 inverts vs k=7 at **C4** (default stays k=7; deep
  concurrency can swap in a k=5 schedule); **a higher acceptance RATIO is not throughput** —
  track the mean accepted length (fell 4.84→4.15 at k=5); **CUDA graphs are flat on TP2**
  (−0.3% aggregate), so keep `--enforce-eager`
- Context: **262,144 (262K)** (TP=2 carries ~97 GiB weights/rank — no 1M here; use the 4x
  recipe for 1M)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`
  (the same image the TP4 DFlash2 recipe uses)

## Speed

Measured (upstream 2026-08-28, warm):

- **46.9 tok/s single-stream · 74.1% draft acceptance** (structured output 54–61 tok/s)
- C1–C6 concurrency sweep, zero failures: aggregate 35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) / 47.7
- = **2.15×** over native MTP-4 (21.8 tok/s)

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, dual-rail RoCEv2;
  API port defaults to `8000`
- ~97 GiB weights per rank; `MAX_NUM_BATCHED_TOKENS` 8192 (upstream pin),
  `MAX_NUM_SEQS=6`, `GPU_MEMORY_UTILIZATION` 0.85
- Host-side `vm.swappiness=0` is mandatory (does not survive a reboot)

## Upstream references

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark):
  the sibling (TP2) repo (`docs/BENCH-C1-C6-DFLASH2.md`, etc.)
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) ·
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) ·
  [vllm-project/vllm](https://github.com/vllm-project/vllm)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
