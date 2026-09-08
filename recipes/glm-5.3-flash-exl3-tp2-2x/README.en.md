# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 850k · Fireworks recipe (2× DGX Spark)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B
MoE) on the **EXL3/TR3 lane** with **2** DGX Spark (head + 1 worker, direct CX7), at 850k
context — a different image and lane from the NVFP4 (marlin) recipes in this repo.

## Model

- Weights: **EXL3/TR3 4bpw** — `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (public mirror of the
  brandonmusic snapshot; 4bpw matches official FP8 KLD ~1.00× at only 54% of the bytes)
- KV: **fp8 · packed `fp8_ds_mla`** (not bf16/nvfp4)
- Speculation: **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`, drafter sharded across TP,
  `DFLASH_DRAFT_TP=2`)
- Context: **850,000** (upstream 2026-09-07 shipped tier: 850k / util 0.85 / rightsize;
  1M returns only with `EXL3_FAT_GROUPED=0` (E2) at util ≥0.87); Vision on by default (image×4 / video×1)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.2.0`
  (ACR-baked from the upstream Dockerfile @6599585; includes the **E3 grouped fat-MoE
  kernels** `exl3_fat_moe_ext`, `EXL3_FAT_GROUPED=1` default)
- ⚠️ `MAX_NUM_BATCHED_TOKENS` defaults to 7168 — **8192 blows the GB10 indexer smem,
  never use it**

## Speed

**Cold prefill** (upstream 2026-09-07 sparkDash bench, E3 · `EXL3_FAT_GROUPED=1` ·
`MAX_MODEL_LEN=900000` · `GPU_MEM_UTIL=0.86` · rightsize · `EXL3_TEMP_ROWS_FUSED=32` ·
MNBT 7168 · `MAX_NUM_SEQS=4` · DFlash2 k=7 draft TP=2 · C4 · thinking off):

| Prompt | Prompt tok | TTFT | Prefill tok/s |
|---|---:|---:|---:|
| ~8k | 8,221 | 5.51 s | **1,492.1** |
| ~16k | 16,411 | 10.56 s | **1,553.7** |
| ~32k | 32,797 | 22.96 s | **1,428.2** |
| ~64k | 65,566 | 41.31 s | **1,587.0** |
| ~128k | 131,101 | 83.95 s | **1,561.7** |
| ~256k | 262,173 | 172.84 s | **1,516.8** |

+37–45% cold prefill over E2 (`EXL3_FAT_GROUPED=0`); E2 vs E3 stay identical to the
LinearEXL3 reference numerically.

**Decode** (upstream 2026-08-28, sparkDash decode bench; DFlash2 k=7 · structured/code
high-accept regime · temp 0 · thinking off · 400 tokens):

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

- Lab: Structured **61.7** tok/s (0.918 accept), Prose 26.9, long-context (~60–100K KV)
  24–27; with `DFLASH_DRAFT_TP=2`, structured **65.1**; MTP k=2 baseline ~24.6
- Block-aligned prefix cache: a ~7.7k follow-up hit 93%, TTFT 9.7 s → 1.17 s

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, **direct CX7 cabling**
  (NCCL cannot use loopback aliases)
- 850k/0.85 boots with ~0.5 GiB of KV margin (KV pool ~989k tokens measured at 900k / 0.85;
  E3 keeps a ~560 MiB fat-row scratch charged to the KV budget) — if your boot shows a small
  pool, first verify the image is v1.2.0, then raise util to ≥0.86; the 500k tier can run 0.84
- `MAX_NUM_SEQS=4` (upstream pin); keep `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` at 1;
  `EXL3_TEMP_ROWS_FUSED` 32 must be ≥ `MAX_NUM_SEQS×(DFLASH_TOKENS+1)` (=32) to keep decode
  a single graph-safe launch

## Upstream references

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):
  parameter-level reference (.env / start.sh / overlay / Dockerfile; E3 became the launcher
  default on 2026-09-07)
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) (ShapleyMCG)
  · [original weights](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0)
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
