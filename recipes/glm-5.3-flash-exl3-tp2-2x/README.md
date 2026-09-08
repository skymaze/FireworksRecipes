# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 850k · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，CX7 直连）上以 **EXL3/TR3 路线** 服务
GLM-5.3-Flash（zai-org，320B / A18B MoE），850k 上下文。与仓库内 NVFP4（marlin）配方为
不同镜像、不同 lane。

## 模型

- 主模型：**EXL3/TR3 4bpw**——`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`（brandonmusic
  快照的公开镜像；4bpw 在 KLD 上与官方 FP8 持平 ~1.00×，仅 54% 字节）
- KV：**fp8 · packed `fp8_ds_mla`**（勿用 bf16/nvfp4）
- 投机：**DFlash2 k=7**（`incoai/GLM-5.3-Flash-DFlash2`，drafter 跨 TP 分片，
  `DFLASH_DRAFT_TP=2`）
- 上下文：**850,000**（上游 2026-09-07 shipped 档：850k / util 0.85 / rightsize；
  1M 仅 `EXL3_FAT_GROUPED=0`（E2）且在 util ≥0.87 时可回）；Vision 默认开（image×4 / video×1）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.2.0`
  （ACR 按上游现行 Dockerfile 烘焙，@6599585；内含 **E3 grouped fat-MoE 内核**
  `exl3_fat_moe_ext`，`EXL3_FAT_GROUPED=1` 默认）
- ⚠️ `MAX_NUM_BATCHED_TOKENS` 默认 7168，**8192 会撑爆 GB10 indexer smem，永远别上**

## 速度

**冷 prefill**（上游 2026-09-07 sparkDash bench，E3 · `EXL3_FAT_GROUPED=1` ·
`MAX_MODEL_LEN=900000` · `GPU_MEM_UTIL=0.86` · rightsize · `EXL3_TEMP_ROWS_FUSED=32` ·
MNBT 7168 · `MAX_NUM_SEQS=4` · DFlash2 k=7 draft TP=2 · C4 · thinking off）：

| Prompt | Prompt tok | TTFT | Prefill tok/s |
|---|---:|---:|---:|
| ~8k | 8,221 | 5.51 s | **1,492.1** |
| ~16k | 16,411 | 10.56 s | **1,553.7** |
| ~32k | 32,797 | 22.96 s | **1,428.2** |
| ~64k | 65,566 | 41.31 s | **1,587.0** |
| ~128k | 131,101 | 83.95 s | **1,561.7** |
| ~256k | 262,173 | 172.84 s | **1,516.8** |

较 E2（`EXL3_FAT_GROUPED=0`）冷 prefill **+37–45%**（E2 vs E3 数值与 LinearEXL3 参考一致）。

**Decode**（上游 2026-08-28 sparkDash bench；DFlash2 k=7 · 结构化/代码高接受档 ·
temp 0 · thinking off · 400 tokens）：

| 并发 | TTFT | 单流 tok/s | 聚合 tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

- 实验室：Structured **61.7** tok/s（0.918 accept）、Prose 26.9、长上下文（~60–100K KV）
  24–27；`DFLASH_DRAFT_TP=2` 后 structured **65.1**；MTP k=2 基线 ~24.6
- 前缀缓存 block-aligned：~7.7k 后续轮 93% 命中，TTFT 9.7 s → 1.17 s

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），**CX7 直连**
  （NCCL 不能走 loopback 别名）
- 850k/0.85 boot 带 ~0.5 GiB KV 余量（KV 池 ~989k token / 900k 0.85 实测；
  E3 常驻 ~560 MiB fat-row scratch 计入 KV 预算）——本机 KV 偏小先核对镜像为 v1.2.0
  再调 util ≥0.86；500k 档可 util 0.84
- `MAX_NUM_SEQS=4`（上游 pin）；`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 保持 1；
  `EXL3_TEMP_ROWS_FUSED` 32 需 ≥ `MAX_NUM_SEQS×(DFLASH_TOKENS+1)`（=32）保持 decode 单图

## 参考上游

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)：
  参数级参考（.env / start.sh / overlay / Dockerfile；E3 于 2026-09-07 成为 launcher 默认）
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)（ShapleyMCG）
  · [原始权重](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC BY-NC-ND 4.0）
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
