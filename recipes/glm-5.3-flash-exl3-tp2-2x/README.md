# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 1M · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，CX7 直连）上以 **EXL3/TR3 路线** 服务
GLM-5.3-Flash（zai-org，320B / A18B MoE），1M 上下文。与仓库内 NVFP4（marlin）配方为
不同镜像、不同 lane。

## 模型

- 主模型：**EXL3/TR3 4bpw**——`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`（brandonmusic
  快照的公开镜像；4bpw 在 KLD 上与官方 FP8 持平 ~1.00×，仅 54% 字节）
- KV：**fp8 · packed `fp8_ds_mla`**（勿用 bf16/nvfp4）
- 投机：**DFlash2 k=7**（`incoai/GLM-5.3-Flash-DFlash2`，drafter 跨 TP 分片，
  `DFLASH_DRAFT_TP=2`）
- 上下文：**1,000,000**；Vision 默认开（image×4 / video×1）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.1.0`
  （ACR 按上游现行 Dockerfile 烘焙，@c707598；内含 E2 fat-expert 内核 `EXL3_FAT_KERNEL=1`）
- ⚠️ `MAX_NUM_BATCHED_TOKENS` 默认 7168，**8192 会撑爆 GB10 indexer smem，永远别上**

## 速度

实测（上游 2026-08-28 sparkDash decode bench；DFlash2 k=7 · 结构化/代码高接受档 ·
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
- 1M 服务：util 0.87 时池 **1,754,237 token（18.67 GiB，1.75×）**；1M 至少需 ~14.5 GiB KV
  ——本机 KV 偏小先核对镜像为 v1.1.0 再调 util ≥0.90
- `MAX_NUM_SEQS=4`（上游 pin）；`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 保持 1

## 参考上游

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)：
  参数级参考（.env / start.sh / overlay / Dockerfile）
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)（ShapleyMCG）
  · [原始权重](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC BY-NC-ND 4.0）
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
