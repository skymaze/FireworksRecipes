# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，双 rail RoCEv2）上以 **fp8 KV + DFlash2 块扩散
投机解码** 服务 GLM-5.3-Flash（zai-org，320B / A18B MoE），262K 上下文——上游姊妹仓库
的 proven & reproducible 档位（「This is the config to copy」）。

## 模型

- 主模型（默认）：`RedHatAI/GLM-5.3-Flash-NVFP4`（**compressed-tensors**，修复 ModelOpt
  构建的间歇性 token 损坏 vLLM #54150，drop-in、加载快 ~2×；ModelOpt 版仍可用但带损坏）
- 草稿模型：`incoai/GLM-5.3-Flash-DFlash2`（DFlash2 k=7，必须 = block_size−1）
- KV：**fp8_e4m3 · profiler 定池（不传 `--kv-cache-memory`）= 581,040-token 池**（上游验证；
  ⚠️ 完全别 pin，否则激活峰值不扣、首条长 prompt 即死）
- 上下文：**262,144（262K）**（TP=2 每 rank ~97 GiB 权重，勿上 1M；要 1M 用 4x 配方）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`
  （与 TP4 DFlash2 配方同一镜像）

## 速度

实测（上游 2026-08-28，warm）：

- **单流 46.9 tok/s · 74.1% 接受率**（结构化输出 54–61 tok/s）
- C1–C6 并发扫描零失败：聚合 35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) / 47.7
- = **2.15×** 原生 MTP-4（21.8 tok/s）

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），双 rail RoCEv2；
  API 端口默认 `8000`
- 每 rank ~97 GiB 权重；`MAX_NUM_BATCHED_TOKENS` 8192（上游 pin）、`MAX_NUM_SEQS=6`、
  `GPU_MEMORY_UTILIZATION` 0.85
- 主机侧需 `vm.swappiness=0`（强制，重启不保留）

## 参考上游

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)：
  姊妹仓库（TP2 变体，`docs/BENCH-C1-C6-DFLASH2.md` 等）
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) ·
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) ·
  [vllm-project/vllm](https://github.com/vllm-project/vllm)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
