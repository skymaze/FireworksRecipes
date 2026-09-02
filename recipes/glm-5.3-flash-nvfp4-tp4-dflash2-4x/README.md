# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上以 **TP=4** 服务
GLM-5.3-Flash（zai-org，320B / A18B MoE）——**上游当前默认配置**：fp8 KV + **DFlash2
k=7 块扩散投机**，**1M 上下文、3,895,606-token KV 池（3.72× 满 1M 请求）**。

## 模型

- 主模型（默认）：`RedHatAI/GLM-5.3-Flash-NVFP4`（**compressed-tensors**，修复 ModelOpt
  间歇性 token 损坏 vLLM #54150；drop-in、零 flag 改动、加载快 ~2×）
- 草稿模型：`incoai/GLM-5.3-Flash-DFlash2`（2.34 GB；k=7 必须 = block_size−1；drafter 层与
  MLA 张量 slot-share，**KV 池成本 ~0**）
- KV/形状：**fp8_e4m3 · 每 rank 24 GiB 预算 = 3,895,606-token 池**（`--block-size 2304`、
  gmu 0.85、`--max-num-seqs 6`、`--max-num-batched-tokens 8192`）
- 上下文：**1,048,576（1M）**；API 端口默认 `8000`
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`
  （ACR；v1→v9 九层 patch + DFlash2 overlay + SM121 indexer 模块已烘焙）

## 速度

实测（上游 2026-08-29 gate 套件，warmed）：

- **单流 54.5 tok/s**（n=1：408 tokens/7.5 s，code prompt，temp 0，thinking off；引用前请带
  prompt——接受率由内容决定，结构化/代码 ~0.70+ vs 自由文本 ~0.33）
- **prefill 4,141.8 tok/s**（warmed；冷首 prefill ~467 tok/s，核 JIT）
- gate = 2× ~41K 深度解码 + 3× 并发 32,879-token prefill + vision + `/health` 全程 200；
  余量 head 15 GiB / worker 19–20 GiB

## 硬件需求

- **4 台** DGX Spark（固定 4 节点 · TP=4），每机 1 GPU（GB10），双 rail RoCEv2
- 每 rank 24 GiB fp8 KV 预算；**发布前每节点必须跑「无条件 flusher」**（启动前起、全程
  跑，`flusher-unconditional.sh`，需无密码 sudo）——这是 24 GiB 档位通过 gate 的全部戏法
- 测你自己的上限：head 永远是最紧约束（API server + 引擎核心在 shard 之上）

## 参考上游

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：
  源部署（上游当前默认）：`launch-glm53-tp4-24g.sh`、`flusher-unconditional.sh`、
  `fleet_watchdog.sh`、`docker/` patch 栈、`overlay-dflash2/`、gate 套件文档
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) ·
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) ·
  [vllm-project/vllm](https://github.com/vllm-project/vllm)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
