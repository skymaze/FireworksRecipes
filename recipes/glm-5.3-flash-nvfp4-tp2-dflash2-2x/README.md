# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，双 rail RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next`）的 **fp8 KV + DFlash2 块扩散投机解码** 服务——**上游盖章的
「the one to copy」配置**（姊妹仓库 `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`
的 proven & reproducible 档位）。

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2`
  （与 TP4 DFlash2 配方同镜像：sm121-v8 + 4-patch DFlash2 overlay）
- 双模型（都按 HF repo id 分发到节点缓存、离线解析）：
  - 主模型 `LibertAIDAI/GLM-5.3-Flash-NVFP4`（120 分片 ~182 GiB）
  - **DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2`**（2.34 GB）
- 拓扑：**2 节点 · TP=2**（`--tensor-parallel-size 2`）、`mp` 后端
- 上下文：**262,144（262K）**；KV：**fp8_e4m3 · `--kv-cache-memory 3221225472`
  （3 GiB → 310,292-token 池）**
- 投机：`--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  （必须 7 = block_size−1）
- 实测（上游 2026-08-28，warm）：**单流 46.9 tok/s · 74.1% 接受率**；C1–C6 并发扫描
  **零失败**：聚合 35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) / 47.7

## 快速开始（发布前就绪）

- 集群：恰好 **2 台**节点（head + 1 worker），双 rail RoCEv2 已配置测试。
- 镜像：`…/glm53-flash-sm121:v8-dflash2`（ACR，已推送；无需再构建）。
- 模型：主模型 + drafter 两个都由 Fireworks 以 `picker=model` 变量分发到节点 HF 缓存。
- **NCCL**：HCA / 网卡 / GID index 由 Fireworks 自动键按节点填充。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v8-dflash2` | 平台镜像（8 层 patch + DFlash2 overlay 已烘焙） |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 主模型 |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（务必分发到节点缓存） |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | 对外服务名 |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `262144` | **上游已验证档位**（TP2 每 rank ~97 GiB 权重，勿上 1M） |
| `MAX_NUM_SEQS` | `6` | 与上游一致 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 与固定 `KV_CACHE_MEMORY` 搭配 |
| `KV_CACHE_MEMORY` | `3221225472` | **3 GiB pin**（上游并发扫描验证；4.1 GiB 会在 3×20K prefill 时 OOM） |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **必须 = block_size−1** |
| `CHAT_TEMPLATE` | （空） | 容器内模板路径；填 mm 模板可开 Vision |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意（任务/项目命名）

任务名即 docker compose 项目名：只允许**小写字母/数字/`-`/`_`，不能含点 `.`**
（节点 Docker Compose v5 硬性限制）。带点任务名会发布失败（502）。建议如 `glm53-dflash2-tp2`。

## 部署注意（源自源仓库实机踩坑）

- **缓存命中可见**：已开 `--enable-prefix-caching`（hybrid 模型默认开启）与
  `--enable-prompt-tokens-details`；API 返回 `usage.prompt_tokens_details.cached_tokens`
  可核对前缀缓存命中 token 数（深会话/agentic 场景命中率 ~100×，同前缀二次请求比对）。
**TP2 每 rank 约 97 GiB 权重，GB10 内存余量是硬约束**：不要调大 `KV_CACHE_MEMORY`
  （上游 ladder：任何 >4.14 GiB 的 KV 都在某些 boot 上 NVRM OOM；3 GiB 是并发验证过的
  shipping pin）。启动期权重大量吃 page cache 时，上游需要 `cache_flusher` sidecar
  顶着——Fireworks 侧未跑该 sidecar，故更应保持保守 pin。
- `num_speculative_tokens` 必须 = 7（block_size−1）。
- drafter 只做文本草稿（vision 请求仍可用、不投机）。
- 冷启动 JIT 需预热；健康签名 `Using Eagle3 auxiliary layers … (6,15,25,34,43)`、
  acceptance 0.6–0.8（~0.15 = aux 捕获/mHC 收缩错了，静默退化）。
- 起停纪律：先全部 teardown 再重启任一 rank；发布核对各节点 `IMAGE` 一致。
- C1–C6 是上游 TP2 实测；Fireworks 侧（无 flusher）建议真机复测后回填自己的数字。

## 参考来源

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)：
  姊妹仓库（TP2 变体）——`docs/BENCH-C1-C6-DFLASH2.md` 的 TP2 表、`docs/GB10-KV-MEMORY-LADDER.md`
- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：4x 主仓库
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿模型
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：NVFP4 量化主模型
