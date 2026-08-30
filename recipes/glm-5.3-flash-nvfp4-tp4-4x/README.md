# GLM-5.3-Flash NVFP4 · TP=4 · Lane A（fp8 KV）· Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next` 架构）的 **NVIDIA consumer Blackwell 首款 NoPE-MLA fp8 KV** 的 **TP=4** 服务；
本配方对应上游部署的 **Lane A — fp8 KV**（FlashInfer SM12x unlock，就速度优化的一路），
采用**原生 MTP k=4** 投机。

> **上游已把 MTP TP4 标为 superseded**：当前默认是 DFlash2（更快、KV 池成本 ~0），TP4
> 匹配对比只做了 TP2（DFlash2 46.9 vs MTP-4 21.8 tok/s = 2.15×）。**新部署请直接改用
> [DFlash2 配方](../glm-5.3-flash-nvfp4-tp4-dflash2-4x/README.md)**；本配方保留给已在跑的
> MTP 存量或想避开 drafter 的场景。

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` = vLLM day-0
  `glm53-flash-arm64-cu130` + 按上游 chain 烘焙的 **8 层 patch 栈**（本配方已推送 ACR）：
- 模型（**默认**）：`RedHatAI/GLM-5.3-Flash-NVFP4`（**compressed-tensors**，drop-in 无 flag
  改动、加载快 ~2×：修复 ModelOpt 构建的**间歇性 token 损坏**，vLLM #54150）；ModelOpt 版
  仍可用但带损坏：censored `LibertAIDAI/GLM-5.3-Flash-NVFP4`（120 分片 ~182 GiB）、
  uncensored `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- 并行：**TP=4**（`--tensor-parallel-size 4`）、`mp` 后端、4 台每机 1 GPU
- KV：**fp8_e4m3**（上游当前 512 B/token/layer NoPE 记录）、每 rank **24 GiB 预算**
  （`--kv-cache-memory 25769803776` = 3,895,606-token 池，gate-passed；5.03M pool 是
  32 GiB/rank 的历史对比值，已 superseded）、`--block-size 2304`、gmu 0.85
- 投机：**原生 MTP k=4**；工具调用 `glm47` parser 开启、思考默认关（请求级可开）
- 上下文：**1,048,576（1M）**；端口默认 `8000`
- 实测（上游 2026-08-27，TP4，warmed）：结构化解码 ~55 tok/s、prefill ~3,530 tok/s · TTFT ~0.2s
- **发布前提（新增）**：24 GiB KV 档位在**无条件 flusher**（每节点、全程）下才通过 gate——
  见部署注意。

> 上游仓库把当前 TP4 部署分为 fp8/NVFP4 KV 两路：**fp8 KV 是每日主力**（本配方）；
> **NVFP4 KV Lane B**（b12x 路径，368 B/token/layer，32 GiB 下 6,652,112-token 池、1.32×，
> 但解码慢 ~33%，`--enforce-eager` 必需）。日常 agentic 生产选 fp8 路即可。

## 快速开始（发布前就绪）

- 集群：**4 台**节点（head + 3 worker），双 rail RoCEv2 已配置测试。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8`（阿里云 ACR；
  Fireworks 拉取后分发到节点）。上游作者的 `radixark/vllm-glm53-flash:sm121-v8` 仅为本地
  构建链 tag、未公开推送，故本配方镜像按上游 chain 自行烘焙后推送 ACR。
- 模型（默认）：`RedHatAI/GLM-5.3-Flash-NVFP4`（**compressed-tensors**，drop-in、加载快
  ~2×、修复 ModelOpt token 损坏）；或 ModelOpt drop-in：`LibertAIDAI/GLM-5.3-Flash-NVFP4`
  （HF hub 布局 `models--LibertAIDAI--GLM-5.3-Flash-NVFP4`，120 分片 ~182 GiB，censored）、
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`（uncensored，同 launcher 已验证，
  但 ModelOpt 版带 token 损坏）。分发到节点缓存；容器内 `HF_HUB_OFFLINE=1` 按 repo id 离线解析
  （`HF_HOME=/cache/huggingface`）。
- **NCCL**：HCA / 网卡 / GID index 全部由 Fireworks 自动键按节点填充；源 launcher 硬编码
  `gid_index=3`，本配方按节点动态解析（重启不漂移）。
- **启动顺序**：Fireworks 发布时自动 worker-first、head 最后（源 hard-won rule #1）。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_IMAGE` | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` | 平台镜像（上游 8 层 patch 栈已烘焙） |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 模型（HF hub 离线解析；或填缓存内 snapshot 绝对路径） |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | 对外服务名（drop-in 名） |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 模型原生 1M；降低（如 300000）更跟手，须 64 对齐 |
| `MAX_NUM_SEQS` | `6` | 源 launcher 生产值 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 与固定 `KV_CACHE_MEMORY` 搭配 |
| `KV_CACHE_MEMORY` | `25769803776` | 每 rank fp8 KV 预算（24 GiB = 3,895,606-token 池，上游当前默认、gate-passed）；防 GB10 UMA OOM；加大必须真实长 prefill 把关 |
| `MTP_NUM_SPECULATIVE_TOKENS` | `4` | 原生 MTP 投机 k；k=3 是可尝试的微调 |
| `CHAT_TEMPLATE` | （空） | 容器内模板路径；填 mm 模板可开 Vision（见下） |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 4）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意（任务/项目命名）

Fireworks 建任务时填的**任务名即 docker compose 项目名**，会原样传给节点 agent 的
`/api/compose/up`。节点上的 **Docker Compose v5 只允许项目名由小写字母/数字/`-`/`_`
组成、不能含点（`.`）**——任务名里若带 `5.3` 这类点（如 `glm5.3-flash-nv`），agent 的
`compose up` 会在任何拉镜像/启动之前直接报 `invalid project name ... must consist only of
lowercase alphanumeric characters, hyphens, and underscores`，控制面表现为 **502 Bad Gateway**
（`http://<node>:9000/api/compose/up`）。

请使用**无点任务名**再发布，例如 `glm53-flash-nv`、`glm53-flash-tp4-4x`；**不要**用
`glm5.3-*`。

## 部署注意（源自源仓库实机踩坑）

- **无条件 flusher（24 GiB 档位的前提）**：24 GiB/rank 是上游在**无条件 flusher**（每节点、
  启动前起、全程跑，`flusher-unconditional.sh`，需无密码 sudo）下才过 gate 的档位；曾经
  10 天的 "phantom KV backing" 误诊实为阈值触发式 page-cache 刷新在饿 NVRM 分配器。发布前
  每节点还要做内存仪式：`vm.swappiness=0`（持久化）、`swapoff -a && swapon -a`、
  `sync; echo 3 > /proc/sys/vm/drop_caches`。测你自己的上限：head 永远是最紧约束。
- **checkpoint 别用 ModelOpt 版做生产**（token 损坏，vLLM #54150——工具调用块内坏 token 让
  parser 失步、生成旋进重复锁）；要 uncensored 目前只有 ModelOpt ablit 可选（带损坏）。
- **缓存命中可见**：已开 `--enable-prefix-caching`（hybrid 模型默认开启）与
  `--enable-prompt-tokens-details`；API 返回 `usage.prompt_tokens_details.cached_tokens`
  可核对前缀缓存命中 token 数（深会话/agentic 场景命中率 ~100×，同前缀二次请求比对）。
**不要改**：`--block-size 2304`（DeepGEMM arch-12 要求 kpool 页 64 对齐；2176 会崩）、
  `--moe-backend marlin`、`--kv-cache-dtype fp8_e4m3`、`--enforce-eager`（b12x/fp8 路径要求，
  同时把单流顶到结构化解码的上限）、`--kv-cache-memory` 默认（GB10 上更大池并发 prefill 会
  NVRM OOM——每次加大必须用真实长 prefill 把关，勿盲升）。
- **起停纪律**（源 hard-won rules）：先全部 teardown 再重启任一 rank；每次发布核对各节点
  `IMAGE` 一致；`docker logs` 在 `rm -f` 之前抓。
- **上下文对齐**：`MAX_MODEL_LEN` 保持 64 对齐（kpool·64 / MLA 对齐规则）。
- **思考默认关**：`--default-chat-template-kwargs '{"enable_thinking": false}'`；请求级用
  `chat_template_kwargs: {"enable_thinking": true}` 重开。开思考时 `max_tokens` 含推理 token。
- **Vision（图片输入）**：checkpoint 自带 text-only 模板，图片请求会 500；想开 Vision 需把
  上游仓库的 `chat_template_mm.jinja` 放进节点缓存，并把 `CHAT_TEMPLATE` 填为该文件在容器内的
  路径（如 `/cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/<hash>/chat_template_mm.jinja`）。
- **InstantTensor（v9）未启用**：多节点下加载后 ~1 分钟 rank 静默死亡，本配方保持 v8 标准加载。

## 参考来源

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：
  本配方（Lane A）的源部署与 `launch-glm53-tp4-24g.sh` launcher（原 `launch-glm53-vllm-tp4.sh`
  已更名）、`flusher-unconditional.sh`、`docker/` patch 栈、`docs/DEPLOY-REPORT.md`、
  `docs/GB10-KV-MEMORY-LADDER.md`、`docs/SM121-CRASH-FORENSICS-2026-08-27.md`
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)：compressed-tensors 主模型（默认）
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：NVFP4 量化权重
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)：基础模型
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎（`glm5_next` 支持 PR #53906）
