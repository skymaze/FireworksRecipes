# GLM-5.3-Flash NVFP4 · TP=4 · Lane A（fp8 KV）· Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next` 架构）的 **NVIDIA consumer Blackwell 首款 NoPE-MLA fp8 KV** 的 **TP=4** 服务；
本配方对应上游部署的 **Lane A — fp8 KV**（FlashInfer SM12x unlock，就速度优化的一路）：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` = vLLM day-0
  `glm53-flash-arm64-cu130` + 按上游 chain 烘焙的 **8 层 patch 栈**（本配方已推送 ACR）：
- 模型：`LibertAIDAI/GLM-5.3-Flash-NVFP4`（NVFP4 权重量化，120 分片 ~182 GiB，censored）；
  uncensored drop-in：`drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- 并行：**TP=4**（`--tensor-parallel-size 4`）、`mp` 后端、4 台每机 1 GPU
- KV：**fp8_e4m3**（656 B/token/layer）、每 rank 24 GiB 预算（5.03M pool 按 32 GiB/rank 测得）、
  `--kv-cache-memory 25769803776`、`--block-size 2304`、gmu 0.85
- 投机：**原生 MTP k=4**；工具调用 `glm47` parser 开启、思考默认关（请求级可开）
- 上下文：**1,048,576（1M）**；端口默认 `8000`
- 实测（上游 2026-08-27，TP4，warmed）：结构化解码 ~55 tok/s、prefill ~3,530 tok/s · TTFT ~0.2s

> 上游仓库把当前 TP4 部署分为 **2×2**：**Lane A = fp8 KV**（本配方，速度优先，每日主力）与
> **Lane B = NVFP4 KV**（b12x 路径，368 B/token/layer，1.32× KV 池但解码慢 ~33%）。需要最大
> KV 池时另看 Lane B 配方（若有）；日常 agentic 生产选本 Lane A。

## 快速开始（发布前就绪）

- 集群：**4 台**节点（head + 3 worker），双 rail RoCEv2 已配置测试。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8`（阿里云 ACR；
  Fireworks 拉取后分发到节点）。上游作者的 `radixark/vllm-glm53-flash:sm121-v8` 仅为本地
  构建链 tag、未公开推送，故本配方镜像按上游 chain 自行烘焙后推送 ACR。
- 模型：`LibertAIDAI/GLM-5.3-Flash-NVFP4`（HF hub 布局 `models--LibertAIDAI--GLM-5.3-Flash-NVFP4`，
  120 分片 ~182 GiB）分发到节点缓存；容器内 `HF_HUB_OFFLINE=1` 按 repo id 离线解析
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
| `KV_CACHE_MEMORY` | `25769803776` | 每 rank fp8 KV 预算（24 GiB）；防 GB10 UMA OOM |
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

- **不要改**：`--block-size 2304`（DeepGEMM arch-12 要求 kpool 页 64 对齐；2176 会崩）、
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
  本配方（Lane A）的源部署与 `launch-glm53-vllm-tp4.sh` launcher、`docker/` patch 栈、
  `docs/DEPLOY-REPORT.md`、`docs/GB10-KV-MEMORY-LADDER.md`
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：NVFP4 量化权重
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)：基础模型
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎（`glm5_next` 支持 PR #53906）
