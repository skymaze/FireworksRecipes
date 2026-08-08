# DeepSeek-V4-Flash-0731 · Fireworks 配方文档

> 本文件是 Fireworks「配方源」目录条目的介绍文档（README.md），随配方一起从 git
> 仓库同步到本地目录，Fireworks 界面可直接预览。想直接开始：在 Fireworks「配方商店」
> 里点这条配方 →「一键下载并运行」。

## 这是什么

用 **Fireworks**（DGX Spark 集群管理工具）在 2 台 DGX Spark 上跑起
**DeepSeek-V4-Flash-0731** 的专属镜像配方：

- 镜像：`fireworks-models/deepseek-v4-flash-0731:0.3.1`（主流 vLLM v0.26.0 +
  Anemll 式 GB10 定向 overlay，构建期烘培 hybrid-draft-loader 补丁与调优 ENV）
- 拓扑：**双节点 · TP=2**（head 1 台 + worker 1 台，经 RoCE 高速网组集群）
- 加载：InstantTensor + dspark 投机解码（MTP）
- KV 缓存：`nvfp4_ds_mla`（DS-MLA 4-bit，已验证 **1M 上下文**）
- MoE：`auto`（DeepGEMM）原生路径 —— 真机验证 b12x 的 prefill 在 v0.26.0 集成下不可用
  （~88 tok/s），auto 为 2200+ tok/s

模型权重**不烤进镜像**（约 167 GB），由 Fireworks 模型管理分发到各节点 HF 缓存，
镜像内固定离线加载（`HF_HUB_OFFLINE=1`）。

## 快速开始

1. **组集群**：在 Fireworks「集群」加入**恰好** 2 台已部署 Agent 的 DGX Spark（head +
   1 worker），配置并测试 RoCE 高速网（本配方为固定 2 节点拓扑）。
2. **下载模型**（可选，发布向导会自动处理）：`deepseek-ai/DeepSeek-V4-Flash-0731`
   由控制平面下载 → 发送 head → RoCE 同步 worker，避免逐节点重复下载。
3. **拉取镜像**：`fireworks-models/deepseek-v4-flash-0731:0.3.1`（同样一键分发）。
4. **发布**：配方商店安装本配方 →「一键下载并运行」进入发布向导 → 选集群 → 点发布。

> 发布向导中的节点数会锁定为 **恰好 2 台**（TP/模型参数按此调优），不再给「2 个以上」
> 的模糊空间；模型/镜像下拉会自动从已下载/已拉取的清单里选，缺失时提示先发送。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_IMAGE` | `fireworks-models/deepseek-v4-flash-0731:0.3.1` | 已拉取镜像 |
| `TENSOR_PARALLEL_SIZE` | `2` | 固定 2（GB10 每机 1 GPU，TP=节点数=2；已按双节点调优） |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文（需 KV `nvfp4_ds_mla`） |
| `GPU_MEMORY_UTILIZATION` | `0.88` | 1M 上下文需 ≥0.88（DeepGEMM 栈） |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | 推荐，`fp8_ds_mla` 可用 |
| `NUM_SPECULATIVE_TOKENS` | `5` | dspark 投机 token 数 |
| `MASTER_PORT` | `25000` | 分布式主端口（head rank0 监听） |
| `DEFAULT_THINKING` | `off` | 思考模式（off/low/high/max）——本配方镜像未集成时勿填 |

`MASTER_ADDR`、`NODES_TOTAL`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*` 均自动
填充（head 的 RoCE IP、节点角色 rank、RoCE 网卡/GID），无需手填。

## 已知问题与提示

- `GPU_MEMORY_UTILIZATION` 降到 0.80 仅 b12x 栈可行；DeepGEMM(1M ctx) 需 ≥0.88。
- 本配方为**固定 2 节点 · TP=2** 调优；如需单节点等其它拓扑，请等待对应配方或自建。
- 权重先由 Fireworks 分发到缓存，首次发布会自动拉起模型加载（耗时取决于存储/网络）。

## 更新记录

- **0.3.1**：对齐 Fireworks 新配方变量模型（`MASTER_ADDR=head_roce_ip` 任务级 head、
  `MASTER_PORT` 改配方用户变量）；部分构建优化。
