# DeepSeek-V4-Flash 2x DGX Spark (DSpark) · Fireworks 配方文档

> 本配方由 Fireworks 原内置种子配方迁移而来（MiaAI-Lab 参考配方路线），现在是配方源
> 里的一个标准配方。在 Fireworks「配方商店」安装后即可一键发布，**不随 Fireworks
> 主程序更新而更新**——本仓库新增/修订即可。

## 是什么

用 **Fireworks** 在 **恰好 2 台** DGX Spark（head + 1 worker，RoCE 高速网组集群）
跑起 DeepSeek-V4-Flash：

- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（Anemll 预构建 vLLM 分发镜像）
- 拓扑：**固定 2 节点 · TP=2**（NVFP4 DS-MLA + FlashInfer b12x + DSpark 投机解码）
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（约 167GB，由 Fireworks 模型管理分发
  到节点 HF 缓存后离线加载）

> 与「专属镜像」配方（`deepseek-v4-flash-0731`）的区别：本配方用 Anemll 分发镜像，
> 适合想快速试 DSpark 投机路径的场景；专属镜像配方为自编译 vLLM v0.26.0 + GB10 定向
> overlay，prefill 性能更优（MoE=auto/DeepGEMM 2200+ tok/s）。按需选其一。

## 快速开始

1. 组 **恰好 2 台** 节点的集群（head+worker，配置并测试 RoCE 高速网）。
2. 下载模型 `deepseek-ai/DeepSeek-V4-Flash-0731`（发布向导可一键发送到节点）。
3. 拉取镜像 `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（一键分发到节点）。
4. 配方商店安装本配方 →「一键下载并运行」→ 选集群 → 发布。

> 发布向导会把节点数递归在 **恰好 2 台**（本配方为固定拓扑），TP/分布式参数按
> 2 节点调优，不解耦「多节点」。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll 镜像 |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_PORT` | `8888` | vLLM API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文 |
| `GPU_MEMORY_UTILIZATION` | `0.80` | 显存利用率（b12x 栈） |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token（≥ checkpoint dspark_block_size=5） |
| `DEFAULT_THINKING` | `off` | off/low/high/max 思考模式 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`（head 的 RoCE IP）、`NODE_RANK`、
`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*` 自动填充。

## 已知问题

- k<5 会静默截断 dspark 草稿块（解码吞吐下降），勿改小 `MTP_NUM_TOKENS`。
- b12x 栈 1M 上下文显存压力大，`GPU_MEMORY_UTILIZATION` 建议 ≥0.80。
- 仅支持恰好 2 节点（TP=2）；如需其他拓扑请选对应配方或自建。

## 更新记录

- **1.0.0**：从 Fireworks 内置种子迁移至配方源；固定 2 节点拓扑；变量模型对齐
  （`MASTER_ADDR=head_roce_ip`、`MASTER_PORT` 为用户变量 25000）。
