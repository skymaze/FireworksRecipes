# DeepSeek-V4-Flash DSpark · Fireworks 配方

用 Fireworks 在 **恰好 2 台** DGX Spark（head + 1 worker，RoCE 组网）上跑起
DeepSeek-V4-Flash：

- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（Anemll 预构建 vLLM 分发镜像）
- 拓扑：**固定 2 节点 · TP=2**，FlashInfer b12x + dspark 投机 · NVFP4 DS-MLA · 1M 上下文
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（约 167GB，Fireworks 分发后离线加载）

> 与「专属镜像」配方 `deepseek-v4-flash-0731` 的区别：本配方直接使用 Anemll 分发镜像，
> 可快速体验 dspark 投机路径；专属镜像为自编译 vLLM v0.26.0 + GB10 定向 overlay，
> prefill 更优（MoE=auto，2200+ tok/s）。按需选其一。

## 快速开始

发布前就绪：

- 集群：**恰好 2 台**节点（head + 1 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731` 已分发到节点。
- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` 已拉取。

> 节点数锁定为**恰好 2**，TP/分布式参数按此调优。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll 镜像 |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_PORT` | `8888` | vLLM API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文 |
| `GPU_MEMORY_UTILIZATION` | `0.80` | 显存利用率（b12x 栈） |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token 数 |
| `DEFAULT_THINKING` | `off` | 思考模式 off/low/high/max |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充。

## 已知问题

- `MTP_NUM_TOKENS` 勿低于 5：k<5 会静默截断 dspark 草稿块，解码吞吐下降。
- b12x 栈 1M 上下文显存压力大，`GPU_MEMORY_UTILIZATION` 建议 ≥0.80。
- 仅支持恰好 2 节点（TP=2）；其他拓扑请选对应配方或自建。

## 参考来源

完整来源见仓库根 `NOTICE.md`：

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)（DSpark 配方路线参考）
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
