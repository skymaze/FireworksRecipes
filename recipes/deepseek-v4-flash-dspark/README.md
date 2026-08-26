# DeepSeek-V4-Flash DSpark · Fireworks 配方

用 Fireworks 在 **恰好 2 台** DGX Spark（head + 1 worker，RoCE 组网）上跑起
DeepSeek-V4-Flash：

- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（Anemll 预构建 vLLM 分发镜像）
- 拓扑：**固定 2 节点 · TP=2**，FlashInfer b12x + dspark 投机 · NVFP4 DS-MLA · 1M 上下文
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（约 167GB，Fireworks 分发后离线加载）
- 对外服务名：`deepseek-v4-flash-0731`（v1.1.0 起与 Mia 上游默认一致）

> 本配方直接使用 Anemll 预构建分发镜像，开箱即体验 FlashInfer b12x + dspark 投机路径，
> 1M 上下文（NVFP4 DS-MLA）。仓库内另有一条 4 节点 TP=4 的 DSpark 配方，按拓扑需要选用。

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
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | 对外服务名 |
| `VLLM_PORT` | `8888` | vLLM API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文 |
| `MAX_NUM_SEQS` | `6` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批最大 token 数 |
| `GPU_MEMORY_UTILIZATION` | `0.835` | 文本显存利用率（与 Mia 配方默认一致） |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token 数 |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | 长 prefill 分块阈值（#27，防 decode 饥饿） |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache 检查点间隔（#26） |
| `DEFAULT_THINKING` | `max` | 思考模式 off/low/high/max（与 Mia 配方默认一致） |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充。

## 与 Mia 上游同步（v1.1.0 · 2026-08-25）

- **默认档位对齐**：`DEFAULT_THINKING=max`、`SERVED_MODEL_NAME=deepseek-v4-flash-0731`、
  `GPU_MEMORY_UTILIZATION=0.835`（文本利用率）。
- **JIT 编译缓存持久化**（Mia #65/#117）：`TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR` /
  `B12X_CUTE_COMPILE_CACHE_DIR` 落到挂载的 HF 卷，容器重建不再重复 JIT（避免 TP 失同步）。
- **长 prefill 分块**（#27）：`--long-prefill-token-threshold 1024`，防止长 prefill 挤占
  decode 通道。
- **prefix-cache 保留**（#26）：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` 稀疏化 SWA
  检查点。

> 说明：Mia 配方里由容器入口脚本在启动时应用的一批 patch 型热修复（#27 inflight cap、
> #43 decode 公平、boot-shape warmup、#133 Triton 特化等）依赖其仓库内 `patches/`
> 目录。本配方是自包含 compose（只跑 `vllm serve`），不打包这些 patches，只携带
> 运行时可直接消费的部署级 knob；需要完整热修复链时请用上游仓库的 `start-*.sh`。

## 已知问题

- `MTP_NUM_TOKENS` 勿低于 5：k<5 会静默截断 dspark 草稿块，解码吞吐下降。
- b12x 栈 1M 上下文显存压力大，`GPU_MEMORY_UTILIZATION` 默认 0.835；显存紧张或图形
  捕获 OOM 时可降至 ~0.80。
- `DEFAULT_THINKING=max` 的推理可能很长（实测中等提示下可达数万字符），需给足
  `max_tokens`，或按请求改用 `low`/`off`。
- 并发长 prefill 仍会排队（#27 语义）：本配方以 1024 阈值分块保证 decode 不被挤占，
  但无法并行服务多个超长冷预填充。
- 仅支持恰好 2 节点（TP=2）；其他拓扑请选对应配方或自建。

## 参考来源

完整来源见仓库根 `NOTICE.md`：

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)（DSpark 配方路线参考，本配方 v1.1.0 与其 2026-08-25 状态对齐）
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
