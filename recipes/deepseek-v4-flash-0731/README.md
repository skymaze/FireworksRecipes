# DeepSeek-V4-Flash-0731 · Fireworks 配方

在 Fireworks 中用 **2 台 DGX Spark**（head + 1 worker，RoCE 组网）跑起
**DeepSeek-V4-Flash-0731**：

- 镜像：`fireworks-models/deepseek-v4-flash-0731:0.3.1`（主流 vLLM v0.26.0 + GB10 定向
  overlay，构建期烘培补丁与调优 ENV）
- 拓扑：**固定 2 节点 · TP=2**
- 加载：InstantTensor + dspark 投机（MTP=5）
- KV：`nvfp4_ds_mla`，**1M 上下文**
- MoE：`auto`（DeepGEMM），真机 prefill 2200+ tok/s

权重不打包进镜像（约 167GB），由 Fireworks 模型管理分发到节点后离线加载。

## 快速开始

发布前就绪：

- 集群：**恰好 2 台**节点（head + 1 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731` 已分发到节点。
- 镜像：`fireworks-models/deepseek-v4-flash-0731:0.3.1` 已拉取。

> 节点数锁定为**恰好 2**（模型参数按此调优）；模型/镜像下拉自动选已就绪项，缺失时提示先发送。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_IMAGE` | `fireworks-models/deepseek-v4-flash-0731:0.3.1` | 已拉取镜像 |
| `TENSOR_PARALLEL_SIZE` | `2` | 固定 2（GB10 每机 1 GPU） |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文，需 `nvfp4_ds_mla` KV |
| `GPU_MEMORY_UTILIZATION` | `0.88` | 1M 上下文需 ≥0.88 |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | 推荐；`fp8_ds_mla` 也可用 |
| `NUM_SPECULATIVE_TOKENS` | `5` | dspark 投机 token 数 |
| `MASTER_PORT` | `25000` | 分布式主端口 |
| `DEFAULT_THINKING` | `off` | 思考模式 off/low/high/max |

`MASTER_ADDR`、`NODES_TOTAL`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充，无需手填。

## 已知问题

- `GPU_MEMORY_UTILIZATION` 低于 0.88 仅 b12x 栈可行；DeepGEMM 1M 上下文需 ≥0.88。
- 固定 2 节点 · TP=2 调优；其他拓扑请选对应配方或自建。
- 首次发布会自动加载权重，耗时取决于存储/网络。

## 参考来源

完整来源见仓库根 `NOTICE.md`：

- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
- [jvr0x/dgx-spark-bench](https://github.com/jvr0x/dgx-spark-bench)
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
