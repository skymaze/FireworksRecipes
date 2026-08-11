# DeepSeek-V4-Flash-0731 · TP=4 · Fireworks 配方（4× DGX Spark）

用 Fireworks 在 **4 台** DGX Spark（head + 3 worker，RoCE 组网）上跑起
**DeepSeek-V4-Flash-0731** 的 **TP=4** 服务：

- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（Anemll 预构建 vLLM 分发镜像，vLLM 0.25.x）
- 拓扑：**固定 4 节点 · TP=4**；FlashInfer b12x + dspark 投机 k=5 · NVFP4 DS-MLA · **1M 上下文**
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（约 167GB，Fireworks 分发后离线加载）
- 默认参数按 **agentic 工作负载实机验证** 结果固化（k=5、`GPU_MEMORY_UTILIZATION=0.80`、
  `DEFAULT_THINKING=max`、1M 上下文）。

## 与两条 2 节点配方的区别

| 维度 | `deepseek-v4-flash-0731` | `deepseek-v4-flash-dspark` | **本配方（TP=4）** |
|---|---|---|---|
| 拓扑 | 2 节点 · TP=2 | 2 节点 · TP=2 | **4 节点 · TP=4** |
| 镜像 | 自编译 vLLM 0.26.0（专属） | Anemll 分发镜像 | Anemll 分发镜像 |
| 投机 | MTP=5 | k=5 | **k=5**（实机验证默认） |
| 上下文 | 1M | 1M | **1M** |

## 快速开始

发布前就绪：

- 集群：4 台节点（head + 3 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731` 已分发到节点。
- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` 已拉取。

> 节点数固定为 4，TP/分布式参数按此调优；`--ulimit nofile=1048576` 已写入配方
> （TP=4 打开 4× NCCL socket，缺了会启动失败）。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll 镜像 |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_PORT` | `8888` | vLLM API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文（实机验证默认） |
| `MAX_NUM_SEQS` | `4` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 对齐 vLLM 预热预算阈值（投机解码会从该值减 `(k−1)×seqs`，并发高时可上调） |
| `GPU_MEMORY_UTILIZATION` | `0.80` | 实机验证默认；过高（0.90）无法开机 |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token 数（勿低于 dspark_block_size=5） |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 只捕获 decode 图（实机验证零损耗）；`{}` 恢复双捕获 |
| `DEFAULT_THINKING` | `max` | 默认思考模式 off/low/high/max，请求级可覆盖 |

`NODES_TOTAL`（固定 4）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充。`max_cudagraph_capture_size` 自动按 `MAX_NUM_SEQS × (k+1)`
= 4×6 = 24 计算（改动 `MAX_NUM_SEQS` / `MTP_NUM_TOKENS` 时同步生效）。

## 已知问题与部署注意

- **JSON 默认值不能写进 `$${VAR:-...}`**：bash 在默认值内第一个 `}` 处截断参数展开，
  变量已由 compose 注入时会在 JSON 尾部残留一个 `}`（`--compilation-config` 报
  `trailing characters ... "FULL_DECODE_ONLY"}}`）。配方内用
  `[ -n "$${VAR:-}" ] || VAR='{"..."}'` 的无嵌套写法兜底，**勿改回** `${VAR:-{...}}`。
- **`max_num_batched_tokens` 不是 prefill 预算**：投机解码下 vLLM 从该值减去
  `(k−1)×max_num_seqs` 得到 `max_num_scheduled_tokens`，低于 8192 会告警；本配方默认 8192 对齐阈值，并发高时可适当上调。
- **`GPU_MEMORY_UTILIZATION`** 夹在损失与 OOM 之间：0.90 无法开机（TP=4 每机仅 ~39GB 权重，
  0.80 已实测稳定）；提高它会回收 KV 池但别顶到 0.90。
- **`--override-generation-config` 已移除**：实机验证后不再把服务端温度覆盖参数下发
  （容器 environment 中 `OVERRIDE_GENERATION_CONFIG` 保留但命令不再引用）；需要压默认温度时
  在做请求侧设置或恢复该参数。若换镜像 / 改引擎，`MTP_NUM_TOKENS` 等投机参数需重新实测。
- 只支持 4 节点（TP=4）；其他拓扑请选对应配方或自建。

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方组件来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)：分发镜像
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎
- [lukealonso/b12x](https://github.com/lukealonso/b12x)：MXFP4 MoE 内核
