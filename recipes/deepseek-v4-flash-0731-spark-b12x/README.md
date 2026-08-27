# DeepSeek-V4-Flash-0731 · Spark-vLLM b12x · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上跑起
**DeepSeek-V4-Flash-0731** 的 **TP=2** 服务：

- 镜像：`eugr/spark-vllm-b12x:latest`（spark-vllm b12x 预构建 vLLM 分发镜像）
- 拓扑：**固定 2 节点 · TP=2**；**B12X MLA SPARSE** 注意力 + **b12x** MoE/线性层 · dspark 投机 k=5 · **FP8 KV** 缓存 · 1M 上下文
- 加载：**instanttensor** + AOT 编译（`VLLM_USE_AOT_COMPILE=1`），首启快
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（Fireworks 分发后离线加载）
- 服务端口默认 `8000`；默认思考模式 `high`

> 发起来源是一条双节点 `docker run` 命令（head + `--headless` worker 两份）。本配方把
> 该命令移植进 Fireworks compose 体系（宿主 shell 变量改自动填充变量），并对照既有
> dspark 配方修正了几处配置，详见[与源命令的差异](#与源命令的差异)。

## 与既有配方的区别

| 维度 | `deepseek-v4-flash-dspark` | `deepseek-v4-flash-0731-tp4-4x` | **本配方（spark-vllm-b12x）** |
|---|---|---|---|
| 拓扑 | 2 节点 · TP=2 | 4 节点 · TP=4 | **2 节点 · TP=2** |
| 镜像 | Anemll `dspark-vllm-gx10` | Anemll `dspark-vllm-gx10` | **`eugr/spark-vllm-b12x`** |
| KV 缓存 | NVFP4 DS-MLA | NVFP4 DS-MLA | **FP8**（本镜像 b12x 栈） |
| MoE/线性后端 | flashinfer_b12x | flashinfer_b12x | **b12x** |
| 注意力后端 | （默认） | （默认） | **B12X_MLA_SPARSE** |
| 投机 | dspark k=5 | dspark k=5 | dspark k=5 |
| 上下文 | 1M | 1M | 1M |
| 加载格式 | HF 离线加载 | HF 离线加载 | **instanttensor（AOT）** |
| 编译模式 | — | FULL_DECODE_ONLY | **FULL_AND_PIECEWISE + custom_ops=all** |
| 并发上限 | 6 | 4 | **8** |
| 默认端口 | 8888 | 8888 | **8000**（源命令） |

## 快速开始

发布前就绪：

- 集群：2 台节点（head + 1 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731` 已分发到节点。
- 镜像：`eugr/spark-vllm-b12x:latest` 已拉取。

> 节点数固定为 2，TP/分布式参数按此调优。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SPARK_VLLM_IMAGE` | `eugr/spark-vllm-b12x:latest` | spark-vllm b12x 镜像 |
| `SPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `VLLM_PORT` | `8000` | vLLM API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文（源命令未显式指定，本配方对齐既有配方默认） |
| `MAX_NUM_SEQS` | `8` | 最大并发序列数（源命令） |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批最大 token 数 |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | `64` | CUDA Graph 捕获上限（源命令固定 64；既有配方按 `MAX_NUM_SEQS×(k+1)`=48 计算） |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 源命令默认；既有配方实机验证 0.80 稳定，0.90 无法开机 |
| `LOAD_FORMAT` | `instanttensor` | 模型加载格式；若 Fireworks 分发的是标准 HF safetensors 需改为空(auto)/safetensors |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token 数（勿低于 dspark_block_size=5） |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` | CUDA Graph 编译模式（源命令） |
| `DEFAULT_THINKING` | `high` | 默认思考模式 off/low/high/max，请求级可覆盖 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充；源命令里的 `$IFACE_NAME / $IB_IF / $HEAD_IP` 是宿主 shell 变量，
已对应映射为 `NCCL_SOCKET_IFNAME(netdev) / NCCL_IB_HCA(hca) / MASTER_ADDR(head_roce_ip)`，
`GLOO/TP/MN/OMPI/UCX` 网卡统一跟随 `NCCL_SOCKET_IFNAME`。

## 与源命令的差异

移植进 Fireworks 时相对原 `docker run` 的命令/行为改动如下。

**修正（对照既有 dspark 配方判定为错误，已改）**

1. **`--reasoning-config` 分隔串为空**：源命令是
   `"reasoning_start_str":"","reasoning_end_str":""`。deepseek_v4 reasoning parser 需要
   思考分隔串来剥离 `reasoning_content` 与工具调用；既有实机配方用
   `" thinking"/" response"`。空串会被当作没有思考边界，导致解析异常，已改为与既有
   配方一致的 `" thinking"/" response"`。
2. **缺 `--max-model-len`**：源命令未指定，上下文长度由引擎自估；既有 dspark 配方固定
   `1048576`（1M）。已补上默认 1M（`MAX_MODEL_LEN` 变量），与商店卡片 `context_length` 一致。
3. **依赖宿主 shell 变量**：`$HEAD_IP/$IFACE_NAME/$IB_IF` 在 Fireworks 发布环境不存在，
   已改由自动填充变量注入（见上）。

**保留（本镜像特性，未按既有配方改动）**

- `--kv-cache-dtype fp8`：既有配方用 `nvfp4_ds_mla`。本镜像走 B12X MLA SPARSE + FP8 GEMM
  栈，`fp8` 为该栈默认；若镜像也支持 `nvfp4_ds_mla`，可切换为 GB10 优化的 DS-MLA 缓存。
- `--load-format instanttensor`：instanttensor 是专用权重格式。若 Fireworks 分发的是标准
  HF safetensors 目录，此处会加载失败，请将 `LOAD_FORMAT` 置空(auto)或 `safetensors`。
- `--moe-backend b12x` / `--linear-backend b12x` / `--attention-backend B12X_MLA_SPARSE`：
  与既有配方的 `flashinfer_b12x` 命名不同的本镜像后端。
- `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`：
  源命令默认；tp4 配方实机验证 `FULL_DECODE_ONLY` 零损耗，可按需切换。
- `--gpu-memory-utilization 0.85`、`--max-num-seqs 8`、`--max-cudagraph-capture-size 64`、
  `--max-num-batched-tokens 8192`、端口 `8000`：源命令取值，均保留为可调变量。
- `--speculative-config` 内多出的 `"attention_backend":"B12X_MLA_SPARSE"` 键：本镜像要求，
  保留。

**沿用既有 dspark 配方的稳定项（源命令未写，但为同裁剪实机验证的集成层）**

- `--enable-chunked-prefill` / `--async-scheduling` / `--enable-prompt-tokens-details` /
  `--pipeline-parallel-size 1`：与 token 批处理预算配套的引擎稳定项。
- RoCE 多节点 `NCCL_*` 完整配置（GID 自动解析、ROCEv2、跨 NIC 等）与 `mp` 分布式后端。
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`（放开长上下文限制，1M 必需）等容器环境。

## 已知问题与部署注意

- **JSON 默认值不能写进 `$${VAR:-...}`**：bash 在默认值内第一个 `}` 处截断参数展开，
  导致 `--compilation-config` 报 `trailing characters ...`。配方沿用 tp4 的
  `[ -n "$${COMPILATION_CONFIG:-}" ] || COMPILATION_CONFIG='{"..."}'` 无嵌套写法兜底，
  **勿改回** `${VAR:-{...}}`。
- **`LOAD_FORMAT=instanttensor` 依赖分发格式**：发布前确认节点缓存里的模型目录是
  instanttensor 布局；否则改 `LOAD_FORMAT`（默认 HF 加载）。
- **`GPU_MEMORY_UTILIZATION` 夹在损耗与 OOM 之间**：0.85 高于既有配方实机验证的 0.80，
  若 1M 上下文 + 并发 8 下显存吃紧，先下调 `MAX_NUM_SEQS` 或 `MAX_MODEL_LEN`，别顶到 0.90。
- **CUDA Graph 捕获上限 64**：高于既有配方按 `seqs×(k+1)` 推算的 48；图捕获批次更大时
  更稳但占用更多显存，可按需下调。
- 本配方按源 `docker run` 整理，**未经实机验证**；换镜像 / 改引擎后投机参数（`MTP_NUM_TOKENS` 等）
  需重新实测。按 README 分支模型，应先入 `dev` 实机验证后再合并 `main`。
- 只支持 2 节点（TP=2）；其他拓扑请选对应配方或自建。

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方组件来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- `eugr/spark-vllm-b12x:latest`：分发镜像（源 docker run 引用）
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎
- [lukealonso/b12x](https://github.com/lukealonso/b12x)：MXFP4 MoE 内核
