# DeepSeek-V4-Flash DSpark · Fireworks 配方

用 Fireworks 在 **恰好 2 台** DGX Spark（head + 1 worker，RoCE 组网）上跑起
DeepSeek-V4-Flash：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix`
  （在 Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` 上烘焙了 **Mia 完整 fail-closed
  热修复链**，启动时应用到 `vllm serve` 前）
- 拓扑：**固定 2 节点 · TP=2**，FlashInfer b12x + dspark 投机 · NVFP4 DS-MLA · 1M 上下文
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（约 167GB，Fireworks 分发后离线加载；
  `DSPARK_REVISION` 默认留空 = 启动时自动使用本地缓存快照的 commit sha，离线安全）
- 对外服务名：`deepseek-v4-flash-0731`

> 本配方直接使用烘焙了 Mia 热修复链的分发镜像（镜像内 `/opt/dspark/entrypoint.sh` 在
> 每次容器启动时按 fail-closed 顺序应用补丁再启 vLLM，行为与 Mia 仓库
> `start-*.sh` 一致）。仓库内另有一条 4 节点 TP=4 的 DSpark 配方，按拓扑需要选用。

## 快速开始

发布前就绪：

- 集群：**恰好 2 台**节点（head + 1 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731` 已分发到节点（含 `encoding/encoding_dsv4.py`）。
- 镜像：ACR 热修复镜像已可拉取。

> 节点数锁定为**恰好 2**，TP/分布式参数按此调优。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `…/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix` | Mia 热修复烘焙镜像 |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 已下载模型 |
| `DSPARK_REVISION` | 空 | 留空=自动用本地缓存快照 sha（离线安全）；显式钉住须匹配实际快照 |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | 对外服务名 |
| `VLLM_PORT` | `8888` | API 端口（仅 `--port`，已与 vLLM 内部端口解耦） |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文 |
| `MAX_NUM_SEQS` | `6` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批最大 token 数 |
| `GPU_MEMORY_UTILIZATION` | `0.835` | 文本显存利用率 |
| `MTP_NUM_TOKENS` | `5` | dspark 投机 token 数 |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | 长 prefill 分块阈值（#27） |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache 检查点间隔（#26） |
| `DEFAULT_THINKING` | `max` | 思考模式 off/low/high/max |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `2` | #27：并发分块 prefill 上限（1-3） |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | 1 = 启用 GPU `thinking_token_budget` |
| `DSPARK_API_KEYS` | 空 | 空格分隔多 key 认证（留空无认证） |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充。

其余热修复开关（`DSPARK_SKIP_HOTFIX` 等）见 recipe 变量。

## 内置热修复链（v1.2.0 · 快照 70a7cc4b，2026-08-25）

镜像在每次启动时按 fail-closed 顺序应用：

- **编码热修复**（#52 reasoning-effort 映射 + #21）：从 HF 快照拷贝
  `encoding_dsv4.py` 后打补丁（缺文件只告警不失败）。
- **Python**：#55 tool-call 截断、#109 空 encoder 输出、#27 partial-prefill 并发、
  #43 decode 公平、#26 SWA 前缀缓存、#133 Triton 特化、suppress-stops-in-reasoning。
- **Shell**：#22 nvfp4_ds_mla 长上下文、#79 spin-wait、6 个 v0.27 性能回移植
  （#50312 / #49486 / #48407 / #48957 / #50298 / grammar-advance）。
- **可选（默认关）**：`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`、`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`、
  `DSPARK_API_KEYS`（多 key 认证 + 日志脱敏）。

同时持久化 Triton / TileLang / B12X-CuTeDSL JIT 编译缓存到 HF 卷（容器重建不重复
JIT，避免 TP 失同步）。

## 已知问题

- `MTP_NUM_TOKENS` 勿低于 5：k<5 会静默截断 dspark 草稿块，解码吞吐下降。
- b12x 栈 1M 上下文显存压力大，`GPU_MEMORY_UTILIZATION` 默认 0.835；显存紧张或图形
  捕获 OOM 时可降至 ~0.80。
- `DEFAULT_THINKING=max` 的推理可能很长（实测中等提示下可达数万字符），需给足
  `max_tokens`，或按请求改用 `low`/`off`。
- 并发长 prefill 仍会排队（#27 语义）：本配方以 1024 阈值分块保证 decode 不被挤占，
  但无法并行服务多个超长冷预填充。
- 仅支持恰好 2 节点（TP=2）；其他拓扑请选对应配方或自建。

## 镜像构建来源

构建上下文在本仓库外的 `FireworksProject/dspark-image-build/`
（Dockerfile + entrypoint.sh + patches/），基镜像 Anemll `0.1.1`，快照
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark @ 70a7cc4b`。

## 参考来源

完整来源见仓库根 `NOTICE.md`：

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
