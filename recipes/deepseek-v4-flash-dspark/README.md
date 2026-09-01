# DeepSeek-V4-Flash-Vision-Exp DSpark · Fireworks 配方

用 Fireworks 在 **恰好 2 台** DGX Spark（head + 1 worker，RoCE 组网）上跑起
DeepSeek-V4-Flash-**Vision-Exp**：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix3`
  （在 Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` 上烘焙了 **Mia 完整 fail-closed
  热修复链 + Vision-Exp 原生图片支持**，启动时应用到 `vllm serve` 前）
- 拓扑：**固定 2 节点 · TP=2**，FlashInfer b12x + dspark 投机 k=6 · NVFP4 DS-MLA · 1M 上下文
- 模型：`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`（官方首个 V4 家族实验多模态模型，
  总参数 ~305B 含 ViT+Aligner，MIT；约 167GB，Fireworks 分发后离线加载；
  `DSPARK_REVISION` 默认留空 = 启动时自动使用本地缓存快照的 commit sha，离线安全）
- 多模态：原生 **图片输入**（OpenAI `image_url`，JPEG/PNG/GIF/WebP；GIF 取静帧；
  每请求 ≤ `LIMIT_MM_PER_PROMPT` 默认 8 张；**仅 user 消息可带图**——system/assistant
  消息带图返回 HTTP 400；**官方权重无视频编码器**）
- 对外服务名：`deepseek-v4-flash-vision-exp`

> 本配方直接使用烘焙了 Mia 热修复链的分发镜像（镜像内 entryscript 在每次容器启动时按
> fail-closed 顺序应用补丁再启 vLLM，行为与 Mia 仓库 `start-*.sh` 一致）。Vision 支持
> 来自上游 2026-08-31 `feat/vision-exp`（PR #164）：启动热修复 `hotfix-dsv4-vision-exp.py`
> + `patches/vision_exp/` 构建图片塔、映射 `vision.*`/`aligner.*` 权重并注册 vLLM
> 多模态 processor，另含 #165 修复（system/assistant 文本中出现的字面 `<image>` 子串
> 不再误判为图片）。上游已移除旧的 Qwen3-VL sidecar / MCP 间接路径。
> 仓库内另有一条 4 节点 TP=4 的 DSpark 配方（仍指向 0731 文本 checkpoint），按拓扑需要选用。

## 快速开始

发布前就绪：

- 集群：**恰好 2 台**节点（head + 1 worker），已配置并测试 RoCE。
- 模型：`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` 已分发到节点（含
  `encoding/encoding_dsv4.py`；Vision-Exp 的 ViT+Aligner 权重比 0731 更占显存，
  KV 池相应变小）。
- 镜像：ACR 热修复镜像 `v0.1.1-hotfix3`（上游快照 `d58c877`，2026-08-31）已可拉取。

> 节点数锁定为**恰好 2**，TP/分布式参数按此调优。

图片请求示例：

```bash
curl -s http://<head-ip>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-vision-exp","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
    {"type":"text","text":"这张图里有什么？"}]}]}'
```

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `…/dspark-vllm-gx10-mia:v0.1.1-hotfix3` | Mia 热修复烘焙镜像（含 Vision-Exp 图片支持） |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | 已下载模型 |
| `DSPARK_REVISION` | 空 | 留空=自动用本地缓存快照 sha（离线安全）；上游官方 pin 为 `86f746b3…` |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-vision-exp` | 对外服务名 |
| `VLLM_PORT` | `8888` | API 端口（仅 `--port`，已与 vLLM 内部端口解耦） |
| `MAX_MODEL_LEN` | `1048576` | 1M 上下文 |
| `MAX_NUM_SEQS` | `6` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批最大 token 数 |
| `LIMIT_MM_PER_PROMPT` | `8` | 每请求图片上限（`image=N` 简写由入口转成 Anemll 的 JSON；无视频） |
| `GPU_MEMORY_UTILIZATION` | `0.835` | 文本显存利用率（**不要手动调**，Mia 由 TEXT 档导出；KV 紧张可降到 ~0.80） |
| `MTP_NUM_TOKENS` | `6` | dspark 投机 token 数（**勿低于 6**：Vision-Exp `num_nextn_predict_layers=3`，k 须 ≥5 且整除 3，k=5 会被 Anemll 拒绝） |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | 长 prefill 分块阈值（#27） |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | SWA prefix-cache 检查点间隔（#26） |
| `DEFAULT_THINKING` | `max` | 思考模式 off/low/high/max |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `1` | #27：并发分块 prefill 上限（1-3）。上游 #154 实测 2 会放大混合流量公平性带宽（3.72–5.14x vs 1 的 1.68–2.04x），1 为安全默认；2-3 需自行压测后显式开启 |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | 1 = 启用 GPU `thinking_token_budget` |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | `0` | 1 = 应用上游 vLLM #52805 XGrammar 终止回移植（#136，源锁定 fail-closed） |
| `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` | `0` | 1 = 兼容 type-less assistant `output_text` 单元素回放（#138，仅缺省补全） |
| `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` | `0` | 1 = sparse-MLA decode 固定 64 行分块（#141 随机卡死 workaround，非根因修复） |
| `DSPARK_API_KEYS` | 空 | 空格分隔多 key 认证（留空无认证） |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_*`
均由 Fireworks 自动填充。

其余热修复开关（`DSPARK_SKIP_HOTFIX` 等）见 recipe 变量。

## KV 池与并发

Vision-Exp 权重比 0731 大（官方模型卡总参数 ~305B vs 284B，ViT+Aligner 常驻显存），
KV 池相应变小。上游同配置（util 0.83）实测 boot 日志参考：

```text
Available KV cache memory: 17.04 GiB
GPU KV cache size: 2,331,430 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.22x
```

`MAX_MODEL_LEN` / `MAX_NUM_SEQS` 是**上限不是预留**；约束是 `sum(活跃 tokens) ≤ KV 池`。
六个正常 agent 回合放得下，六个同时打满 1M 的请求放不下（会排队）。

上游 Anemll 1M/6 档位实测（results/RESULTS-2026-08-14.md）：

| 工作负载 | 参考值 |
|---|---|
| 单聊天（≤128K 任意 prompt） | 首 token 后 ~62–83 decode tok/s |
| **六路短聊天**（数百 token） | **~160–190 tok/s 聚合**（~30–37/路） |
| 六路冷 32K–128K 同时 prefill | 排队（#27），decode ~8 tok/s 保底 |

## 内置热修复链（v1.4.0 · 快照 d58c877，2026-08-31）

镜像在每次启动时按 fail-closed 顺序应用：

- **Vision-Exp 图片支持**（`hotfix-dsv4-vision-exp.py` + `patches/vision_exp/`，
  无条件 fail-closed）：构建 ViT+Aligner 图片塔、映射 `vision.*`/`aligner.*`/
  `bias_vl` 权重（含 MoE 0–2 层 `ffn.gate.bias_vl` remap）、注册多模态 processor、
  图片仅限 user 消息（#165 修复：system/assistant **文本提及** `<image>` 标签不再误判）。
- **编码热修复**（#52 reasoning-effort 映射 + #21）：从 HF 快照拷贝
  `encoding_dsv4.py` 后打补丁（缺文件只告警不失败）。
- **Python**：#55 tool-call 截断、#109 空 encoder 输出、#27 partial-prefill 并发、
  #43 decode 公平、#26 SWA 前缀缓存、#133 Triton 特化、suppress-stops-in-reasoning。
- **Shell**：#22 nvfp4_ds_mla 长上下文、#79 spin-wait、6 个 v0.27 性能回移植
  （#50312 / #49486 / #48407 / #48957 / #50298 / grammar-advance）。
- **可选（默认关）**：`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`、`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`、
  `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX`（vLLM #52805 终止回移植）、
  `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT`（Responses 历史回放）、
  `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK`（sparse-MLA 64 行分块）、
  `DSPARK_API_KEYS`（多 key 认证 + 日志脱敏）。

> **2026-08-31 上游同步（v1.4.0）**：retarget 到 `DeepSeek-V4-Flash-Vision-Exp`
> （上游 PR #164 `feat/vision-exp`，快照 `de230b45bc49…`），模型/服务名/镜像 tag 全部
> 切换，`MTP_NUM_TOKENS` 默认 5 → 6（Vision-Exp `n_predict=3`），新增
> `LIMIT_MM_PER_PROMPT`（每请求图片上限）。相对 `v0.1.1-hotfix2`（快照 `0107cef`），
> 镜像还新增：Vision-Exp 原生图片支持（#164）、`<image>` 配对标签角色检查（#165）、
> tool 结果文本不再误判图片（#167）。**`v0.1.1-hotfix3` 已烘焙推送 ACR**
> （manifest `sha256:e9c9dca7…`，单架构 arm64，2026-09-01），容器内干跑验证：
> 热修复链序完整、vision 三件套（model/encoding/dspark）APPLIED、
> `--limit-mm-per-prompt {"image":N}` 转换与 `MTP=42` capture 尺寸正确、
> 缺 checkpoint encoding 时 fail-closed 拒启。**checkpoint 分发仍需实机验证**，
> 发布前请确认「模型页」已分发 `DeepSeek-V4-Flash-Vision-Exp` 到两节点。

同时持久化 Triton / TileLang / B12X-CuTeDSL JIT 编译缓存到 HF 卷（容器重建不重复
JIT，避免 TP 失同步）。

## 已知问题

- `MTP_NUM_TOKENS` 勿低于 6：Vision-Exp `num_nextn_predict_layers=3`，k 须 ≥
  `dspark_block_size`(5) 且整除 3，k=5 会被 Anemll 直接拒绝。
- 图片仅放 **user 消息**：system/assistant 消息带 `image`/`image_url` 部分返回
  HTTP 400（对齐官方 Chat Completions 限制）；纯文本提到 `<image>` 标签没问题
  （#165 已修，hotfix3 生效）。
- **无视频**：官方权重没有视频编码器，GIF 按静帧解码。
- b12x 栈 1M 上下文显存压力大，Vision-Exp 权重比 0731 更占显存、KV 池更小，
  `GPU_MEMORY_UTILIZATION` 默认 0.835；显存紧张或图形捕获 OOM 时可降至 ~0.80。
- `DEFAULT_THINKING=max` 的推理可能很长（实测中等提示下推理可达 ~12.5k token /
  数万字符），需给足 `max_tokens`（万级），或按请求改用 `low`/`off`。
- 并发长 prefill 仍会排队（#27 语义）：本配方以 1024 阈值分块保证 decode 不被挤占，
  但无法并行服务多个超长冷预填充。
- 并发稳定性（上游 #141/#143）：sparse-MLA decode 卡死是**逐 burst 随机**的，目前
  **没有任何一个 `MAX_NUM_SEQS` 值被证明普遍安全**——改动它只改变出现概率，不是修复。
  最容易被忽视的故障信号不是重启，而是缺失 `finish_reason` 的静默流截断。默认
  N=6/k=6 的 verify 行数为 42（6×7），引擎可能把 CUDA graph 捕获行数钳到 ~32；
  提高并发前请先读上游 #141 证据与 #151 的 opt-in workaround。
- 仅支持恰好 2 节点（TP=2）；其他拓扑请选对应配方或自建。
- **dev 分支配方**：v1.4.0 依赖 `v0.1.1-hotfix3` 镜像（已推送 ACR，2026-09-01）与
  `DeepSeek-V4-Flash-Vision-Exp` checkpoint 分发（**未实机验证**）；发布前请确认
  「模型页」已把 checkpoint 分发到两节点。

## 参考来源

完整来源见仓库根 `NOTICE.md`：

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  （vision-exp · 快照 `d58c877`，2026-08-31）
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
- 上游 checkpoint/编码器文档：[docs/DEEPSEEK_V4_FLASH_0731.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)
- 上游实测数据：[results/RESULTS-2026-08-14.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/results/RESULTS-2026-08-14.md)
