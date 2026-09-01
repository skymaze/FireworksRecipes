# Qwen3.8-Flash-Next · 单节点 vLLM · Fireworks 配方（1× DGX Spark）

用 Fireworks 在 **1 台** DGX Spark（GB10，128 GiB 统一内存）上跑起 **Qwen3.8-Flash-Next**
（176.9B 参数）的 vLLM 服务，**完整 262,144 token 原生上下文**：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0`
  （**dev：待按上游烘焙并推送 ACR 后配方方可发布**，见下「镜像」节）
- 拓扑：**固定单节点**（GB10 单 GPU）
- 模型：`RadixArk/Qwen3.8-Flash-Next-NVFP4`（NVFP4 checkpoint，~122 GiB，保留**全视觉塔**
  与**全部 31 个 MTP 张量**）
- 关键机制：176.9B 参数中 **51.2B 是一个从不做运算、只被读取的 n-gram/PLE 查找表**，
  本配方把它**留在 NVMe 上流式加载**（`VLLM_PLE_MMAP=1`）——这正是 122 GiB checkpoint 能
  和可用 KV 池同住一台机器、且保持全上下文的原因
- 投机解码：模型自带**训练 MTP 草稿头**（k=3），任何文本上表现平稳（自由文本 ~32 tok/s）
- 多模态：NVFP4 checkpoint 含完整视觉塔（333 张量、未量化），atlas 图像评测 0.967
- 服务端口默认 `8000`；served-model-name `qwen3.8-flash-next`；OpenAI 兼容 API

> 本配方移植自 [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
> 的 **longctx**（vLLM）路线——其 README 自述的推荐默认（「Not sure → Writing & long
> documents」）。源仓库另一条 **edit**（llama.cpp）路线需在宿主上编译，不在 Fireworks
> 容器分发模型范围内，本配方不覆盖。

## 镜像（发布前必须完成）

上游 `blazux/qwen3.8-Flash-DGX`（Apache-2.0）**不发布现成镜像**：其 `setup.sh` 在本地
`docker build`（基底 `vllm/vllm-openai:qwen38-flash-next@sha256:fc120e…`，叠加 7 个补丁，
其中 PLE-mmap 补丁是本配方成立的前提）。按本仓库惯例，把该镜像烘焙并推送到
`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0` 后，Fireworks
才能拉取分发。

> dev 阶段镜像尚未烘焙——**发布前请先构建推送**，并把镜像内上游 commit
> （`de.qwen38fn.upstream-sha` label）记录到本 README，便于复现与排障。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_IMAGE` | `.../aixn-public/qwen38-flash-next:v1.0.0` | vLLM 镜像（待烘焙） |
| `QWN38_MODEL` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 主模型（已分发到节点 HF 缓存） |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | 对外服务名 |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `262144` | 上下文长度（模型原生最大） |
| `MAX_NUM_SEQS` | `16` | 并发上限（16 是最后一段服务良好的档位） |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批最大 token |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 0.85 实测 KV 池 641,601 token（勿用上游默认 0.78） |
| `MTP` | `3` | MTP 投机 token 数（0 = 关；>3 更差） |
| `PREFIX_CACHE` | `1` | 前缀缓存（共享前缀 +76%；0 = 复现源仓库发布的 prefill 数字） |
| `PLE_MMAP_WORKERS` | `32` | NVMe 流式预取并发 worker 数 |
| `PLE_MMAP_PREWARM` | `1` | 启动时预热 PLE 表（首个请求不冷） |
| `HF_TOKEN` | 空 | 受 gating 模型授权 token（分发到缓存通常可留空） |

`NODES_TOTAL=1`；单节点无分布式变量（无 `NODE_RANK/HEADLESS/NCCL_*` 需要）。

## 与源 serve.sh 的差异（Fireworks 集成层）

- **快照路径 → repo id**：源脚本把 `HF_CACHE/hub/models--…/snapshots/<sha>/` 精确路径传给
  vLLM；本配方按 Fireworks 惯例传 repo id（`RadixArk/Qwen3.8-Flash-Next-NVFP4`）+
  `HF_HOME=/hf` + `HF_HUB_OFFLINE=1` 离线解析（模型由管理平台分发到节点缓存）。
- `-p 127.0.0.1:8000:8000` → host 网络 + `--host 0.0.0.0 --port 8000`（Fireworks 统一 host
  网络；如需限制访问，在集群/防火墙层处理而非容器绑定）。
- `-v ~/.cache/huggingface:/hf` 挂载保留，`HF_HOME=/hf`、`HF_HUB_OFFLINE=1`、
  `HF_HUB_DISABLE_XET=1`（源 setup 即禁用 Xet，避免部分 Spark 上假完成）。
- `--shm-size 16g` → `shm_size: "16gb"`；`--ipc=host`、`--gpus all` 原样映射。
- 其余命令行（`PIECEWISE` CUDA-graph 捕获 + splitting_ops 清单、`--no-enable-flashinfer-autotune`、
  MTP 投机、tool/reasoning parser、`--max-num-batched-tokens 8192`）原样保留；`MTP=0` 与
  `PREFIX_CACHE=0` 做成开关变量，行为与源完全一致。

## 行为与调参（源自源仓库实测）

- **MTP 投机只在批未饱和时有收益**：单流 +35%（17→27 tok/s 量级），16 并发下不可测
  （批一满，投机不再转换为吞吐）。k>3 反而更差。
- **16 是并发墙**：16 并发内 TTFT < 2.7 s、聚合 96–109 tok/s；32 时 TTFT 16 s，64 时 70 s。
  批量任务（无人等首 token）可设 `MAX_NUM_SEQS=64`，聚合再 +35%。
- **前缀缓存默认开**：多轮对话/agent 反复带同一 system prompt / 共享前缀时 +76% 聚合、
  TTFT 减半以上；无关 prompt 的工作负载无收益。源仓库发布的所有 prefill 数字都是无缓存
  测得，`PREFIX_CACHE=0` 才能复现。
- **思考 token 占大头**：源实测 86% 的生成 token 是 reasoning。请求级关思考
  `{"chat_template_kwargs":{"enable_thinking":false}}` 让同一回答从 ~55 s 降到 ~15 s。
- **KV 池**：gmu 0.85 时 18.13 GiB = 641,601 token（≈2.4× 满上下文）；上游默认 0.78 只有
  227,651 token，单条全长请求放不下——**不要降回 0.78**。
- **首次启动慢**：读 ~83 GiB 权重 12–15 分钟（`VLLM_ENGINE_READY_TIMEOUT_S=3600`、
  healthcheck `start_period: 900s`）。之后重启更快。
- **磁盘**：checkpoint ~126 GB + 镜像 ~21 GB，发布前确认节点 ≥ ~150 GB 空闲。

## 已知问题与部署注意（源自源仓库）

- **CUDA-graph 只能 PIECEWISE**：n-gram 查表是 CPU op + host→device 拷贝，必须声明为
  splitting op 并在 PIECEWISE 模式下捕获；切到 FULL 捕获会坏。
- **`VLLM_PLE_CPU_OFFLOAD=1` 会挂**：该路径期待一个本镜像不启动的 offload worker，无限
  空转；**只有 `VLLM_PLE_MMAP` 可用**（本配方已固定）。
- **前缀缓存与块大小**：修复在 blazux 容器 `8347e7c`（2026-08-29）之后；镜像烘焙时务必
  取该 commit 之后的版本，早期镜像上不要开前缀缓存。
- **推理**：`VLLM_TORCH_PROFILER_DIR` 在该构建 inert（已迁移到 `--profiler-config`），勿依赖。
- 单节点配方：只支持 1 节点，不要分配多节点拓扑。
- 大 checkpoint 分发走平台模型分发通道（HF 缓存挂载 `/hf`），离线加载。

## 验证

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方组件来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  （MIT）：本配方的服务配置、调参与实测数字来源（longctx 路线）
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)（Apache-2.0）：
  PLE-mmap 等 7 个补丁的 vLLM 容器（镜像烘焙来源）
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)：
  NVFP4 checkpoint（权重携带 Qwen 自己的许可）
- [Qwen](https://qwen.ai)：Qwen3.8-Flash-Next 模型与 n-gram/PLE 技术报告
