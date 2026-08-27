# Qwen3.8-27B · SGLang DSPARK · Fireworks 配方（单节点 DGX Spark）

用 Fireworks 在 **1 台** DGX Spark 上跑起 **RadixArk/Qwen3.8-27B-NVFP4** 的 SGLang 服务：

- 镜像：`lmsysorg/sglang:qwen38-27b`（官方 SGLang 镜像，model 专用 tag）
- 拓扑：**固定单节点**（GB10 单 GPU）
- 模型：`RadixArk/Qwen3.8-27B-NVFP4`（NVFP4 4-bit 权重）+ `--kv-cache-dtype fp8_e4m3`
- 投机解码：**DSPARK**（`RadixArk/Qwen3.8-27B-DSpark` mamba 草稿模型，草稿注意力 flashinfer）
- 注意力/prefill：flashinfer、`--chunked-prefill-size 2048`、`--mem-fraction-static 0.85`
- 服务端口默认 `30000`；reasoning/tool parser 用 `qwen3` / `qwen3_coder`
- 模型由 Fireworks 分发到节点 HF 缓存后**离线加载**（`HF_HOME=/root/.cache/huggingface`）

> 来源是一条单机 `docker run`（端口映射 + HF 缓存挂载）。本配方按 Fireworks compose 体系
> 移植：`-p 30000:30000` → host 网络 + `--port 30000`，`~/.cache/huggingface` 挂载保留，
> `HF_TOKEN` 占位符做成可选变量。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_IMAGE` | `lmsysorg/sglang:qwen38-27b` | SGLang 镜像 |
| `SGLANG_MODEL` | `RadixArk/Qwen3.8-27B-NVFP4` | 主模型（已分发到缓存） |
| `SGLANG_DRAFT_MODEL` | `RadixArk/Qwen3.8-27B-DSpark` | DSpark mamba 草稿模型（需已分发） |
| `SGLANG_PORT` | `30000` | SGLang API 端口 |
| `MEM_FRACTION_STATIC` | `0.85` | `--mem-fraction-static` |
| `CHUNKED_PREFILL_SIZE` | `2048` | `--chunked-prefill-size` |
| `MAMBA_FULL_MEMORY_RATIO` | `11.01` | `--mamba-full-memory-ratio`（**疑似笔误，见下**） |
| `HF_TOKEN` | 空 | 受 gating 模型授权 token（分发到缓存通常可留空） |

`NODES_TOTAL=1`；单节点无分布式变量（无 `NODE_RANK/HEADLESS/NCCL_*` 需要）。

## 与源命令的差异

- **`--mamba-full-memory-ratio 11.01`（重点核对）**：该参数是「mamba 全量计算预留显存比例」，
  取值应落在 `[0,1]`。源命令的 `11.01` 明显不合理，很可能是 **`0.1101`（即 11.01%）** 或
  `0.11` 的笔误。保留为 `MAMBA_FULL_MEMORY_RATIO` 变量（默认仍 11.01），**实机发布前务必
  按 SGLang 文档/实测修正**，否则可能启动即失败或异常占用显存。
- `-p 30000:30000` → host 网络 + `--host 0.0.0.0 --port 30000`（Fireworks 配方统一 host 网络）。
- `~/.cache/huggingface:/root/.cache/huggingface` 挂载保留，并把 `HF_HOME` 指到
  `/root/.cache/huggingface`（SGLang 默认值），加 `HF_HUB_OFFLINE=1`（平台分发后离线加载）。
- `HF_TOKEN=<your-hf-token>` 占位符 → `HF_TOKEN` 可选变量（默认空）；若镜像/模型在启动时
  校验 gating，发布时填写。
- `--shm-size 32g` → compose `shm_size: "32gb"`；`--ipc=host`、`--gpus all` 原样映射。
- 其余命令行参数（`fp8_e4m3`、backends、DSPARK、mamba 相关）原样保留。

> 本配方是**仓库内首个 SGLang 配方**，无同引擎实机配方可对照；以上仅做格式级审查，
> 请实机验证后再按 README 分支模型并入 `main`。

## 已知问题与部署注意

- **`MAMBA_FULL_MEMORY_RATIO>1`**：最可能的配置错误，见上。
- **上下文长度**：源命令未指定，由模型 config 决定；商店卡片 `context_length` 相应未固化。
- **API 的 model id**：SGLang 默认以模型路径为 `model` id（`RadixArk/Qwen3.8-27B-NVFP4`），
  客户端请求需传该值（或服务端自行加 `--served-model-name`）。
- **草稿模型需分发**：`RadixArk/Qwen3.8-27B-DSpark` 必须也在节点缓存，否则投机解码启动失败。
- 单节点配方：只支持 1 节点，不要给它分配多节点拓扑。

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方组件来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- `lmsysorg/sglang:qwen38-27b`：官方 SGLang 镜像（源 docker run 引用）
- [sgl-project/sglang](https://github.com/sgl-project/sglang)：引擎
- [RadixArk](https://huggingface.co/RadixArk)：模型与 DSpark 草稿模型
