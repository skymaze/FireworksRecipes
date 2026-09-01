# Qwen3.8-Flash-Next · Edit · 单节点 llama.cpp · Fireworks 配方（1× DGX Spark）

用 Fireworks 在 **1 台** DGX Spark（GB10，128 GiB 统一内存）上跑起 **Qwen3.8-Flash-Next**
的 **llama.cpp 编程编辑路线**（源仓库 `edit` 配置的容器化版）：适合「编码 agent 反复改写
你交付的文件」这类工作，**完整 262,144 token 上下文**：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0`
  （源自上游 `recipes/llamacpp-edit/Dockerfile` 烘焙：CUDA 13 devel 构建 llama.cpp
  @qwen4exp PR `035e227` + runtime 精简，只留 `llama-server` / `llama-cli`）
- 拓扑：**固定单节点**（GB10 单 GPU）
- 模型：`unsloth/Qwen3.8-Flash-Next-GGUF` 的 **UD-Q4_K_XL** 分片（~104 GiB）+
  `mmproj-F16.gguf`（~0.9 GiB 视觉投影，可选）
- 关键机制同 vLLM 配方：176.9B 参数中 **51.2B n-gram/PLE 查找表从不做运算**，本路线用
  `-lm mmap` + `-ot per_layer_token_embd=CPU` 把它**留在 NVMe 页缓存**上，不放 GPU
- 投机：**ngram-mod** 上下文复制投机——从 prompt 里找重复跨度一次验证 60 token，**exact**、
  输出与不投机逐字节一致（改文件 88 tok/s、自由文本 ~28 tok/s）
- 与 vLLM 配方的区别：GGUF 转换器丢掉了模型的训练 MTP 草稿头（本配方无 MTP、无可跨引擎
  投机），但无需下载 NVFP4 checkpoint、且可选中更低比特量化省磁盘

> 这是 [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
> 的 **edit**（llama.cpp）路线；聊长文档/高强度问答请用同仓库
> [`qwen38-flash-next-vllm`](../qwen38-flash-next-vllm/README.md) 配方（更稳定、prefill 快
> 5 倍）。两条路线共用 GPU，同一时刻只跑一个。

## 镜像（发布前必须完成）

镜像由本仓库烘焙并推送到
`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0` 后，Fireworks
才能拉取分发。烘焙来源 = 上游 `recipes/llamacpp-edit/Dockerfile`（CUDA 13 devel 阶段构建
llama.cpp @qwen4exp PR `035e227` + runtime 精简）：

> ```bash
> cd <上游 0xBakeer/qwen38-flash-next-spark 克隆>
> docker build -t registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0 \
>   --label "de.qwen38fn.llamacpp-ref=pinned@035e227" \
>   -f recipes/llamacpp-edit/Dockerfile .
> docker push registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0
> ```

> **构建状态（2026-09 尝试）：** 两个 CUDA 基座层已全部下载至本地 Docker 缓存（构建曾推进到
> runtime apt 阶段），但本机网络当前无法完成——Docker Hub 直连 IPv6 超时、Docker Desktop
> 残留静态代理 `127.0.0.1:1082`（指向已停用的 Clash 客户端）使镜像兜底失败。**镜像尚未
> 烘焙/推送**；网络恢复后按上面命令继续即可（基座层已缓存，将直接进入 llama.cpp 编译）。

## 快速使用

1. 模型：把 `unsloth/Qwen3.8-Flash-Next-GGUF` 的 `UD-Q4_K_XL/*`（+ 需要图像时
   `mmproj-F16.gguf`）分发到节点 HF 缓存（管理平台模型分发）。
2. 发布：选本配方 + 1 节点集群，`nodes=1` 恰好匹配。
3. 验证：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50}'
```

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLAMACPP_IMAGE` | `.../qwen38-flash-next-edit:v1.0.0` | llama.cpp 镜像（ACR 烘焙） |
| `QWN38_GGUF_REPO` | `unsloth/Qwen3.8-Flash-Next-GGUF` | GGUF 仓库（只需 UD-Q4_K_XL + mmproj 分发） |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | 对外服务名 |
| `LLAMACPP_PORT` | `8000` | llama-server 端口 |
| `MAX_MODEL_LEN` | `262144` | 上下文长度（按 PARALLEL 平分） |
| `PARALLEL` | `2` | 并发槽（2 = 每请求 131,072；1 = 全窗） |
| `SPEC` | `ngram-mod` | 投机模式（`none` 关闭） |
| `MMPROJ_MODE` | `auto` | 缓存有 mmproj-F16.gguf 即传 `--mmproj`（`none` = 纯文本） |
| `EXTRA_ARGS` | 空 | 追加到命令尾部的 llama-server 参数 |
| `HF_TOKEN` | 空 | 受 gating 模型授权 token（分发到缓存通常可留空） |

`NODES_TOTAL=1`；单节点无分布式变量。

## 与源 Dockerfile / run.sh 的差异（Fireworks 集成层）

- **快照解析**：镜像 CMD 用固定的 `/model/...` 路径；本配方改为从节点 HF 缓存
  （`HF_HOME=/hf`）按 repo id 解析 `snapshots/*/UD-Q4_K_XL/*00001-of-*.gguf`（含 mmproj
  自动探测），模型由管理平台分发、离线加载。
- **端口**：容器内 `--port ${LLAMACPP_PORT:-8000}`（镜像 CMD 默认 8000 保留）；
  host 网络 + `--host 0.0.0.0`。
- **`-v ~/.cache/huggingface:/hf`**：与源容器化版本一致，`HF_HOME=/hf`、
  `HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_XET=1`。
- 投机/并行/视觉与否做成开关变量（`SPEC=ngram-mod` / `none`、`PARALLEL`、`MMPROJ_MODE`），
  与源行为一致；`-lm mmap`、`-ot per_layer_token_embd=CPU`、`--n-gpu-layers 999`、
  `--temp 1.0 --top-p 0.95 --top-k 20` 为固定默认（可在 `EXTRA_ARGS` 覆盖）。

## 行为与调参（源自源仓库实测）

- **3.2× 任务间吞吐落差是「投机来源」决定的**：改文件时答案几乎全在 prompt 里（88 tok/s），
  写新文本没有可复制内容（~28）。**速度按任务读，别拿单一数字对比。**
- **投机 exact**：目标模型验证每个草稿 token，输出与关投机逐字节一致，不是用质量换速度。
- **不要在冷/热页缓存上纠结**：`WARM` 预热实测无收益（17.7% 与 58.1% 驻留都是 27.8 tok/s）。
- **别为速度换低比特 K-quant**：UD-Q3_K_XL 移动字节少 19% 却慢 14%（本配置非带宽受限）。
  换小量化只该为了磁盘/质量。
- **f16 KV**：量化 KV 在本架构上 abort，保持 f16（~24 KB/token，很便宜）。
- **两请求，不是两百**：`--parallel 2` 是并发（各占半窗上下文），负载下 1.24–1.30×、单
  调用者零成本；`PARALLEL=1` 恢复整窗。64 槽会抖坏 prompt cache 并丢请求。
- **思考 token**：同 vLLM 配方——86% 生成 token 是 reasoning，请求级
  `{"chat_template_kwargs":{"enable_thinking":false}}` 可把回答从 ~55 s 压到 ~15 s。

## 已知问题与部署注意

- 上下文按槽平分：两槽下超长 prompt 返回 400（不截断）；`prefill-128k`（161K token）在两槽
  全失败、单槽可容纳——喂超长文档时设 `PARALLEL=1` 或改用 vLLM 配方。
- **无 MTP**：GGUF 无训练草稿头，长文档/稳定吞吐场景用 vLLM 配方更优。
- 磁盘：UD-Q4_K_XL ~104 GiB + 镜像 ~4-6 GiB。
- 单节点配方：只支持 1 节点，不要分配多节点拓扑。

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  （MIT）：edit 路线的容器 Dockerfile、服务配置与实测（`recipes/llamacpp-edit/`）
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)：
  UD-Q4_K_XL 量化与 mmproj（权重携带 Qwen 自己的许可）
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)：引擎
