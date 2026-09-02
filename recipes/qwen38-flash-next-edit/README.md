# Qwen3.8-Flash-Next · Edit · 单节点 llama.cpp · Fireworks 配方（1× DGX Spark）

用 Fireworks 在 **1 台** DGX Spark（GB10，128 GiB 统一内存）上跑起 **Qwen3.8-Flash-Next**
的 **llama.cpp 编程编辑路线**（源仓库 `edit` 配置的容器化版），适合「编码 agent 反复改写
你交付的文件」，**完整 262,144 token 上下文**。

## 模型

- 主模型：`unsloth/Qwen3.8-Flash-Next-GGUF` 的 **UD-Q4_K_XL** 分片（~104 GiB）+
  `mmproj-F16.gguf`（~0.9 GiB 视觉投影，可选）
- 关键机制同 vLLM 配方：51.2B n-gram/PLE 查找表留 **NVMe 页缓存**（`-lm mmap` +
  `-ot per_layer_token_embd=CPU`），不放 GPU
- 投机：**ngram-mod** 上下文复制投机（一次验证 60 token，**exact**——输出与不投机逐字节
  一致）；**无 MTP**（GGUF 转换器丢掉训练草稿头）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0`
- 服务端口默认 `8000`
- 与 vLLM 配方区别：无需下载 NVFP4 checkpoint、可选中更低比特量化省磁盘；聊长文档/高强度
  问答请用 [`qwen38-flash-next-vllm`](../qwen38-flash-next-vllm/README.md) 配方（更稳定，
  prefill 快 5 倍；共用 GPU，同一时刻只跑一个）

## 速度

实测（源仓库）：

- **速度按任务读，别拿单一数字对比**：改文件（答案全在 prompt 里）**88 tok/s**，写新文本
  （无可复制内容）**~28 tok/s**
- 投机 exact：输出与关投机逐字节一致，不是用质量换速度
- `--parallel 2` 并发下负载约 1.24–1.30×、单调用者零成本；思考 token 占大头（同 vLLM 配方，
  请求级关思考 ~55 s → ~15 s）

## 硬件需求

- **1 台** DGX Spark（固定单节点 · GB10，128 GiB 统一内存）
- 上下文按槽平分：两槽下超长 prompt 返回 400（不截断）——喂超长文档设 `PARALLEL=1` 或改
  用 vLLM 配方
- **磁盘**：UD-Q4_K_XL ~104 GiB + 镜像 ~4–6 GiB

## 参考上游

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  （MIT）：edit 路线的容器 Dockerfile、服务配置与实测（`recipes/llamacpp-edit/`）
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)：
  UD-Q4_K_XL 量化与 mmproj
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)：引擎

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
