# Qwen3.8-Flash-Next · 单节点 vLLM · Fireworks 配方（1× DGX Spark）

用 Fireworks 在 **1 台** DGX Spark（GB10，128 GiB 统一内存）上跑起 **Qwen3.8-Flash-Next**
（176.9B 参数）的 vLLM 服务，**完整 262,144 token 原生上下文**。

## 模型

- 主模型：`RadixArk/Qwen3.8-Flash-Next-NVFP4`（NVFP4 checkpoint，~122 GiB，保留全视觉塔
  与全部 31 个 MTP 张量；含多模态，atlas 图像评测 0.967）
- 关键机制：176.9B 中 **51.2B n-gram/PLE 查找表从不做运算**，本配方留 NVMe 流式加载
  （`VLLM_PLE_MMAP=1`）——这是 122 GiB checkpoint 能与 KV 池同住一台、保持全上下文的原因
- 投机：模型自带训练 MTP 草稿头（k=3）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0`
- 服务端口默认 `8000`；served-model-name `qwen3.8-flash-next`；OpenAI 兼容 API
- 来源：`0xBakeer/qwen38-flash-next-spark` 的 **longctx**（vLLM）路线；编码改写文件场景请用
  姊妹配方 [`qwen38-flash-next-edit`](../qwen38-flash-next-edit/README.md)（两条路线共用 GPU，
  同一时刻只跑一个）

## 速度

实测（源仓库）：

- 单流：自由文本 **~32 tok/s**（MTP 在批未饱和时 +35%，17→27 tok/s 量级）
- **16 是并发墙**：16 并发内 TTFT < 2.7 s、聚合 96–109 tok/s；32 时 TTFT 16 s、64 时 70 s
  （批量任务可设 64，聚合再 +35%）
- 前缀缓存默认开：共享前缀 +76% 聚合、TTFT 减半以上
- 思考 token 占大头（86%）：请求级关思考让同一回答从 ~55 s 降到 ~15 s

## 硬件需求

- **1 台** DGX Spark（固定单节点 · GB10，128 GiB 统一内存）
- gmu 0.85 实测 KV 池 641,601 token（**勿降回上游默认 0.78**，否则单条全长请求放不下）
- 首启读 ~83 GiB 权重需 12–15 分钟；**磁盘**：checkpoint ~126 GB + 镜像 ~21 GB，发布前
  确认节点 ≥ ~150 GB 空闲

## 参考上游

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  （MIT）：服务配置、调参与实测数字来源（longctx 路线）
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)（Apache-2.0）：
  含 PLE-mmap 等 7 个补丁的 vLLM 容器（镜像烘焙来源）
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) ·
  [Qwen](https://qwen.ai)（模型与 n-gram/PLE 技术报告）

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
