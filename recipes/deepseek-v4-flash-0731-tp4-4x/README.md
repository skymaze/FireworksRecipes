# DeepSeek-V4-Flash-0731 · TP=4 · Fireworks 配方（4× DGX Spark）

用 Fireworks 在 **4 台** DGX Spark（head + 3 worker，RoCE 组网）上以 **TP=4** 服务
DeepSeek-V4-Flash-0731（1M 上下文），默认参数按 agentic 工作负载实机验证固化。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（~167 GB，Fireworks 分发后离线加载）
- 量化/投机：NVFP4 DS-MLA · FlashInfer b12x + dspark 投机（k=5）· **1M 上下文**
- 镜像：`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（Anemll 预构建 vLLM 分发镜像）
- API 端口默认 `8888`

## 速度

未附本地实测（上游 benchmark 见参考来源）。

## 硬件需求

- **4 台** DGX Spark（固定 4 节点 · TP=4），每机 1 GPU（GB10），RoCE 组网
- 实机验证默认：`GPU_MEMORY_UTILIZATION=0.80`、`DEFAULT_THINKING=max`、1M 上下文
- 并发上限 `MAX_NUM_SEQS=4`（`max_cudagraph_capture_size` 自动按 4×6 计算）

## 参考上游

- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
