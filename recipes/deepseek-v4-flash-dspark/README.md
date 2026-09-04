# DeepSeek-V4-Flash DSpark · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上以 **TP=2** 服务
DeepSeek-V4-Flash（1M 上下文）。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（~167 GB，Fireworks 分发后离线加载）
- 量化/投机：NVFP4 DS-MLA · FlashInfer b12x + dspark 投机（k=5）· **1M 上下文**
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix6`
  （Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + Mia fail-closed 热修复链）
- 对外服务名：`deepseek-v4-flash-0731`；API 端口默认 `8888`
- 默认思考 `low`（请求级可覆盖 off/low/high/max）

## 速度

未附本地实测（上游 benchmark 见参考来源）。

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），RoCE 组网
- `GPU_MEMORY_UTILIZATION` 默认 0.835；1M 上下文显存吃紧时可降至 ~0.80

## 参考上游

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
