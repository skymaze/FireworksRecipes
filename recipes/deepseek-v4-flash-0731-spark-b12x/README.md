# DeepSeek-V4-Flash-0731 · Spark-vLLM b12x · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上以 **TP=2** 服务
DeepSeek-V4-Flash-0731（1M 上下文）。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（Fireworks 分发后离线加载）
- 后端/量化：**B12X MLA SPARSE** 注意力 + **b12x** MoE/线性层 · **FP8 KV** · dspark 投机
  （k=5）· 1M 上下文；instanttensor + AOT 编译（首启快）
- 镜像：`eugr/spark-vllm-b12x:latest`（spark-vllm b12x 预构建 vLLM 分发镜像）
- API 端口默认 `8000`；默认思考模式 `high`

## 速度

未附本地实测（该路线尚未实机验证，见参考来源）。

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），RoCE 组网
- 并发上限 `MAX_NUM_SEQS=8`（源命令值）；`GPU_MEMORY_UTILIZATION` 默认 0.85
  （既有配方实机验证 0.80 稳定，0.90 无法开机）
- `LOAD_FORMAT=instanttensor` 依赖节点缓存中的 instanttensor 布局（标准 HF safetensors
  分发需改回 auto/safetensors）

## 参考上游

- `eugr/spark-vllm-b12x:latest`：分发镜像（源 docker run 引用）
- [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
