# DeepSeek-V4-Flash-Vision-Exp · Spark-vLLM b12x · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上以 **TP=2** 服务
DeepSeek-V4-Flash-Vision-Exp（1M 上下文，原生图片输入）。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`（Fireworks 分发后离线加载）
- 后端/量化：**B12X** 注意力 + **b12x** MoE/线性层（sparse indexer）· **FP8 KV** · dspark
  投机（k=6）· 1M 上下文；instanttensor + AOT 编译（首启快）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0`
  （自烘焙：eugr `eugr/spark-vllm-b12x` nightly-20260909 基镜像 + `instanttensor-hybrid-draft-loader`
  patch 在构建期打入，见 [`docker/spark-vllm-b12x`](../../docker/spark-vllm-b12x/README.md)；
  对应上游 `--exp-b12x` 视觉构建）
- API 端口默认 `8000`；默认思考模式 `high`（对齐上游 `thinking=true / effort=high`）

## 速度

未附本地实测（该路线尚未实机验证，参数对齐上游 2026-09-08 新增的 vision-exp 配方，
即 2026-09-03 新 b12x 分支：`--attention-backend B12X`、capture 48、MEGA_AOT_ARTIFACT 1）。

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），RoCE 组网
- 并发上限 `MAX_NUM_SEQS=8`（源命令值）；`GPU_MEMORY_UTILIZATION` 默认 0.85
  （既有配方实机验证 0.80 稳定，0.90 无法开机）
- `LOAD_FORMAT=instanttensor` 依赖节点缓存中的 instanttensor 布局（标准 HF safetensors
  分发需改回 auto/safetensors）

## 与上游的差异

- 上游 `mods/instanttensor-hybrid-draft-loader`（target 走 InstantTensor、dspark 草稿切
  lazy safetensors）已**烘焙进镜像**（构建期 patch，`INSTANTTENSOR_DRAFT_LOADER=auto`），
  不需要上游 `--apply-mod` 的运行时步骤；基镜像钉在 09-09 nightly，patch 仅在基镜像
  vLLM 源码匹配时构建成功（fail-fast）
- 上游 `--reasoning-config` 分隔符为空串，此处按已验证的 dspark 配方修正为
  `" thinking"` / `" response"`（否则思考段落无法正确切分）
- `--max-model-len` 由上游 `auto` 固定为 1048576；多节点协调使用 Fireworks 自动变量
  （`--nnodes/--node-rank/--master-addr/--headless`，上游为宿主 shell 变量）
- 继承既有 b12x 配方的 RoCE/NCCL 集成层与 ulimit 修正（nofile=1048576）

## 参考上游

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)（MIT）：
  `recipes/deepseek-v4-flash-vision-exp.yaml`、`recipes/deepseek-v4-flash-0731.yaml`
- `eugr/spark-vllm-b12x:latest`：分发镜像（`--exp-b12x` 构建）
- [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
