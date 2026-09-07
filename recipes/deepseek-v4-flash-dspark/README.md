# DeepSeek-V4-Flash DSpark · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上以 **TP=2** 服务
DeepSeek-V4-Flash（1M 上下文）。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（~167 GB，Fireworks 分发后离线加载）
- 量化/投机：NVFP4 DS-MLA · FlashInfer b12x + dspark 投机（k=5）· **1M 上下文**
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix7`
  （Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + Mia fail-closed 热修复链，上游快照
  `957890ac…` 2026-09-06）
- 对外服务名：`deepseek-v4-flash-0731`；API 端口默认 `8888`
- 默认思考 `low`（请求级可覆盖 off/low/high/max）

## 09-05/09-06 新增 opt-in 热修复（hotfix7，默认关）

镜像已 bake 上游最新补丁树，以下开关默认 `0`，需要时按环境变量开启（配方暴露其中 4 项）：

- `DSPARK_ENABLE_DSPARK_SWA_PREFIX`=1：前缀缓存命中时重算草稿滑窗（Anemll #2 移植，避免
  重复相同 prompt 退化成截断输出）——**正确性修复，建议开**
- `DSPARK_ENABLE_DSML_RECOVERY`=1：裸 `<invoke name=...>` 按声明工具校验后提交（vllm#52645）
- `DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED`=1：命名/required tool_choice 严格契约
- `DSPARK_ENABLE_C128A_PREFILL_CACHE`=1：复用 C128A prefill 索引转换（SM120，无端到端提速承诺）
- 其余（`DSPARK_ENABLE_ROPE_SWA_FIX` / `DSPARK_ENABLE_DSPARK_BLOCK_K` /
  `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN` / `DSPARK_ENABLE_MXFP4_INDEXER_CACHE`，MXFP4 需
  配 `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`）由环境注入即可

## 速度

未附本地实测（上游 benchmark 见参考来源）。

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），RoCE 组网
- `GPU_MEMORY_UTILIZATION` 默认 0.835；1M 上下文显存吃紧时可降至 ~0.80

## 参考上游

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
