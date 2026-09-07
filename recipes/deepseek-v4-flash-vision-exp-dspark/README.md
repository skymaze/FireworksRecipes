# DeepSeek-V4-Flash-Vision-Exp DSpark · Fireworks 配方（2× DGX Spark）

用 Fireworks 在 **2 台** DGX Spark（head + 1 worker，RoCE 组网）上以 **TP=2** 服务
DeepSeek-V4-Flash-**Vision-Exp**（DeepSeek V4 家族首个实验多模态模型，1M 上下文）。

## 模型

- 主模型：`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`（~305B 含 ViT+Aligner，MIT；
  ~167 GB，Fireworks 分发后离线加载）
- 多模态：原生**图片输入**（OpenAI `image_url`，JPEG/PNG/GIF/WebP；GIF 取静帧；每请求
  默认 ≤ 8 张，**仅 user 消息可带图**；官方权重**无视频编码器**）
- 量化/投机：NVFP4 DS-MLA · FlashInfer b12x + dspark 投机（k=6）· 1M 上下文
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix7`
  （Anemll + Mia 热修复链 + Vision-Exp 原生图片支持，上游快照 `957890ac…` 2026-09-06；含
  09-05 image-cap 修复：`--limit-mm-per-prompt` 真正约束每请求图片上限）
- 对外服务名：`deepseek-v4-flash-vision-exp`；API 端口默认 `8888`

## 09-05/09-06 新增 opt-in 热修复（hotfix7，默认关）

- `DSPARK_ENABLE_DSPARK_SWA_PREFIX`=1：前缀缓存命中时重算草稿滑窗（Anemll #2 移植）——
  **正确性修复，建议开**
- `DSPARK_ENABLE_DSPARK_BLOCK_K`=1：投机 k 跟随训练 `dspark_block_size`(5)——Vision-Exp 可跑
  k=5 而非强制 6（`num_nextn_predict_layers=3`），要把 `MTP_NUM_TOKENS` 同步调到 5
- `DSPARK_ENABLE_C128A_PREFILL_CACHE`=1：复用 C128A prefill 索引转换（SM120）
- 其余（`DSPARK_ENABLE_ROPE_SWA_FIX` / `DSPARK_ENABLE_DSML_RECOVERY` /
  `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN` / `DSPARK_ENABLE_MXFP4_INDEXER_CACHE` +
  `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS` / `DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED`）
  由环境注入即可

## 速度

上游 Anemll 1M/6 档实测（results/RESULTS-2026-08-14.md）：

- 单聊（≤128K 任意 prompt）：首 token 后 **~62–83 decode tok/s**
- **六路短聊**（数百 token）：**~160–190 tok/s 聚合**（~30–37/路）
- 六路冷 32K–128K 同时 prefill：排队（#27），decode ~8 tok/s 保底

## 硬件需求

- **2 台** DGX Spark（固定 2 节点 · TP=2），每机 1 GPU（GB10），RoCE 组网
- Vision-Exp 权重大于 0731（ViT+Aligner 常驻显存），KV 池更小：upstream 同配置实测
  **2,331,430-token 池**（17.04 GiB KV，util 0.83）；`GPU_MEMORY_UTILIZATION` 默认 0.835
- 并发经 N=6/k=6；`MTP_NUM_TOKENS` 勿低于 6（Vision-Exp `num_nextn_predict_layers=3`）

## 参考上游

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  （vision-exp · 上游快照 `bc2ef473a1…`）
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)
- 上游实测数据：[results/RESULTS-2026-08-14.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/results/RESULTS-2026-08-14.md)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
