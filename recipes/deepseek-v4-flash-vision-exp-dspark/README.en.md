# DeepSeek-V4-Flash-Vision-Exp DSpark · Fireworks recipe (2× DGX Spark)

Serve DeepSeek-V4-Flash-**Vision-Exp** at **TP=2** on **2** DGX Spark nodes (head + 1
worker over RoCE) from Fireworks — DeepSeek's first experimental multimodal model of the
V4 family (1M context).

## Model

- Base model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (~305B total incl. ViT+Aligner,
  MIT; ~167 GB, distributed by Fireworks, loaded offline)
- Multimodality: native **image input** (OpenAI `image_url`, JPEG/PNG/GIF/WebP; GIF decoded
  as a still frame; up to 8 per request by default, **user messages only**; **no video
  encoder** in the official weights)
- Quant/speculation: NVFP4 DS-MLA · FlashInfer b12x + dspark speculation (k=6) · 1M context
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix5`
  (Anemll + the Mia hotfix chain + native Vision-Exp image support; upstream snapshot
  `d97c808ec…`)
- Served name: `deepseek-v4-flash-vision-exp`; API port defaults to `8888`

## Speed

Upstream Anemll measurements on the 1M/6 tier (results/RESULTS-2026-08-14.md):

- One chat (any prompt through 128K): **~62–83 decode tok/s** after the first token
- **Six short chats** (hundreds of tokens): **~160–190 tok/s aggregate** (~30–37 per stream)
- Six cold 32K–128K prompts at once: prefills queue (#27), ~8 tok/s decode floor

## Hardware requirements

- **2** DGX Spark nodes (fixed 2 nodes · TP=2), one GB10 GPU each, RoCE
- Vision-Exp weights are larger than 0731 (resident ViT+Aligner), so the KV pool shrinks:
  upstream measured **2,331,430-token pool** (17.04 GiB KV at util 0.83);
  `GPU_MEMORY_UTILIZATION` defaults to 0.835
- N=6/k=6; keep `MTP_NUM_TOKENS` at least 6 (Vision-Exp `num_nextn_predict_layers=3`)

## Upstream references

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  (vision-exp · snapshot `d97c808ec1c…`, 2026-09-01)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [vllm-project/vllm](https://github.com/vllm-project/vllm) · [lukealonso/b12x](https://github.com/lukealonso/b12x)
- Upstream benches: [results/RESULTS-2026-08-14.md](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/results/RESULTS-2026-08-14.md)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
