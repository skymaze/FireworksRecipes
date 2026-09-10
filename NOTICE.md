# NOTICE

This project is licensed under the **Apache License, Version 2.0**. See
[`LICENSE`](./LICENSE) for the full license text.

Copyright 2026 FireworksRecipes contributors.

## Third-party sources & attribution

This repository contains model **recipes** only (parameters, topology, image references);
it does not bundle or build third-party code. Recipe parameters are tuned against the
following public reference works (configuration reference only):

| Component | Upstream | Relation |
|---|---|---|
| 1M / NVFP4 KV dual-node recipe reference | [`jvr0x/dgx-spark-bench`](https://github.com/jvr0x/dgx-spark-bench) | Parameter/configuration reference only |
| Dual-node NVFP4 KV reference | [`tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | Parameter/configuration reference only |
| Dual-node Spark-vLLM b12x serve (2× GB10, B12X stack) | [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) | Parameter/configuration reference only (MIT) |
| Dual-node DSpark recipe route | [`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | Parameter/configuration reference only |
| Dual-node GLM-5.3-Flash EXL3 serve (2× GB10) | [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | Parameter/configuration reference only (AGPL-3.0-or-later) |
| Single-node Qwen3.8-Flash-Next vLLM serve (1× GB10, TP=1) | [`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) | Parameter/configuration reference only (AGPL-3.0-or-later) |

Recipes reference prebuilt images hosted on registries (e.g.
`ghcr.io/anemll/dspark-vllm-gx10`); those images carry their own licenses and are **not**
bundled in this repository.

`docker/spark-vllm-b12x/` vendors the `mods/instanttensor-hybrid-draft-loader` sources from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) at commit
`60dee5b` (MIT, Copyright 2026 Eugene Rakhmatulin) to build the derived serving image
`registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0` (base:
`eugr/spark-vllm-b12x` `nightly-20260909`).
