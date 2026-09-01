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
| Dual-node NVFP4 KV reference | [`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | Parameter/configuration reference only |
| Dual-node DSpark recipe route | [`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | Parameter/configuration reference only |
| Dual-node GLM-5.3-Flash EXL3 serve (2× GB10) | [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | Parameter/configuration reference only |
| Single-node Qwen3.8-Flash-Next vLLM serve (1× GB10) | [`0xBakeer/qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark) | Parameter/configuration reference only (longctx + edit lanes; MIT) |
| Single-node Qwen3.8-Flash-Next PLE-mmap vLLM container | [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX) | Image bake source (Apache-2.0; not vendored) |

Recipes reference prebuilt images hosted on registries (e.g.
`ghcr.io/anemll/dspark-vllm-gx10`); those images carry their own licenses and are **not**
bundled in this repository.
