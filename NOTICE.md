# NOTICE

This project is licensed under the **Apache License, Version 2.0**. See
[`LICENSE`](./LICENSE) for the full license text.

Copyright 2026 FireworksRecipes contributors.

## Third-party sources & attribution

This repository builds a per-model serving image for DGX Spark (GB10) and
adapts optimizations from the following open-source projects. Attributions
are kept in-file (SPDX headers) and listed here for compliance:

| Component | Upstream | License | Relation |
|---|---|---|---|
| vLLM source tree + build | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) @ `v0.26.0` | Apache-2.0 | Base engine, built from source; `overlay/vllm/` carries targeted modifications on top |
| GB10 performance recipe / attention overlay | [`Anemll/dspark-vllm-gx10`](https://github.com/Anemll/dspark-vllm-gx10) | MIT (repo) / Apache-2.0 (overlay files it carries) | Ported and adapted to vLLM v0.26.0 in `overlay/vllm/` |
| `model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py` | copied verbatim from Anemll's overlay (`Anemll/dspark-vllm-gx10`) | Apache-2.0 (SPDX header retained) | Verbatim copy, not authored here — see file header |
| MXFP4 MoE kernels (SparkInfer) | [`lukealonso/b12x`](https://github.com/lukealonso/b12x) @ `7dc6fb8` | Apache-2.0 (declared via package metadata) | Installed as an independent package at build time |
| 1M / NVFP4 KV dual-node recipe reference | [`jvr0x/dgx-spark-bench`](https://github.com/jvr0x/dgx-spark-bench) (`recipes/deepseek-v4-flash-0731-dual`) | - | Parameter/configuration reference only |
| Dual-node NVFP4 KV reference | [`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | - | Parameter/configuration reference only |

### Notes

- Files under `overlay/vllm/` carry their original Apache-2.0 SPDX headers
  (`SPDX-License-Identifier: Apache-2.0`) and copyright notices in-file.
- `b12x_mxfp4_moe.py` is a verbatim copy from Anemll's overlay; it is **not**
  an original FireworksRecipes file. Do not remove or alter its header.
- No third-party prebuilt images/binaries are pulled; all components are
  compiled from source at build time (`docker/vllm-b12x.Dockerfile`).
