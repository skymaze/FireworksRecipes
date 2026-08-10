# DeepSeek-V4-Flash-0731 · Fireworks recipe

Serves **DeepSeek-V4-Flash-0731** on **exactly 2** DGX Spark nodes (head + 1 worker over RoCE):

- Image: `fireworks-models/deepseek-v4-flash-0731:0.3.1` (mainline vLLM v0.26.0 + a
  GB10-targeted overlay; patches and tuned ENV baked at build time)
- Topology: **fixed 2 nodes · TP=2**
- Loading: InstantTensor + dspark speculative decoding (MTP=5)
- KV cache: `nvfp4_ds_mla`, **1M context**
- MoE: `auto` (DeepGEMM), 2200+ tok/s prefill measured on real hardware

Weights are not baked into the image (~167 GB); Fireworks distributes them to each node and
the image loads offline.

## Quick start

Before publishing from Fireworks:

- Cluster: **exactly 2** nodes (head + 1 worker), RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` distributed to the nodes.
- Image: `fireworks-models/deepseek-v4-flash-0731:0.3.1` pulled.

> The node count is locked at **exactly 2** (model parameters are tuned for it);
> model/image pickers select ready items and prompt you to distribute anything missing.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `VLLM_IMAGE` | `fireworks-models/deepseek-v4-flash-0731:0.3.1` | Pulled image |
| `TENSOR_PARALLEL_SIZE` | `2` | Fixed 2 (1 GPU per GB10 node) |
| `MAX_MODEL_LEN` | `1048576` | 1M context; needs `nvfp4_ds_mla` KV |
| `GPU_MEMORY_UTILIZATION` | `0.88` | 1M context needs ≥0.88 |
| `KV_CACHE_DTYPE` | `nvfp4_ds_mla` | Recommended; `fp8_ds_mla` also works |
| `NUM_SPECULATIVE_TOKENS` | `5` | DSpark speculative tokens |
| `MASTER_PORT` | `25000` | Distributed master port |
| `DEFAULT_THINKING` | `off` | Thinking mode off/low/high/max |

`MASTER_ADDR`, `NODES_TOTAL`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*` are
auto-filled by Fireworks.

## Known issues

- `GPU_MEMORY_UTILIZATION` below 0.88 only works on the b12x stack; DeepGEMM at 1M context
  needs ≥0.88.
- Tuned for a fixed 2-node · TP=2 topology; pick another recipe (or author your own) for
  other topologies.
- The model is loaded on the first publish; time depends on storage/network.

## References

References (full attribution in the repo-root `NOTICE.md`):

- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
- [lukealonso/b12x](https://github.com/lukealonso/b12x)
- [jvr0x/dgx-spark-bench](https://github.com/jvr0x/dgx-spark-bench)
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
