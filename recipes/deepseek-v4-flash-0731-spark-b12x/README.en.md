# DeepSeek-V4-Flash-0731 · Spark-vLLM b12x · Fireworks recipe (2× DGX Spark)

Serves **DeepSeek-V4-Flash-0731** at **TP=2** on **2** DGX Spark nodes (head + 1 worker
over RoCE) from Fireworks:

- Image: `eugr/spark-vllm-b12x:latest` (spark-vllm b12x prebuilt vLLM distribution image)
- Topology: **fixed 2 nodes · TP=2**; **B12X MLA SPARSE** attention + **b12x** MoE/linear · dspark
  speculation k=5 · **FP8 KV** cache · 1M context
- Loading: **instanttensor** with AOT compile (`VLLM_USE_AOT_COMPILE=1`), fast first boot
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (distributed by Fireworks, loaded offline)
- API port defaults to `8000`; default thinking mode `high`

> Originates from a 2-node `docker run` command (head + `--headless` worker). This recipe
> ports that command into the Fireworks compose model (host shell vars become auto-filled
> variables) and fixes several config points against the existing dspark recipes — see
> [Differences from the source command](#differences-from-the-source-command).

## How it differs from the existing recipes

| Axis | `deepseek-v4-flash-dspark` | `deepseek-v4-flash-0731-tp4-4x` | **This recipe (spark-vllm-b12x)** |
|---|---|---|---|
| Topology | 2 nodes · TP=2 | 4 nodes · TP=4 | **2 nodes · TP=2** |
| Image | Anemll `dspark-vllm-gx10` | Anemll `dspark-vllm-gx10` | **`eugr/spark-vllm-b12x`** |
| KV cache | NVFP4 DS-MLA | NVFP4 DS-MLA | **FP8** (this image's b12x stack) |
| MoE/linear backend | flashinfer_b12x | flashinfer_b12x | **b12x** |
| Attention backend | (default) | (default) | **B12X_MLA_SPARSE** |
| Speculation | dspark k=5 | dspark k=5 | dspark k=5 |
| Context | 1M | 1M | 1M |
| Load format | HF offline | HF offline | **instanttensor (AOT)** |
| Compilation mode | — | FULL_DECODE_ONLY | **FULL_AND_PIECEWISE + custom_ops=all** |
| Max sequences | 6 | 4 | **8** |
| Default port | 8888 | 8888 | **8000** (source command) |

## Quick start

Before publishing from Fireworks:

- Cluster: 2 nodes (head + 1 worker), RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` distributed to the nodes.
- Image: `eugr/spark-vllm-b12x:latest` pulled.

> Node count is fixed at 2; TP/distributed params are tuned for it.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `SPARK_VLLM_IMAGE` | `eugr/spark-vllm-b12x:latest` | spark-vllm b12x image |
| `SPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `VLLM_PORT` | `8000` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context (source left it unset; aligned with existing recipes) |
| `MAX_NUM_SEQS` | `8` | Max concurrent sequences (source) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | `64` | CUDA graph capture limit (source pinned 64; existing recipes compute `MAX_NUM_SEQS×(k+1)`=48) |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Source default; existing recipes validated 0.80 stable, 0.90 fails to boot |
| `LOAD_FORMAT` | `instanttensor` | Model load format; switch to empty (auto)/safetensors for standard HF distribution |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens (keep ≥ checkpoint dspark_block_size=5) |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}` | CUDA graph compilation mode (source) |
| `DEFAULT_THINKING` | `high` | Thinking mode off/low/high/max, overridable per request |

`NODES_TOTAL` (fixed 2), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks; the source's host shell vars `$IFACE_NAME / $IB_IF / $HEAD_IP`
map to `NCCL_SOCKET_IFNAME(netdev) / NCCL_IB_HCA(hca) / MASTER_ADDR(head_roce_ip)`, and
`GLOO/TP/MN/OMPI/UCX` all follow `NCCL_SOCKET_IFNAME`.

## Differences from the source command

Changes vs the original `docker run` when ported into Fireworks.

**Fixed (judged wrong against the existing dspark recipes)**

1. **Empty `--reasoning-config` delimiters**: the source used
   `"reasoning_start_str":"","reasoning_end_str":""`. The deepseek_v4 reasoning parser needs
   delimiters to strip `reasoning_content` and tool calls; the verified recipes use
   `" thinking"/" response"`. Empty strings mean no thinking boundary and mis-parse, so this
   recipe uses the verified `" thinking"/" response"`.
2. **Missing `--max-model-len`**: the source left it unset, so context length was derived at
   runtime; existing dspark recipes pin `1048576` (1M). Added with a 1M default
   (`MAX_MODEL_LEN`), matching the store card's `context_length`.
3. **Host shell variables**: `$HEAD_IP/$IFACE_NAME/$IB_IF` do not exist in the Fireworks
   publish environment; they're injected from auto-filled variables instead (see above).

**Kept (image-specific, not aligned to existing recipes)**

- `--kv-cache-dtype fp8`: existing recipes use `nvfp4_ds_mla`. This image's B12X MLA SPARSE +
  FP8 GEMM stack defaults to `fp8`; if the image also supports `nvfp4_ds_mla`, switching gets
  the GB10-optimized DS-MLA cache.
- `--load-format instanttensor`: a dedicated weight format. If Fireworks distributes standard
  HF safetensors here, loading fails — set `LOAD_FORMAT` empty (auto) or `safetensors`.
- `--moe-backend b12x` / `--linear-backend b12x` / `--attention-backend B12X_MLA_SPARSE`:
  this image's backend naming (existing recipes use `flashinfer_b12x`).
- `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`:
  source default; the tp4 recipe verified `FULL_DECODE_ONLY` is cost-free, switchable.
- `--gpu-memory-utilization 0.85`, `--max-num-seqs 8`, `--max-cudagraph-capture-size 64`,
  `--max-num-batched-tokens 8192`, port `8000`: source values, all kept as tunables.
- The extra `"attention_backend":"B12X_MLA_SPARSE"` key in `--speculative-config`: required
  by this image, kept.

**Carried over from the verified dspark recipes (integration layer, not in the source)**

- `--enable-chunked-prefill` / `--async-scheduling` / `--enable-prompt-tokens-details` /
  `--pipeline-parallel-size 1`: engine stability options paired with the token budget.
- Full multi-node RoCE `NCCL_*` set (auto GID, ROCEv2, cross-NIC) and the `mp` distributed
  backend.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (needed for the 1M context), etc.

## Known issues & deployment notes

- **JSON defaults must not go into `$${VAR:-...}`**: bash cuts the expansion at the first `}`
  inside the default, so `--compilation-config` fails with `trailing characters ...`. The
  recipe uses the tp4-style `[ -n "$${COMPILATION_CONFIG:-}" ] || COMPILATION_CONFIG='{"..."}'`
  fallback; **don't revert it** to `${VAR:-{...}}`.
- **`LOAD_FORMAT=instanttensor` depends on the distributed format**: confirm the node-cache
  model dir is an instanttensor layout before publishing; otherwise change `LOAD_FORMAT`
  (default HF load).
- **`GPU_MEMORY_UTILIZATION`** sits between a loss and an OOM: 0.85 is above the 0.80
  validated in existing recipes. If memory is tight at 1M context + 8 sequences, lower
  `MAX_NUM_SEQS` or `MAX_MODEL_LEN` first — don't push to 0.90.
- **CUDA graph capture limit 64** is above the 48 computed by existing recipes
  (`seqs×(k+1)`); a larger capture batch is more robust but uses more memory.
- This recipe is **not yet hardware-validated**; after an image/engine change, re-measure
  speculation params such as `MTP_NUM_TOKENS`. Per the branch model, land it on `dev` first
  and merge to `main` after real-machine validation.
- Exactly **2 nodes (TP=2)** only; pick a matching recipe or author your own for other
  topologies.

## References

This project is **Apache-2.0** (see [`LICENSE`](../../LICENSE) at the repo root); third-party
component sources and derivations are listed in the repo-root
[`NOTICE.md`](../../NOTICE.md).

- `eugr/spark-vllm-b12x:latest`: distribution image (referenced by the source docker run)
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine
- [lukealonso/b12x](https://github.com/lukealonso/b12x): MXFP4 MoE kernels
