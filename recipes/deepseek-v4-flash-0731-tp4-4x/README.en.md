# DeepSeek-V4-Flash-0731 · TP=4 · Fireworks recipe (4× DGX Spark)

Serves **DeepSeek-V4-Flash-0731** at **TP=4** on **4** DGX Spark nodes (head + 3
workers over RoCE) from Fireworks:

- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (Anemll prebuilt vLLM distribution image)
- Topology: **fixed 4 nodes · TP=4**; FlashInfer b12x + dspark speculation k=5 · NVFP4
  DS-MLA · **1M context**
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (~167 GB, distributed by Fireworks, loaded
  offline)
- Defaults are pinned to what passed **real-hardware validation on an agentic workload**
  (k=5, `GPU_MEMORY_UTILIZATION=0.80`, `DEFAULT_THINKING=max`, 1M context).

## How it differs from the 2-node recipe

| Axis | `deepseek-v4-flash-dspark` | **This recipe (TP=4)** |
|---|---|---|
| Topology | 2 nodes · TP=2 | **4 nodes · TP=4** |
| Image | Anemll distribution | Anemll distribution |
| Speculation | k=5 | **k=5** (verified default) |
| Context | 1M | **1M** |

## Quick start

Before publishing from Fireworks:

- Cluster: 4 nodes (head + 3 workers), RoCE configured and tested.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` distributed to the nodes.
- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` pulled.

> Node count is fixed at 4; TP/distributed params are tuned for it. The recipe sets
> `--ulimit nofile=1048576` (TP=4 opens 4× the NCCL sockets and will fail without it).

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `DSPARK_VLLM_IMAGE` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | Anemll image |
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Downloaded model |
| `VLLM_PORT` | `8888` | vLLM API port |
| `MAX_MODEL_LEN` | `1048576` | 1M context (verified default) |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | At vLLM's 8192 warmup-budget threshold (spec decode subtracts `(k−1)×seqs`; raise if concurrency is high) |
| `GPU_MEMORY_UTILIZATION` | `0.80` | Verified default; too high (0.90) fails to boot |
| `MTP_NUM_TOKENS` | `5` | DSpark speculative tokens (keep ≥ checkpoint dspark_block_size=5) |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | Capture decode graphs only (verified cost-free on hardware); `{}` restores dual capture |
| `DEFAULT_THINKING` | `max` | Thinking mode off/low/high/max, overridable per request |

`NODES_TOTAL` (fixed 4), `MASTER_ADDR`, `NODE_RANK`, `HEADLESS`, `VLLM_HOST_IP`, `NCCL_*`
are auto-filled by Fireworks. `max_cudagraph_capture_size` is computed as
`MAX_NUM_SEQS × (k+1)` = 4×6 = 24 (tracks `MAX_NUM_SEQS` / `MTP_NUM_TOKENS` changes).

## Known issues & deployment notes

- **JSON defaults must not go into `$${VAR:-...}`**: bash cuts the expansion at the first `}`
  inside the default, so when the var is already injected by compose a stray `}` is left at
  the end of the JSON (`--compilation-config` fails with
  `trailing characters ... "FULL_DECODE_ONLY"}}`). The recipe uses the nested-brace-free
  `[ -n "$${VAR:-}" ] || VAR='{"..."}'` fallback instead; **don't revert it** to `${VAR:-{...}}`.
- **`max_num_batched_tokens` is not a prefill budget**: with spec decode vLLM subtracts
  `(k−1)×max_num_seqs` to get `max_num_scheduled_tokens` and warns below 8192; this recipe
  defaults to exactly 8192 (on the threshold). Raise it if concurrency is high.
- **`GPU_MEMORY_UTILIZATION`** is pinned between a loss and an OOM: too high (0.90) won't boot;
  0.80 is the validated default on hardware. Raising it reclaims KV pool — just don't
  push to 0.90.
- **`--override-generation-config` has been removed**: after real-hardware validation the
  server-side temperature override is no longer passed to vLLM (the `OVERRIDE_GENERATION_CONFIG`
  env var stays in the container but the command no longer references it); set temperature on the
  request side, or restore the flag, if you need a server-side freeze. If you change the
  image/engine, re-measure speculation params such as `MTP_NUM_TOKENS` — defaults don't transfer
  across engines.
- Exactly **4 nodes (TP=4)** only; pick a matching recipe or author your own for other
  topologies.

## References

This project is **Apache-2.0** (see [`LICENSE`](../../LICENSE) at the repo root); third-party
component sources and derivations are listed in the repo-root
[`NOTICE.md`](../../NOTICE.md).

- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10): distribution image
- [vllm-project/vllm](https://github.com/vllm-project/vllm): engine
- [lukealonso/b12x](https://github.com/lukealonso/b12x): MXFP4 MoE kernels
