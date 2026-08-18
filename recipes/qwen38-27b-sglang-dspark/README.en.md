# Qwen3.8-27B · SGLang DSPARK · Fireworks recipe (single DGX Spark)

Serves **RadixArk/Qwen3.8-27B-NVFP4** via SGLang on **1** DGX Spark from Fireworks:

- Image: `lmsysorg/sglang:qwen38-27b` (official SGLang image, model-specific tag)
- Topology: **fixed single node** (GB10, 1 GPU)
- Model: `RadixArk/Qwen3.8-27B-NVFP4` (NVFP4 4-bit weights) + `--kv-cache-dtype fp8_e4m3`
- Speculation: **DSPARK** (`RadixArk/Qwen3.8-27B-DSpark` mamba draft model, draft attention flashinfer)
- Attention/prefill: flashinfer, `--chunked-prefill-size 2048`, `--mem-fraction-static 0.85`
- API port defaults to `30000`; reasoning/tool parsers `qwen3` / `qwen3_coder`
- The model is distributed by Fireworks and loaded **offline** (`HF_HOME=/root/.cache/huggingface`)

> Originates from a single-node `docker run` (port mapping + HF cache mount). Ported into the
> Fireworks compose model: `-p 30000:30000` → host network + `--port 30000`, the
> `~/.cache/huggingface` mount is kept, and the `HF_TOKEN` placeholder became an optional variable.

## Main tunables

| Variable | Default | Notes |
|---|---|---|
| `SGLANG_IMAGE` | `lmsysorg/sglang:qwen38-27b` | SGLang image |
| `SGLANG_MODEL` | `RadixArk/Qwen3.8-27B-NVFP4` | Base model (distributed to cache) |
| `SGLANG_DRAFT_MODEL` | `RadixArk/Qwen3.8-27B-DSpark` | DSpark mamba draft model (must be distributed) |
| `SGLANG_PORT` | `30000` | SGLang API port |
| `MEM_FRACTION_STATIC` | `0.85` | `--mem-fraction-static` |
| `CHUNKED_PREFILL_SIZE` | `2048` | `--chunked-prefill-size` |
| `MAMBA_FULL_MEMORY_RATIO` | `11.01` | `--mamba-full-memory-ratio` (**suspected typo, see below**) |
| `HF_TOKEN` | empty | Auth token for gated models (usually empty when pre-distributed) |

`NODES_TOTAL=1`; a single-node recipe needs no distributed variables (no
`NODE_RANK/HEADLESS/NCCL_*`).

## Differences from the source command

- **`--mamba-full-memory-ratio 11.01` (check first)**: this parameter is the fraction of memory
  reserved for mamba's full-state compute and must be in `[0,1]`. The source value `11.01` is
  clearly invalid and is probably a typo for **`0.1101` (=11.01%)** or `0.11`. It's exposed as
  `MAMBA_FULL_MEMORY_RATIO` (default still 11.01), but **fix it against SGLang docs or real
  tests before publishing** — a value >1 can fail at startup or over-reserve memory.
- `-p 30000:30000` → host network + `--host 0.0.0.0 --port 30000` (recipes use host networking).
- `~/.cache/huggingface:/root/.cache/huggingface` mount kept, with `HF_HOME` pointed at
  `/root/.cache/huggingface` (SGLang's default) and `HF_HUB_OFFLINE=1` (offline load after
  platform distribution).
- `HF_TOKEN=<your-hf-token>` placeholder → optional `HF_TOKEN` variable (default empty); fill it
  in if the image/model checks gating at startup.
- `--shm-size 32g` → compose `shm_size: "32gb"`; `--ipc=host` and `--gpus all` mapped as-is.
- All other CLI args (`fp8_e4m3`, backends, DSPARK, mamba options) kept verbatim.

> This is the **first SGLang recipe** in the repo, so there is no same-engine verified recipe to
> cross-check; only format-level review has been done above. Validate on real hardware before
> merging to `main` per the branch model.

## Known issues & deployment notes

- **`MAMBA_FULL_MEMORY_RATIO > 1`**: the most likely config error, see above.
- **Context length**: the source does not set it, so it comes from the model config; the store
  card's `context_length` is left unspecified accordingly.
- **API model id**: SGLang exposes the model path as the `model` id
  (`RadixArk/Qwen3.8-27B-NVFP4`); clients must pass it (or the server should set
  `--served-model-name`).
- **Draft model must be distributed**: `RadixArk/Qwen3.8-27B-DSpark` also needs to be in the
  node cache or speculative decoding won't start.
- Single-node only; don't assign this recipe a multi-node topology.

## References

This project is **Apache-2.0** (see [`LICENSE`](../../LICENSE) at the repo root); third-party
component sources and derivations are listed in the repo-root
[`NOTICE.md`](../../NOTICE.md).

- `lmsysorg/sglang:qwen38-27b`: official SGLang image (referenced by the source docker run)
- [sgl-project/sglang](https://github.com/sgl-project/sglang): engine
- [RadixArk](https://huggingface.co/RadixArk): model and DSpark draft model
