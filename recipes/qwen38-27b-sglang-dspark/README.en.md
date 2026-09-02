# Qwen3.8-27B · SGLang DSPARK · Fireworks recipe (single DGX Spark)

Serves **RadixArk/Qwen3.8-27B-NVFP4** via SGLang on **1** DGX Spark from Fireworks
(including DSPARK speculative decoding).

## Model

- Base model: `RadixArk/Qwen3.8-27B-NVFP4` (NVFP4 4-bit weights) + `--kv-cache-dtype
  fp8_e4m3`
- Draft model: `RadixArk/Qwen3.8-27B-DSpark` (DSpark mamba draft model, draft attention
  flashinfer; **must be distributed to the node cache** or speculative decoding won't start)
- Engine/image: SGLang, `lmsysorg/sglang:qwen38-27b` (official image, model-specific tag);
  flashinfer, `--chunked-prefill-size 2048`, `--mem-fraction-static 0.85`
- API port defaults to `30000`; reasoning/tool parsers `qwen3` / `qwen3_coder`
- ⚠️ `MAMBA_FULL_MEMORY_RATIO` defaults to `11.01` (must be in `[0,1]`; likely a typo for
  `0.1101`) — fix it against SGLang docs or real tests before publishing

## Speed

No local measurements included (this is the first SGLang recipe in the repo; backfill after
hardware validation).

## Hardware requirements

- **1** DGX Spark node (fixed single node · one GB10 GPU)
- The model is distributed by Fireworks and loaded **offline** from the node HF cache
  (`HF_HOME=/root/.cache/huggingface`, `HF_HUB_OFFLINE=1`); single-node only — do not assign
  a multi-node topology

## Upstream references

- `lmsysorg/sglang:qwen38-27b`: official SGLang image (from the source docker run)
- [sgl-project/sglang](https://github.com/sgl-project/sglang): engine
- [RadixArk](https://huggingface.co/RadixArk): model and DSpark draft model

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
