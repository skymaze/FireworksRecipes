# Qwen3.8-Flash-Next · single-node vLLM · Fireworks recipe (1× DGX Spark)

Serve **Qwen3.8-Flash-Next** (176.9B params) through vLLM on **one** DGX Spark (GB10,
128 GiB unified memory) at the **full native 262,144-token context**.

## Model

- Base model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (NVFP4 checkpoint, ~122 GiB, keeps the
  full vision tower and all 31 MTP tensors; multimodal, 0.967 on the atlas image eval)
- The key trick: 51.2B of the 176.9B parameters are a lookup table that is only read, never
  multiplied — this recipe **streams it from NVMe** (`VLLM_PLE_MMAP=1`), which is how the
  122 GiB checkpoint fits next to a usable KV pool at full context
- Speculation: the model's trained MTP draft head (k=3)
- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0`
- API on port `8000` by default; served model name `qwen3.8-flash-next`; OpenAI-compatible
- Source: the **longctx** (vLLM) lane of `0xBakeer/qwen38-flash-next-spark`; for coding
  rewrites use the companion recipe
  [`qwen38-flash-next-edit`](../qwen38-flash-next-edit/README.en.md) (they share the GPU —
  only one runs at a time)

## Speed

Measured (upstream):

- Single stream: **~32 tok/s** free-form (MTP pays +35%, 17→27 tok/s, only while the batch
  is unsaturated)
- **16 is the concurrency wall**: under 16, TTFT < 2.7 s and aggregate is 96–109 tok/s; at 32
  TTFT 16 s, at 64 70 s (batch work can use 64 for another ~+35% aggregate)
- Prefix caching on by default: +76% aggregate and usually half the first-token latency on
  shared prefixes
- Thinking tokens dominate (86%): request-level thinking-off cuts the same answer from ~55 s
  to ~15 s

## Hardware requirements

- **1** DGX Spark node (fixed single node · GB10, 128 GiB unified memory)
- At gmu 0.85 the KV pool is 641,601 tokens (**do not drop back to the upstream default
  0.78** — less than one full-length request)
- First boot takes 12–15 min reading ~83 GiB of weights; **disk**: checkpoint ~126 GB +
  image ~21 GB — confirm ≥ ~150 GB free before publishing

## Upstream references

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  (MIT): serving configuration, tuning and measurements (longctx lane)
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (Apache-2.0):
  the vLLM container this image is based on (8 patches incl. PLE-mmap and the
  deterministic persistent_topk [@jschmied / vllm#55122] — deterministic outputs, and it
  removes the exact-topk prefill penalty: 2,436 tok/s @8k / 2,904 @32k)
- Deterministic top-k on by default: `VLLM_QSA_DET_TOPK=1` (`VLLM_QSA_EXACT_TOPK=1` still
  wins; set to 0 to revert to the old path)
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) ·
  [Qwen](https://qwen.ai) (model and n-gram/PLE tech report)

Full attribution and derivations in the repo-root [`NOTICE.md`](../../NOTICE.md).
