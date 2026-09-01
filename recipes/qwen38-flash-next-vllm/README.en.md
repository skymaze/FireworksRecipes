# Qwen3.8-Flash-Next · single-node vLLM · Fireworks recipe (1× DGX Spark)

Serve **Qwen3.8-Flash-Next** (176.9B params) through vLLM on **one** DGX Spark (GB10,
128 GiB unified memory) at the **full native 262,144-token context**:

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0`
  (**dev: must first be baked and pushed to ACR before the recipe is publishable** — see *Image*)
- Topology: **fixed single node** (one GB10 GPU)
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (NVFP4 checkpoint, ~122 GiB, keeps the **full
  vision tower** and **all 31 MTP tensors**)
- The key trick: 51.2B of the 176.9B parameters are a **lookup table that is only read, never
  multiplied**. This recipe **streams it from NVMe** (`VLLM_PLE_MMAP=1`) — which is how a
  122 GiB checkpoint fits next to a usable KV pool on one box at full context
- Speculative decoding: the model's **trained MTP draft head** (k=3), steady on any text
  (~32 tok/s free-form)
- Multimodal: the NVFP4 checkpoint carries the full vision tower (333 tensors, unquantized);
  0.967 on the atlas image eval
- API on port `8000` by default; served model name `qwen3.8-flash-next`; OpenAI-compatible.

> Ported from the **longctx** (vLLM) lane of
> [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) —
> that repo's own recommended default ("Not sure → Writing & long documents"). The other
> **edit** (llama.cpp) lane compiles on the host and is outside Fireworks' container model;
> this recipe does not cover it.

## Image (must be done before publish)

Upstream `blazux/qwen3.8-Flash-DGX` (Apache-2.0) does **not publish a prebuilt image**: its
`setup.sh` runs a local `docker build` (base `vllm/vllm-openai:qwen38-flash-next@sha256:fc120e…`
plus 7 patches, of which the PLE-mmap patch is what makes this recipe possible). Following this
repo's convention, bake that image and push it to
`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.0.0` so Fireworks can pull
and distribute it.

> The dev image is not baked yet — **build & push it before publishing**, and record the baked
> upstream commit (the `de.qwen38fn.upstream-sha` label) in this README for reproducibility.

## Main variables

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_IMAGE` | `.../aixn-public/qwen38-flash-next:v1.0.0` | vLLM image (to be baked) |
| `QWN38_MODEL` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Main model (pre-distributed to the node HF cache) |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | Served model name |
| `VLLM_PORT` | `8000` | API port |
| `MAX_MODEL_LEN` | `262144` | Context length (the model's native maximum) |
| `MAX_NUM_SEQS` | `16` | Concurrency cap (16 is the last tier served well) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 0.85 measures a 641,601-token KV pool (do not fall back to upstream 0.78) |
| `MTP` | `3` | MTP speculative tokens (0 = off; >3 is worse) |
| `PREFIX_CACHE` | `1` | Prefix caching (+76% on shared prefixes; 0 = reproduce the published prefill figures) |
| `PLE_MMAP_WORKERS` | `32` | Concurrent NVMe streaming workers |
| `PLE_MMAP_PREWARM` | `1` | Stream the table once at boot so the first request is not cold |
| `HF_TOKEN` | (empty) | Auth token for gated models (usually empty when pre-distributed) |

`NODES_TOTAL=1`; single-node — no distributed variables (`NODE_RANK/HEADLESS/NCCL_*`).

## Differences from the upstream serve.sh (Fireworks integration layer)

- **Snapshot path → repo id**: upstream passes the exact `…/snapshots/<sha>/` path; this recipe
  uses the repo id (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) with `HF_HOME=/hf` +
  `HF_HUB_OFFLINE=1` for offline resolution (the platform distributes the model to the node cache).
- `-p 127.0.0.1:8000:8000` → host network + `--host 0.0.0.0 --port 8000` (Fireworks' uniform
  host networking; restrict access at the cluster/firewall layer instead of the container bind).
- `~/.cache/huggingface:/hf` mount kept; `HF_HOME=/hf`, `HF_HUB_OFFLINE=1`,
  `HF_HUB_DISABLE_XET=1` (upstream disables Xet in setup — it stalls on some Spark setups).
- `--shm-size 16g` → `shm_size: "16gb"`; `--ipc=host`, `--gpus all` mapped as-is.
- The rest of the command line (PIECEWISE CUDA-graph capture + splitting-op list,
  `--no-enable-flashinfer-autotune`, MTP speculation, tool/reasoning parsers,
  `--max-num-batched-tokens 8192`) is kept verbatim; `MTP=0` and `PREFIX_CACHE=0` become
  variable switches with identical behaviour.

## Behaviour & tuning (from upstream measurements)

- **MTP speculation pays only while the batch is unsaturated**: +35% single-stream
  (17→27 tok/s class), unmeasurable at 16 concurrent (once the batch saturates the box,
  accepted drafts no longer convert to throughput). k>3 is worse.
- **16 is the concurrency wall**: under 16, TTFT stays < 2.7 s and aggregate is 96–109 tok/s;
  at 32 TTFT jumps to 16 s, at 64 to 70 s. For batch work (nobody waiting on a first token)
  `MAX_NUM_SEQS=64` buys another ~+35% aggregate.
- **Prefix caching is on by default**: multi-turn chat / agent loops resending a system prompt /
  shared prefixes get +76% aggregate and usually more than half the first-token latency;
  unrelated prompts gain nothing. Every prefill figure upstream published was measured
  cache-free — `PREFIX_CACHE=0` reproduces them.
- **Thinking tokens dominate**: upstream measured 86% of generated tokens as reasoning.
  Per-request `{"chat_template_kwargs":{"enable_thinking":false}}` cuts the same answer from
  ~55 s to ~15 s.
- **KV pool**: at gmu 0.85, 18.13 GiB = 641,601 tokens (~2.4× a full context); the upstream
  default 0.78 leaves only 227,651 — less than one full-length request. **Do not drop back to 0.78.**
- **First boot is slow**: ~12–15 min reading ~83 GiB of weights (`VLLM_ENGINE_READY_TIMEOUT_S=3600`,
  healthcheck `start_period: 900s`); later starts are faster.
- **Disk**: checkpoint ~126 GB + image ~21 GB — confirm ≥ ~150 GB free on the node before publishing.

## Known issues & deployment notes (from upstream)

- **CUDA-graph capture must stay PIECEWISE**: the n-gram gather is a CPU op + host→device copy,
  declared a splitting op; switching to FULL capture breaks it.
- **`VLLM_PLE_CPU_OFFLOAD=1` hangs**: it expects an offload worker this image does not launch,
  and spins a core indefinitely. **Only `VLLM_PLE_MMAP` works** (fixed in this recipe).
- **Prefix caching with block size**: the fix landed in the blazux container from `8347e7c`
  (2026-08-29) on; bake the image from a commit at or after that, and don't enable prefix
  caching on an older image.
- **Profiling**: `VLLM_TORCH_PROFILER_DIR` is inert in this build (moved to
  `--profiler-config`); don't rely on it.
- Single-node recipe: 1 node only — do not assign a multi-node topology.
- The large checkpoint goes through the platform model-distribution channel (HF cache mounted
  at `/hf`, offline load).

## Verify

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## Sources

**Apache-2.0** (repo root [`LICENSE`](../LICENSE)); third-party sources and derivations in
[`NOTICE.md`](../NOTICE.md).

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  (MIT): the serving configuration, tuning and measurements this recipe is based on (longctx lane)
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (Apache-2.0):
  the vLLM container with the 7 patches including PLE-mmap (image bake source)
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4):
  NVFP4 checkpoint (weights carry Qwen's own licence)
- [Qwen](https://qwen.ai): the Qwen3.8-Flash-Next model and n-gram/PLE tech report
