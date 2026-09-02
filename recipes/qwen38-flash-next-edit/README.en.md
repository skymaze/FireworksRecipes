# Qwen3.8-Flash-Next · Edit · single-node llama.cpp · Fireworks recipe (1× DGX Spark)

Serve **Qwen3.8-Flash-Next** on **one** DGX Spark (GB10, 128 GiB unified memory) with the
**llama.cpp coding-edit lane** (the containerized form of the upstream `edit` setup) — best for
coding agents that rewrite files you hand them, at the **full 262,144-token context**:

- Image: `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0`
  (baked from the upstream `recipes/llamacpp-edit/Dockerfile`: CUDA 13 devel stage builds
  llama.cpp @qwen4exp PR `035e227` + a slim runtime with only `llama-server` / `llama-cli`)
- Topology: **fixed single node** (one GB10 GPU)
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` **UD-Q4_K_XL** shards (~104 GiB) +
  `mmproj-F16.gguf` (~0.9 GiB vision projector, optional)
- Same key trick as the vLLM recipe: 51.2B of the 176.9B parameters is a **lookup table that is
  only read**; this lane keeps it **on the NVMe page cache** with `-lm mmap` +
  `-ot per_layer_token_embd=CPU` instead of on the GPU
- Speculation: **ngram-mod** context-copying — lifts 60-token spans that already exist in your
  prompt and verifies them in one pass; **exact**, byte-identical output (88 tok/s on file
  rewrites, ~28 tok/s free-form)
- Versus the vLLM recipe: the GGUF converter drops the model's trained MTP draft head (no MTP
  and no cross-engine speculation here), but no NVFP4 checkpoint to download, and a lower-bit
  quant is available if you want to trade disk for quality

> This is the **edit** (llama.cpp) lane of
> [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark).
> For chat, reasoning and long documents use the companion
> [`qwen38-flash-next-vllm`](../qwen38-flash-next-vllm/README.en.md) recipe (steadier, ~5x
> faster prefill). They share the GPU — only one runs at a time.

## Image (baked & pushed to ACR)

Baked and pushed:
`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0`
(llama.cpp @qwen4exp PR `035e227`; ACR digest `sha256:0147ea29…`). Two bake-time fixes over the
upstream `recipes/llamacpp-edit/Dockerfile` (both in the image label
`de.qwen38fn.bake-note`):

1. **`-DGGML_CUDA_NO_VMM=ON`**: with no libcuda to link in the container, the default VMM path
   fails at the llama-server link (`undefined reference to cuMem*` — the CUDA driver symbols).
   VMM off uses cudaMalloc instead (functionally equivalent); the real driver is mounted from
   the DGX Spark host under `--gpus all` at runtime.
2. **Copy the whole `build/bin` + `LD_LIBRARY_PATH`**: upstream copies only
   `llama-server`/`llama-cli`, but current llama.cpp splits the server into
   `libllama-server-impl.so` etc. — the runtime container then dies with
   `error while loading shared libraries`.

Verified: `ldd` shows no missing libs, `llama-server --version` launches (the CUDA-init error
only appears because the verification container had no `--gpus all`, which is expected).

> Build/push cheat-sheet (this machine's Docker Desktop containerd store produces manifests ACR
> rejects, so pushes go through skopeo from a `docker-archive` tar):
>
> ```bash
> cd <upstream 0xBakeer/qwen38-flash-next-spark clone>
> docker build -t qwen38-flash-next-edit:v1.0.0 \
>   --label "de.qwen38fn.llamacpp-ref=pinned@035e227" \
>   --label "de.qwen38fn.bake-note=GGML_CUDA_NO_VMM=ON + full bin copy (driverless bake fixes)" \
>   -f <this repo edit-bake/Dockerfile> .
> docker save qwen38-flash-next-edit:v1.0.0 -o edit-image.tar
> skopeo copy docker-archive:edit-image.tar \
>   docker://registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next-edit:v1.0.0
> ```

## Quick start

1. Model: distribute `unsloth/Qwen3.8-Flash-Next-GGUF`'s `UD-Q4_K_XL/*` (plus
   `mmproj-F16.gguf` for vision) to the node HF cache via the platform model distribution.
2. Publish: pick this recipe + a 1-node cluster (`nodes=1` must match exactly).
3. Verify:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50}'
```

## Main variables

| Variable | Default | Meaning |
|---|---|---|
| `LLAMACPP_IMAGE` | `.../qwen38-flash-next-edit:v1.0.0` | llama.cpp image (ACR-baked) |
| `QWN38_GGUF_REPO` | `unsloth/Qwen3.8-Flash-Next-GGUF` | GGUF repo (only UD-Q4_K_XL + mmproj need distributing) |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | Served model name |
| `LLAMACPP_PORT` | `8000` | llama-server port |
| `MAX_MODEL_LEN` | `262144` | Context length (split across PARALLEL slots) |
| `PARALLEL` | `2` | Slots (2 = 131,072 per request; 1 = full window) |
| `SPEC` | `ngram-mod` | Speculation mode (`none` disables) |
| `MMPROJ_MODE` | `auto` | Pass `--mmproj` when mmproj-F16.gguf is in the cache (`none` = text-only) |
| `EXTRA_ARGS` | (empty) | Space-separated llama-server args appended at the end |
| `HF_TOKEN` | (empty) | Auth token for gated models (usually empty when pre-distributed) |

`NODES_TOTAL=1`; single-node — no distributed variables.

## Differences from the upstream Dockerfile / run.sh (integration layer)

- **Snapshot resolution**: the image CMD expects a fixed `/model/...` path; this recipe resolves
  `snapshots/*/UD-Q4_K_XL/*00001-of-*.gguf` from the node HF cache (`HF_HOME=/hf`) by repo id
  (with mmproj auto-detection), so the platform can distribute and load offline.
- **Port**: `--port ${LLAMACPP_PORT:-8000}` inside the container (the image CMD default of 8000
  is kept); host networking + `--host 0.0.0.0`.
- **`~/.cache/huggingface:/hf`**: same as the upstream container; `HF_HOME=/hf`,
  `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`.
- Speculation / parallel / vision become switches (`SPEC=ngram-mod`/`none`, `PARALLEL`,
  `MMPROJ_MODE`) with identical behaviour; `-lm mmap`, `-ot per_layer_token_embd=CPU`,
  `--n-gpu-layers 999`, `--temp 1.0 --top-p 0.95 --top-k 20` are fixed defaults (overridable
  via `EXTRA_ARGS`).

## Behaviour & tuning (from upstream measurements)

- **The 3.2x task-to-task spread comes from where speculation copies from**: rewriting a file
  you handed it (answer already in the prompt) is 88 tok/s; writing something new has nothing to
  copy (~28). Judge speed by task, not by a single number.
- **Speculation is exact**: the target verifies every drafted token, so output is byte-identical
  to disabling it — no quality-for-speed trade.
- **Don't obsess over warm/cold page cache**: a deferred warm measured no gain (27.8 tok/s at
  both 17.7% and 58.1% residency).
- **Don't drop to a lower-bit K-quant for speed**: UD-Q3_K_XL moves 19% fewer bytes yet is 14%
  slower (this config is not bandwidth-bound). Pick a smaller quant for disk or quality only.
- **f16 KV**: quantized KV aborts on this architecture; keep f16 (~24 KB/token, cheap).
- **Two requests, not two hundred**: `--parallel 2` is a concurrency cap that claims its share
  of context; 1.24–1.30x under load, free at one caller; `PARALLEL=1` restores the full window.
  At 64 slots the prompt cache thrashes and requests are lost.
- **Thinking tokens**: as with the vLLM recipe — 86% of generated tokens were reasoning;
  per-request `{"chat_template_kwargs":{"enable_thinking":false}}` cuts the same answer from
  ~55 s to ~15 s.

## Known issues & deployment notes

- Context is split across slots: with two slots, a too-long prompt returns 400 (not truncated);
  the `prefill-128k` workload (161K tokens) fails entirely at two slots and fits at one — for
  very long documents set `PARALLEL=1` or use the vLLM recipe.
- **No MTP**: the GGUF has no trained draft head; prefer the vLLM recipe for long documents and
  steady-throughput chat.
- Disk: UD-Q4_K_XL ~104 GiB + image ~4-6 GiB.
- Single-node recipe: 1 node only — do not assign a multi-node topology.

## Sources

**Apache-2.0** (repo root [`LICENSE`](../LICENSE)); third-party sources and derivations in
[`NOTICE.md`](../NOTICE.md).

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  (MIT): the edit lane's container Dockerfile, serving config and measurements
  (`recipes/llamacpp-edit/`)
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
  UD-Q4_K_XL quants and mmproj (weights carry Qwen's own licence)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): the engine
