# 🎇 FireworksRecipes (English)

Build repository that produces **per-model serving images** and **native recipes**
for **Fireworks** (a DGX Spark cluster management tool).

> Unlike the common "one vLLM image + runtime patching" approach, this repo builds a
> **dedicated image per model**, baking the model's required vLLM patches and tuned
> environment variables at **build time**; no runtime patching needed. Fireworks pulls
> the image and publishes tasks directly.
> No third-party prebuilt images are used — every component is compiled from source
> locally, maintained in this repository.

中文版（Chinese original）: [../README.md](../README.md)

## Supported models

| Model | Image tag | Notes |
|---|---|---|
| DeepSeek-V4-Flash-0731 | `fireworks-models/deepseek-v4-flash-0731:0.3.0` | Mainline vLLM v0.26.0 / Anemll-style GB10 overlay / InstantTensor + dspark speculative MTP=5 / 2-node TP=2 (KV=nvfp4_ds_mla, 1M context, MoE=auto/DeepGEMM) · **fw-warmup patch (fix NVIDIA mHC warmup no-op) + baked JIT cache (`/opt/fw/vllm-cache-seed`, zero inference-time compilation)** |

## Repository layout

```
FireworksRecipes/
├── versions.conf        # Global version lock (all source/dependency pins)
├── LICENSE / NOTICE.md  # Apache-2.0 license and third-party attribution
├── scripts/
│   ├── build.sh         # Local source build driver (base / model / all)
│   ├── bench/           # Benchmark harness: bench_decode.py / bench_prefill.py
│   └── deploy/          # 2-node deployment: deploy_v0260.sh / run_vllm_node.sh
├── docker/
│   └── vllm-b12x.Dockerfile   # Multi-stage source build: flashinfer/vllm/deepgemm/nccl → runner
├── overlay/
│   └── vllm/            # vLLM source overlay (Anemll recipe port; rsync'd into source at build)
├── docs/
│   ├── README.en.md           # This document
│   └── BENCHMARK-v0260.md     # 2-node benchmark results and comparison
└── models/
    └── deepseek-v4-flash-0731/
        ├── build.conf
        ├── Dockerfile.model
        ├── patches/           # hybrid-draft-loader / fw-warmup
        └── recipe/fireworks.recipe.json
```

## Architecture

### 1. Source build system (`docker/vllm-b12x.Dockerfile`)

Multi-stage, fully local build, **no third-party images/prebuilt artifacts**:

| Stage | Contents |
|---|---|
| `base` | CUDA 13 devel + build deps + PyTorch(cu130) + **NCCL from source** (sm_121 gencode) |
| `flashinfer` | FlashInfer from source (commit 0472b9b ≈ 0.6.15; cubins skipped by default) |
| `vllm` | Mainline vLLM v0.26.0 (incl. DeepGEMM) Rust-frontend wheel build; overlay baked in via rsync |
| `runner` | Runtime image: wheels + ray/fastsafetensors/instanttensor + b12x(SparkInfer) + NCCL ordering |

All versions are pinned in `versions.conf`.

### 2. Per-model layer (`models/<model>/Dockerfile.model`)

- **Patches**: `hybrid-draft-loader` (draft model → lazy safetensors under instanttensor);
  `fw-warmup` (fix NVIDIA mHC warmup no-op + sparse MLA coverage → no inference-time JIT/OOM).
- **Tuned ENV**: GB10-specific defaults, overridable via recipe `environment`.
- Metadata `FW_MODEL_ID` + OCI LABELS.

> **Weights are not baked into the image** (~167 GB for DeepSeek-V4-Flash-0731). Distribute
> via Fireworks model management; image loads offline (`HF_HUB_OFFLINE=1`).

### 3. Recipe (`recipe/fireworks.recipe.json`)

Follows the Fireworks `POST /api/recipes/import` schema (`image` field validated against
`build.sh` output; `compose_template` uses v0.26.0 `vllm serve` args with multi-node
distributed settings).

## Quick start

Prerequisites: linux/arm64 (aarch64) build host (DGX Spark itself), Docker ≥ 23
(BuildKit), network access to GitHub/HuggingFace, ≥ 100 GB disk.

```bash
# 1) build base runner
./scripts/build.sh base

# 2) build model image
./scripts/build.sh model --model deepseek-v4-flash-0731 --push-registry ghcr.io/<org>
#    or: --save dist/   (docker load route)
#    optional: SEED_CACHE_DIR=seed-cache bash scripts/build.sh model  # bake JIT cache

# 3) run in Fireworks: import image + recipe, publish 2-node TP=2 task (head=rank0)

# 4) verify
curl -s http://<head-ip>:8000/v1/models
curl -s http://<head-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"你好"}],"thinking":true}'

# 5) bench & deploy (placeholders are overridable via env)
python3 scripts/bench/bench_decode.py  --base-url http://<head-ip>:8000/v1 --model deepseek-v4-flash-0731 --concurrency 1,2,4 --max-tokens 128
HEAD_IP=<head-ip> WORKER_IP=<worker-ip> ./scripts/deploy/deploy_v0260.sh [IMAGE]
```

> Deployment scripts read node addresses from `HEAD_IP`/`WORKER_IP`/`MASTER_ADDR` env vars
> (no hardcoded lab IPs). A 32 GiB host swap is recommended as a safety cushion for
> compile-time memory peaks.

## Benchmark highlights (v0.26.0 · 2× DGX Spark TP=2)

See **[BENCHMARK-v0260.md](./BENCHMARK-v0260.md)** for full methodology and history.

- decode (512-token fixed): **41.2 / 94.4 / 151.2 tok/s** @1/2/4; per-session 54.8 / 34.1
- prefill: 1300–2400 tok/s @2–16K; **128K=1566 / 256K=1289 / 512K=1400 / 1M=900+ tok/s**
- stability: long contexts (200K–1M) run without crash; inference-time JIT compilation
  eliminated via **fw-warmup patch + baked JIT cache** (mHC warmup <1s on fresh nodes).

Final config: `--kv-cache-dtype nvfp4_ds_mla --max-model-len 1048576 --max-num-seqs 6
--max-cudagraph-capture-size 36 --gpu-memory-utilization 0.88` +
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` + MoE=auto (DeepGemmFP4Experts).

## License & attribution

Apache-2.0. See [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md) for third-party
attribution (vLLM, Anemll/dspark-vllm-gx10, lukealonso/b12x, jvr0x/dgx-spark-bench,
tonyd2wild). See also
[CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md).
