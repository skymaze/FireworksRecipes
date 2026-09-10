# 🎇 FireworksRecipes

DGX Spark (GB10) **model recipes** for **Fireworks** (DGX Spark cluster manager): each
recipe is a `fireworks.recipe.json` you can import directly (image / fixed topology / tuned
parameters / auto-filled network vars), plus the `recipes/index.json` catalog the "Recipe
Store" reads.

> This repository ships **recipes and catalog only — no image build code.** Recipes reference
> ready-made image tags; Fireworks pulls and distributes them to nodes.

**Chinese**: [../README.md](../README.md)

Current recipes:

| Recipe | Image | Description |
|---|---|---|
| [DeepSeek-V4-Flash-Vision-Exp (TP=4)](../recipes/deepseek-v4-flash-vision-exp-tp4-4x/README.md) | `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix8` | **4-node TP=4** (multimodal) · hotfix8 entrypoint **TP-parametrized** (DSPARK_TP/NODES_TOTAL) · FlashInfer b12x + dspark **k=3** (measured best) · NVFP4 DS-MLA · **1M context** · tested: 54.9 single-stream / ~160 t/s at ws=6, ~7.57M-token KV pool per rank · **TP8 impossible** (FlashInfer DSV4 sparse-MLA lacks an 8-head template; model has 64 heads) |
| DeepSeek-V4-Flash (TP=4) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | **4-node TP=4** DSpark · FlashInfer b12x + dspark k=5 · NVFP4 DS-MLA · **1M context** · verified on a real agentic workload |
| [DeepSeek-V4-Flash-0731 (Spark b12x)](../recipes/deepseek-v4-flash-0731-spark-b12x/README.md) | `registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0` | **2-node TP=2** Spark-vLLM · B12X MLA SPARSE + b12x MoE/linear · dspark k=5 · **FP8 KV** · instanttensor + AOT · 1M context · tested: 33.4 single-stream / 71.5 t/s aggregate at 8-way, prefix cache hit (verified 09-10) |
| [DeepSeek-V4-Flash-Vision-Exp (Spark b12x)](../recipes/deepseek-v4-flash-vision-exp-spark-b12x/README.md) | `registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0` | **2-node TP=2** (multimodal) · native image input + thinking mode · **FP8 KV** · dspark k=6 · 1M context · tested: image understanding correct, 27.4 single-stream / 59.1 t/s aggregate at 8-way (verified 09-10) |
| [Qwen3.8-Flash-Next (single-node TP=1 · MiaAI-Lab)](../recipes/qwen38-flash-next-single-dgx-spark/README.md) | `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.2.0` | **single-node TP=1** (176.9B · Mia-AiLab NVFP4 ~99 GB) · **26.82 GiB n-gram/PLE lookup memory-mapped from NVMe** (`VLLM_PLE_CPU_OFFLOAD=1` + MADV_RANDOM + PLE prefetch, 1,366 -> 57 KiB disk read/token) · MTP k=3 + **reduced-vocabulary drafting** (MTP_DRAFT_VOCAB, +25% decode) · **FP8 KV** (~1.85x pool) · **262K native / 512K optional YaRN** · host-side capped budget (KV_TARGET_GIB 20 → GMU 0.786, BF16 GDN + V2 runner pinned, 09-06 aligned) · reasoning on by default · image/video multimodal · decode ~46-49 tok/s single-stream (108-114 aggregate at 4) · prefill peak ~2,265 tok/s @32k (from the MiaAI-Lab single-Spark repo) |
| GLM-5.2 QuantTrio (DCP4) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit` | **4-node TP=4 + DCP4** · B12X MLA SPARSE + a2a · MTP k=2 · **nvfp4_ds_mla KV** · **315,968** context · spark-kit production overlays |
| GLM-5.3-Flash (DFlash2 TP=2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **2-node TP=2** · fp8 KV + **DFlash2** (incoai drafter) · **262K context** · 46.9 tok/s single-stream · C1-C6 zero failures (upstream one-to-copy tier) · pinned 6 GiB KV (678,661-token pool, 09-02) · `--enforce-eager` · default RedHatAI checkpoint |
| GLM-5.3-Flash (DFlash2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **4-node TP=4** (upstream current default) · fp8 KV + **DFlash2** k=7 block-diffusion spec (incoai drafter, ~zero KV-pool cost) · **1M context** · **3.9M-token KV pool** (24 GiB/rank, needs the unconditional flusher) · aggregate ~503-530 tok/s (09-02 speed-up: max-num-seqs 64 / mnbt 16384 / FULL_AND_PIECEWISE) · default RedHatAI checkpoint |
| GLM-5.3-Flash (EXL3 TP=2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.2.0` | **2-node TP=2** · **EXL3/TR3 4bpw weights** (Mia-AiLab mirror, KLD≈official FP8) × fp8 KV + **DFlash2** k=7 · **E3 grouped fat-MoE kernels** (EXL3_FAT_GROUPED=1, +37-45% cold prefill) + EXL3_TEMP_ROWS_FUSED=32 + MNBT 7168 · **850k context** (upstream 09-07 shipped: 850k/0.85/rightsize) · Vision on by default · 62.9 tok/s single-stream (×4 aggregate 146.5) · cold prefill 256k ~1,517 tok/s |

> **Fixed topology**: every recipe declares an exact node count (e.g., 2 nodes · TP=2 or
> 4 nodes · TP=4); Fireworks publish must match it exactly, as parameters are tuned for the
> topology. Pick the matching recipe for your topology.

## Branching model

| Branch | Content |
|---|---|
| `main` | **Hardware-tested** recipes: store-stable, ready to publish |
| `dev` | **In-test** recipes: new/changed recipes land here first, merged to `main` after validation |

Flow: any new/changed recipe goes to `dev` first (docs and `recipes/index.json` travel with the
branch) → validated on real hardware → `git checkout main && git merge dev`. Unfinished / failed
experiments stay on `dev`.

Fireworks lets you pick a branch per recipe source (default `main`), so a running cluster can
preview `dev` recipes and switch back to `main` once stable.

## Catalog (read by Fireworks)

- `recipes/index.json` — the catalog manifest (`id / provider / model / path / readme /
  readme_en / version / params / context_length / modality / nodes / image / tags`, plus
  `name/description` incl. `*_en`). Fireworks reads only this file, no tree scan.
- `recipes/<id>/fireworks.recipe.json` — runnable recipe, aligned with Fireworks
  `POST /api/recipes/import` schema; `image` is a ready-made registry tag.
- `recipes/<id>/README.md` (+ optional `README.en.md`) — rendered in the Recipe Store detail.
- Bilingual fields: `xxx_en` siblings (`name_en / description_en / label_en / help_en`) are
  selected by UI language, falling back to the primary (zh) language.
- Field spec: [docs/RECIPE-FORMAT.md](./RECIPE-FORMAT.md)
- Variable model follows Fireworks: `MASTER_ADDR`=`cluster/head_roce_ip`, `MASTER_PORT` user
  var (default 25000), `NODE_RANK / HEADLESS / VLLM_HOST_IP / NCCL_*` auto-filled node vars.

## Layout

```
FireworksRecipes/
├── LICENSE / NOTICE.md / SECURITY.md   # license, attributions, security
├── .github/workflows/ci.yml            # lightweight validation CI (validate.py)
├── .gitignore
├── recipes/
│   ├── index.json                  # catalog manifest (store data source)
│   ├── deepseek-v4-flash-0731-tp4-4x/   # 4-node TP=4 (agentic-tuned, verified)
│   ├── deepseek-v4-flash-0731-spark-b12x/   # 2-node TP=2 · spark-vllm-b12x (verified on hardware, 09-10)
│   ├── qwen38-flash-next-single-dgx-spark/   # single-node TP=1 · vLLM (PLE table offloaded from NVMe + MTP, from the MiaAI-Lab single-Spark repo)
│   │   ├── fireworks.recipe.json
│   │   └── README.md / README.en.md
│   └── glm-5.2-quanttrio-tp4-dcp4-4x/   # 4-node TP=4 + DCP4 · spark-kit production stack (image built & pushed to ACR)
├── scripts/validate.py   # recipe/manifest validation
├── schemas/  manifest.schema.json · recipe.schema.json
└── docs/  README.en.md · RECIPE-FORMAT.md
```

## Run in Fireworks

1. Recipes page: add this repo as a recipe source → install `recipes/<id>/fireworks.recipe.json`.
2. Images page: ensure the referenced image is pullable (Fireworks distributes to nodes).
3. Publish a task: pick the recipe + matching cluster (head = rank0, exact node count).
4. Verify:

```bash
curl -s http://<head-ip>:8888/v1/models
curl -s http://<head-ip>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"thinking":true}'
```

## Add / change a recipe

1. `cp -r recipes/deepseek-v4-flash-0731-spark-b12x recipes/<new-id>` (drop non-recipe files).
2. Edit `fireworks.recipe.json` (`name / description / image / variable defaults / nodes / version`).
3. Write `README.md` (+ optional `README.en.md`).
4. Register the entry in `recipes/index.json` (must match the recipe's
   `image/version/nodes`).
5. `python3 scripts/validate.py`.
6. Commit to `dev`, merge to `main` after real-hardware validation.

## Validation

`python3 scripts/validate.py` checks recipe schema, catalog-manifest consistency, and that
`source=cluster/node` variables use known `auto` keys. CI runs it on main/dev pushes and PRs
(zero dependencies).

## References & acknowledgments

**Apache-2.0** ([`LICENSE`](../LICENSE)); sources and derivations in
[`NOTICE.md`](../NOTICE.md).

Parameter-level references (recipe tuning sources):
- [jvr0x/dgx-spark-bench](https://github.com/jvr0x/dgx-spark-bench)
- [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
