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
| DeepSeek-V4-Flash (DSpark) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 2-node TP=2 DSpark · FlashInfer b12x + dspark spec · NVFP4 DS-MLA · 1M context |
| DeepSeek-V4-Flash (TP=4) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | **4-node TP=4** DSpark · FlashInfer b12x + dspark k=5 · NVFP4 DS-MLA · **1M context** · verified on a real agentic workload |
| DeepSeek-V4-Flash (Spark b12x) | `eugr/spark-vllm-b12x:latest` | **2-node TP=2** Spark-vLLM · B12X MLA SPARSE + b12x MoE/linear · dspark k=5 · **FP8 KV** · instanttensor + AOT · 1M context |
| Qwen3.8-27B (SGLang DSPARK) | `lmsysorg/sglang:qwen38-27b` | **single node** SGLang · flashinfer + DSPARK spec (mamba draft) · **FP8 KV** · `--mamba-full-memory-ratio 11.01` (suspected typo, to verify) |
| GLM-5.2 QuantTrio (DCP4) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit` | **4-node TP=4 + DCP4** · B12X MLA SPARSE + a2a · MTP k=2 · **nvfp4_ds_mla KV** · **315,968** context · spark-kit production overlays |
| GLM-5.3-Flash (Lane A) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` | **4-node TP=4** · **fp8 KV** (FlashInfer SM12x unlock) · MTP k=4 · **1M context** · ~55 tok/s structured decode · NVFP4 quant (default RedHatAI compressed-tensors, uncensored drop-in available) · ⚠️ upstream has marked MTP TP4 superseded — use DFlash2 for new deployments |
| GLM-5.3-Flash (DFlash2 TP=2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **2-node TP=2** · fp8 KV + **DFlash2** (incoai drafter) · **262K context** · 46.9 tok/s single-stream · C1-C6 zero failures (upstream one-to-copy tier) · KV profiler-sized (581K, do not pin) · default RedHatAI checkpoint |
| GLM-5.3-Flash (DFlash2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **4-node TP=4** (upstream current default) · fp8 KV + **DFlash2** k=7 block-diffusion spec (incoai drafter, ~zero KV-pool cost) · **1M context** · **3.9M-token KV pool** (24 GiB/rank, needs the unconditional flusher) · 54.5 tok/s single-stream · default RedHatAI checkpoint |
| GLM-5.3-Flash (EXL3 TP=2) | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | **2-node TP=2** · **EXL3/TR3 4bpw weights** (Mia-AiLab mirror, KLD≈official FP8) × fp8 KV + **DFlash2** k=7 · **1M context** (padded slot-share) · Vision on by default · 62.9 tok/s single-stream (×4 aggregate 146.5) |

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
│   ├── deepseek-v4-flash-dspark/   # fireworks.recipe.json + README(.en)
│   ├── deepseek-v4-flash-0731-tp4-4x/   # 4-node TP=4 (agentic-tuned, verified)
│   ├── deepseek-v4-flash-0731-spark-b12x/   # 2-node TP=2 · eugr/spark-vllm-b12x (from docker run, not yet verified)
│   ├── qwen38-27b-sglang-dspark/   # single node · SGLang + DSPARK (from docker run, not yet verified)
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

1. `cp -r recipes/deepseek-v4-flash-dspark recipes/<new-id>` (drop non-recipe files).
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
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
