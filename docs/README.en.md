# 🎇 FireworksRecipes

DGX Spark (GB10) **model recipes** for **Fireworks** (DGX Spark cluster manager): each
recipe is a `fireworks.recipe.json` you can import directly (image / fixed topology / tuned
parameters / auto-filled network vars), plus the `recipes/index.json` catalog the "Recipe
Store" reads.

> This repository ships **recipes and catalog only — no image build code.** Recipes reference
> ready-made image tags; Fireworks pulls and distributes them to nodes.

**中文**: [../README.md](../README.md)

Current recipes:

| Recipe | Image | Description |
|---|---|---|
| DeepSeek-V4-Flash (DSpark) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 2-node TP=2 DSpark · FlashInfer b12x + dspark spec · NVFP4 DS-MLA · 1M context |
| DeepSeek-V4-Flash (TP=4) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | **4-node TP=4** DSpark · FlashInfer b12x + dspark k=5 · NVFP4 DS-MLA · **1M context** · verified on a real agentic workload |

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
  readme_en / version / params / context_length / modality / nodes / image / tags`). Fireworks
  reads only this file, no tree scan.
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
├── LICENSE / NOTICE.md   # Apache-2.0 license & attributions
├── recipes/
│   ├── index.json                  # catalog manifest (store data source)
│   ├── deepseek-v4-flash-dspark/   # fireworks.recipe.json + README(.en)
│   └── deepseek-v4-flash-0731-tp4-4x/   # 4-node TP=4 (agentic-tuned, verified)
├── scripts/validate.py   # recipe/manifest validation
├── schemas/manifest.schema.json
└── docs/  README.en.md · RECIPE-FORMAT.md
```

## Run in Fireworks

1. Recipes page: add this repo as a recipe source → install `recipes/<id>/fireworks.recipe.json`.
2. Images page: ensure the referenced image is pullable (Fireworks distributes to nodes).
3. Publish a task: pick the recipe + matching cluster (head = rank0, exact node count).
4. Verify:

```bash
curl -s http://<head-ip>:8000/v1/models
curl -s http://<head-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"thinking":true}'
```

## Add / change a recipe

1. `cp -r recipes/deepseek-v4-flash-dspark recipes/<new-id>` (drop non-recipe files).
2. Edit `fireworks.recipe.json` (`name / description / image / variable defaults / nodes / version`).
3. Write `README.md` (+ optional `README.en.md`).
4. Register the entry in `recipes/index.json` (must match the recipe's
   `image/version/nodes/tensor_parallel`).
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
