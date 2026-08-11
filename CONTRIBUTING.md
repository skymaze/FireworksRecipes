# Contributing

Thanks for your interest in FireworksRecipes! This repository provides Fireworks model
recipes for DGX Spark (GB10) — most changes are to a single recipe directory.

## Development flow

1. **Branch off `dev` first.** Branching model: `main` = tested recipes (store-stable); new /
   changed recipes land on `dev` first and are merged to `main` only after real-hardware
   validation. Use `feat/`, `fix/`, `docs/` prefixed branch/commit names.
2. **No build artifacts in the repo.** Local leftovers (`logs/`, `dist/`, `.build-ref/`,
   `seed-cache/`, `__pycache__/`) are git-ignored; don't stage them.
3. **A recipe change touches these files together:**
   - `recipes/<id>/fireworks.recipe.json` — parameters / image / topology.
   - `recipes/<id>/README.md` (+ optional `README.en.md`) — user documentation.
   - `recipes/index.json` — catalog entry (`image/version/nodes/tensor_parallel` must match
     the recipe, or `validate.py` fails).
   - Variables with `source=cluster/node` must use a known `auto` key (see
     `docs/RECIPE-FORMAT.md` "Auto-fill keys"); new keys must be mirrored in the Fireworks
     backend's `recipe_render.AUTO_KEYS`.

## Before submitting

```bash
python3 scripts/validate.py                 # recipes/manifest validation (schema + consistency + auto keys)
python3 -m json.tool recipes/<id>/fireworks.recipe.json >/dev/null   # changed recipe JSON parses
```

- Run `git status --short --ignored` to confirm no stale local artifacts get staged.

## CI

`.github/workflows/ci.yml` runs `scripts/validate.py` on `main`/`dev` pushes and PRs
(zero dependencies, no image build). Keep recipe changes green there.

## Commit style

Concise imperative subjects (≤ ~70 chars); body explains *why*. Squash work-in-progress
commits before merging to `main`.
