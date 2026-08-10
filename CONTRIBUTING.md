# Contributing

Thanks for your interest in FireworksRecipes! This project builds per-model serving
images for DGX Spark (GB10) — most changes fall into a few well-defined buckets.

## Development flow

1. **Fork & branch** off `master`. Use descriptive branch names
   (`feat/`, `fix/`, `docs/`).
2. **Keep the repo buildable from a clean checkout.** Build artifacts that are
   git-ignored (`.build-ref/`, `dist/`, `logs/`, `seed-cache/`, `__pycache__/`)
   must never be committed. The default docker build must succeed without them
   (the committed `seed-cache-empty/` placeholder guarantees this).
3. **Version pins live in `versions.conf`.** When bumping vLLM / FlashInfer /
   torch etc., update the lock, then adjust `overlay/vllm/` anchors if needed and
   re-run `./scripts/build.sh base && ./scripts/build.sh model`.
4. **Patch changes** go under `recipes/<id>/patches/<name>/` with:
   - an idempotent apply script (`run.sh` + `patch_*.py`),
   - **AST validation and anchor checks** that *fail loudly* on version drift
     (never silently corrupt a file), following `hybrid-draft-loader` /
     `fw-warmup` as templates.
5. **License headers**: files derived from vLLM carry
   `SPDX-License-Identifier: Apache-2.0` and the vLLM copyright line. Preserve
   them. Anything copied verbatim from another project must be flagged in
   `NOTICE.md` and kept in-file.
6. **No hardcoded lab addresses** in scripts. Node IPs / base URLs are read from
   env vars with `<placeholder>` documentation defaults.

## Before submitting

- `bash -n` on all changed shell scripts; `python3 -m py_compile` on Python files;
  validate any `.json` (recipe) parses.
- Run `git status --short --ignored` to confirm no build artifacts are staged.
- If behavior/performance changed, update `docs/BENCHMARK-v0260.md` accordingly.

## Commit style

Concise imperative subject lines (≤ ~70 chars), body explains *why*. Squash
work-in-progress commits before opening a PR.

## 提 PR 前的本地检查

```bash
python3 scripts/validate.py                 # 配方/manifest 全部校验（schema + 一致性 + auto 键）
bash scripts/render-model-dockerfile.sh --check deepseek-v4-flash-0731   # Dockerfile.model 与模板一致
```

- 新增/修改配方：改 `recipes/<id>/recipe/fireworks.recipe.json` + `README(_en).md`，并在 `recipes/index.json` 登记（`image/version/nodes/tensor_parallel` 必须与配方一致，否则 `validate.py` 报错）。
- 变量 `source=cluster/node` 必须用 `auto` 已知键（见 `docs/RECIPE-FORMAT.md`《自动填充键》）；新增键需同步该表与 Fireworks 后端 `recipe_render.py` 的 `AUTO_KEYS`。
- 自建模型镜像：在 `recipes/<id>/build.conf` 声明 `MODEL_PATCH_DIRS/MODEL_TITLE/MODEL_LABEL_*`，补丁片段放 `templates/patches/<dir>.inc`，运行 `bash scripts/render-model-dockerfile.sh <model>` 生成 `Dockerfile.model` 后再提交（不许手改）。

## CI 与镜像构建

- `.github/workflows/ci.yml` 的 `validate` 在任意 PR/push 上跑（零依赖）；镜像构建/推送**不走 CI**。
- 镜像构建/推送到 registry 在本地 aarch64（DGX Spark / GB10）构建机完成：
  `SEED_CACHE_DIR=seed-cache bash scripts/build.sh base` →
  `bash scripts/build.sh model --model <name> --push-registry <registry>`。
  例如推送本配方到 GHCR：`bash scripts/build.sh model --model deepseek-v4-flash-0731 --push-registry ghcr.io/skymaze`（需先 `docker login ghcr.io`）。
