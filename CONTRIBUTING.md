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
4. **Patch changes** go under `models/<model>/patches/<name>/` with:
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
