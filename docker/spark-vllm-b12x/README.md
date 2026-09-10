# spark-vllm-b12x 派生镜像（FireworksRecipes）

本目录产出 b12x 服务线的自烘焙镜像：以 `eugr/spark-vllm-b12x:nightly-20260909`
（digest `sha256:4f5482a…`，2026-09-09）为基镜像，把上游
`mods/instanttensor-hybrid-draft-loader`（eugr/spark-vllm-docker @ `60dee5b`，MIT）
**在构建时**打入镜像，使 dspark 投机草稿在容器启动时即走 lazy safetensors，无需节点
联网拉补丁、也不依赖运行时 `--apply-mod`。

## 构建与推送

```bash
cd docker/spark-vllm-b12x
docker build -t registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0 .
docker push registry.cn-shanghai.aliyuncs.com/aixn-public/spark-vllm-b12x:v1.0.0
```

## 镜像内容

- 基镜像：`eugr/spark-vllm-b12x`（b12x 分支预构建 vLLM 分发镜像；钉在 09-09 nightly）
- 附加：`/opt/spark-vllm-mods/instanttensor-hybrid-draft-loader/`（run.sh + patch_model_loader.py），
  由 Dockerfile `RUN` 在构建期执行，patch 进 `vllm/model_executor/model_loader/__init__.py`
- `INSTANTTENSOR_DRAFT_LOADER` 默认 `auto`：draft 与 target 同模型路径/revision 时，
  draft 用 `load_format=safetensors` + `safetensors_load_strategy=lazy`，target 保持 InstantTensor

## 为什么烘焙而不是运行时 apply

上游 `launch-cluster.sh --apply-mod` 在容器启动时从宿主机挂载 mod 并打补丁；Fireworks
节点为离线部署（模型/镜像由控制平面分发），运行时拉取上游脚本既引入故障点，也无法
随配方固定版本。烘焙进镜像后行为一致且可复现：patch 精确匹配基镜像的 vLLM 源码，
源码不匹配时构建直接失败（fail-fast），不会产出半吊子镜像。

## 维护

- 上游 `eugr/spark-vllm-b12x` 每天出 nightly；升级基镜像时必须重建并确认 patch 仍能
  应用（`run.sh` + patcher 会硬性校验），成功后提升镜像 tag 并同步
  `recipes/*/fireworks.recipe.json`、`recipes/index.json`、`NOTICE.md`。
- vLLM 支持 per-draft load-format 后（mod README 的既定目标），移除本烘焙并回到纯基镜像。

## 归属

- 镜像/基镜像与 mod 源码版权归各上游所有：`eugr/spark-vllm-docker`（MIT，
  Copyright (c) 2026 Eugene Rakhmatulin）、vLLM（Apache-2.0）、B12X 项目等；
  详见仓库根 [`NOTICE.md`](../../NOTICE.md)。
