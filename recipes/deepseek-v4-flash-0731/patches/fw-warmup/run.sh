#!/usr/bin/env bash
# =============================================================================
# run.sh —— fw-warmup 补丁执行入口（Dockerfile.model 构建期调用）
#
# 幂等：patch_warmup.py 带 MARKER 标记，重复执行安全（已补丁则跳过）。
# 安全降级：锚点漂移/形状不符时拒绝打补丁（退出码 1 → 构建失败，宁可显式
# 失败也不静默打坏文件）。若确认上游已修复，删除本目录即可。
# =============================================================================
set -euo pipefail

VLLM_SITE="${VLLM_SITE:-/usr/local/lib/python3.12/dist-packages/vllm}"
WARMUP_DIR="${VLLM_SITE}/model_executor/warmup"

echo "[fw-warmup] target: ${WARMUP_DIR}"
python3 "$(dirname "$0")/patch_warmup.py" "${WARMUP_DIR}" --check
python3 "$(dirname "$0")/patch_warmup.py" "${WARMUP_DIR}"
echo "[fw-warmup] done."
