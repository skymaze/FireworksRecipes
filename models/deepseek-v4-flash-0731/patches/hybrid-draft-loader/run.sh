#!/bin/bash
set -euo pipefail

# hybrid-draft-loader —— 把「目标 InstantTensor / 投机草稿 lazy safetensors」混合
# 加载补丁写入已安装的 vllm wheel（纯文件级 Python 补丁，幂等 + AST 校验）。
#
# 本仓库在【构建期】由 Dockerfile.model 调用，把补丁烘培进专属模型镜像，
# 运行期无需再 apply。也可手动在运行中的容器里执行。
#
# 覆盖 site-packages 路径：VLLM_SITE_PACKAGES

PREFIX="[hybrid-draft-loader]"
PATCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_model_loader.py"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
TARGET="${PYTHON_ROOT}/vllm/model_executor/model_loader/__init__.py"

echo "${PREFIX} PYTHON_ROOT=${PYTHON_ROOT}"

if [ ! -d "${PYTHON_ROOT}/vllm" ]; then
    echo "${PREFIX} ERROR: vLLM 未在 ${PYTHON_ROOT} 找到；请用 VLLM_SITE_PACKAGES 指定实际路径。" >&2
    exit 1
fi
if [ ! -f "${TARGET}" ]; then
    echo "${PREFIX} ERROR: 目标文件不存在：${TARGET}" >&2
    exit 1
fi

python3 "${PATCHER}" --check "${TARGET}"
python3 "${PATCHER}" "${TARGET}"
python3 "${PATCHER}" --check "${TARGET}"

# 清掉 __pycache__，避免运行期加载旧字节码
find "$(dirname "${TARGET}")" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "${PREFIX} 完成：目标模型 InstantTensor，投机草稿 lazy safetensors（FW_HYBRID_DRAFT_LOADER=${FW_HYBRID_DRAFT_LOADER:-auto}）。"
