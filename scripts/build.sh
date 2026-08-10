#!/bin/bash
set -euo pipefail

# =============================================================================
# scripts/build.sh —— 本地源码编译驱动（base / model / all）
#
#   base  编译基础 runner（docker/vllm-b12x.Dockerfile，全部组件本地源码编译）
#   model 在 base 之上构建模型专属镜像（recipes/<id>/Dockerfile.model）
#   all   先 base 后 model
#
# 用法：
#   ./scripts/build.sh all --model deepseek-v4-flash-0731
#   ./scripts/build.sh base               # 只编译基础 runner
#   ./scripts/build.sh model --skip-base  # 已有 base 时只构建模型层（避免重复编译）
#   ./scripts/build.sh model --push-registry ghcr.io/<org>
#   ./scripts/build.sh model --save dist/
#   ./scripts/build.sh all --json         # 机器可读输出
#
# 版本只读 versions.conf（构建参数唯一来源）；升级 vLLM 需同步 overlay/vllm/ 源码覆写
# （版本漂移时 overlay 锚点硬失败提示），然后：./scripts/build.sh base && ./scripts/build.sh model
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
MODEL_DEFAULT="deepseek-v4-flash-0731"

log()  { printf '\033[1;36m[build]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build!]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[build ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------- 参数解析 ----------
TARGET=""
MODEL="${MODEL_DEFAULT}"
PUSH_REGISTRY=""
SAVE_DIR=""
PRINT_JSON=false
SKIP_BASE=false
FORCE_BASE=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        base|model|all)
            [[ -n "$TARGET" ]] && fail "目标只能指定一次：$1"
            TARGET="$1"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --push-registry) PUSH_REGISTRY="$2"; shift 2 ;;
        --push-registry=*) PUSH_REGISTRY="${1#*=}"; shift ;;
        --save) SAVE_DIR="$2"; shift 2 ;;
        --save=*) SAVE_DIR="${1#*=}"; shift ;;
        --skip-base) SKIP_BASE=true; shift ;;
        --force-base) FORCE_BASE=true; shift ;;
        --json) PRINT_JSON=true; shift ;;
        -h|--help|help) usage ;;
        *) fail "未知参数: $1（见 --help）" ;;
    esac
done
TARGET="${TARGET:-all}"

# ---------- 配置 ----------
# shellcheck disable=SC1091
source "${REPO_ROOT}/versions.conf"

MODEL_DIR="${REPO_ROOT}/recipes/${MODEL}"
if [[ ! -d "${MODEL_DIR}" ]]; then
    fail "模型目录不存在: ${MODEL_DIR}（--model 指定 recipes/ 下的目录名）"
fi
# shellcheck disable=SC1091
source "${MODEL_DIR}/build.conf"

mkdir -p "${LOG_DIR}"

# ---------- 环境预检 ----------
command -v docker >/dev/null 2>&1 || fail "未找到 docker"
docker info >/dev/null 2>&1 || fail "docker daemon 不可用（无权限或未启动）"

DOCKER_ARCH="$(docker info --format '{{.Architecture}}' 2>/dev/null || echo unknown)"
if [[ "${DOCKER_ARCH}" != "aarch64" && "${DOCKER_ARCH}" != "arm64" ]]; then
    warn "当前 docker 架构为 ${DOCKER_ARCH}，目标是 linux/arm64 (DGX Spark/GB10)。"
    warn "请使用 aarch64 构建机，否则编译产物无法在节点运行。"
fi

docker buildx version >/dev/null 2>&1 \
    || warn "未检测到 buildx；本 Dockerfile 使用 BuildKit 特性（mount=type=cache），需要 Docker >= 23 且开启 BuildKit。"

# ========================== 1) 编译基础 runner ===========================
build_base() {
    local base_tag="$1"
    local tag
    tag="$(date +%Y%m%d-%H%M%S)"
    local logfile="${LOG_DIR}/base-${tag}.log"

    log "编译基础 runner: ${base_tag}（源码编译 flashinfer/vllm/nccl，耗时较长，日志: ${logfile}）"
    docker build \
        --progress=plain \
        -t "${base_tag}" \
        -f "${REPO_ROOT}/docker/vllm-b12x.Dockerfile" \
        --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
        --build-arg "GPU_ARCH=${GPU_ARCH}" \
        --build-arg "NCCL_NVCC_GENCODE=${NCCL_NVCC_GENCODE}" \
        --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
        --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
        --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
        --build-arg "FLASHINFER_REPO=${FLASHINFER_REPO}" \
        --build-arg "FLASHINFER_REF=${FLASHINFER_REF}" \
        --build-arg "VLLM_REPO=${VLLM_REPO}" \
        --build-arg "VLLM_REF=${VLLM_REF}" \
        --build-arg "SPARKINFER_REPO=${SPARKINFER_REPO}" \
        --build-arg "SPARKINFER_REF=${SPARKINFER_REF}" \
        --build-arg "DEEPGEMM_REPO=${DEEPGEMM_REPO}" \
        --build-arg "DEEPGEMM_REF=${DEEPGEMM_REF}" \
        --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" \
        --build-arg "APACHE_TVM_FFI_VERSION=${APACHE_TVM_FFI_VERSION}" \
        --build-arg "CUDA_PYTHON_VERSION=${CUDA_PYTHON_VERSION}" \
        "${REPO_ROOT}" 2>&1 | tee "${logfile}"
    log "基础 runner 完成: ${base_tag}"
}

# ========================= 2) 构建模型专属镜像 ============================
build_model() {
    local base_tag="$1"
    local image_repo="$2"
    local image_tag="$3"
    local final_image="${PUSH_REGISTRY:+${PUSH_REGISTRY}/}${image_repo}:${image_tag}"

    # base 就绪校验：缺失或 --force-base 时编译；--skip-base 时要求已存在
    if docker image inspect "${base_tag}" >/dev/null 2>&1; then
        if [[ "${FORCE_BASE}" == true ]]; then
            build_base "${base_tag}"
        else
            log "复用已有基础镜像: ${base_tag}（如需重新编译基础层用 --force-base 或 ./scripts/build.sh base）"
        fi
    else
        [[ "${SKIP_BASE}" == true ]] \
            && fail "使用了 --skip-base 但本地无基础镜像 ${base_tag}，请先 ./scripts/build.sh base"
        build_base "${base_tag}"
    fi

    local tag
    tag="$(date +%Y%m%d-%H%M%S)"
    local logfile="${LOG_DIR}/model-${MODEL}-${tag}.log"

    log "构建模型专属镜像: ${final_image}（base=${base_tag}）"
    local seed_arg=()
    if [[ -n "${SEED_CACHE_DIR:-}" ]]; then
        seed_arg=(--build-arg "SEED_CACHE_DIR=${SEED_CACHE_DIR}")
        log "  JIT 缓存种子: ${SEED_CACHE_DIR}（bake 进 /opt/fw/vllm-cache-seed）"
    fi
    docker build \
        --progress=plain \
        -t "${final_image}" \
        -f "${MODEL_DIR}/Dockerfile.model" \
        --build-arg "BASE_IMAGE=${base_tag}" \
        "${seed_arg[@]}" \
        "${MODEL_DIR}" 2>&1 | tee "${logfile}"
    log "模型专属镜像完成: ${final_image}"

    # 分发
    if [[ -n "${PUSH_REGISTRY}" ]]; then
        log "push → ${final_image}"
        docker push "${final_image}"
    fi
    if [[ -n "${SAVE_DIR}" ]]; then
        mkdir -p "${SAVE_DIR}"
        local archive="${SAVE_DIR}/${image_repo//\//_}-${image_tag}.tar"
        log "save → ${archive}（可供 Fireworks 镜像管理 docker load）"
        docker save -o "${archive}" "${final_image}"
    fi

    # 与 recipe.json.image 的一致性校验（仅当未 push-registry 时唯一确定）
    check_recipe_image "${final_image}"

    if [[ "${PRINT_JSON}" == true ]]; then
        printf '{"model":"%s","base_tag":"%s","image":"%s","arch":"%s","log_dir":"%s"}\n' \
            "${MODEL}" "${base_tag}" "${final_image}" "${DOCKER_ARCH}" "${LOG_DIR}"
    fi
}

# recipe.json.image 与最终 tag 一致性提示
check_recipe_image() {
    local final_image="$1"
    local recipe_file="${MODEL_DIR}/fireworks.recipe.json"
    [[ -f "${recipe_file}" ]] || return 0
    command -v python3 >/dev/null 2>&1 || return 0
    local recipe_image
    recipe_image="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("image",""))' "${recipe_file}" 2>/dev/null || true)"
    if [[ "${recipe_image}" != "${final_image}" ]]; then
        if [[ -n "${PUSH_REGISTRY}" ]]; then
            warn "配方 recipe.json 的 image 字段是「${recipe_image}」，而本次产出为「${final_image}」。"
            warn "推送/拉取用全名（registry 前缀）时，请把配方 image 更新为 ${final_image}（导入 Fireworks 前改）。"
        else
            warn "配方 recipe.json 的 image 字段「${recipe_image}」与产出「${final_image}」不一致，请对齐后导入。"
        fi
    else
        log "recipe.json.image 与镜像 tag 一致 ✓"
    fi
}

# ---------- 执行 ----------
case "${TARGET}" in
    base)
        build_base "${BASE_TAG}"
        if [[ "${PRINT_JSON}" == true ]]; then
            printf '{"model":null,"base_tag":"%s","image":null,"arch":"%s","log_dir":"%s"}\n' \
                "${BASE_TAG}" "${DOCKER_ARCH}" "${LOG_DIR}"
        fi
        ;;
    model)
        build_model "${BASE_TAG}" "${IMAGE_REPO}" "${IMAGE_TAG}"
        ;;
    all)
        build_model "${BASE_TAG}" "${IMAGE_REPO}" "${IMAGE_TAG}"
        ;;
esac
