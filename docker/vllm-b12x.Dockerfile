# syntax=docker/dockerfile:1

# =============================================================================
#  docker/vllm-b12x.Dockerfile —— 本地源码编译多阶段构建（v0.26.0 主路径）
#
#  全部组件从 pinned 源码仓库编译（clone → 编译 → wheel → 安装进 runner），
#  不拉取任何第三方预编译镜像/预编译产物，运行期无外部项目依赖。
#
#  ★ 基座 = **主流 vllm-project/vllm v0.26.0**（2026-07 发布），放弃自定义
#    b12x fork（local-inference-lab/dev/gilded-gnosis）。DGX Spark（GB10 /
#    sm_121a / aarch64）性能来自 Anemll/dspark-vllm-gx10 配方的定向移植：
#      - 显式 12.1a 编译 arch（TORCH_CUDA_ARCH_LIST + CMakeLists 12.1 保留）
#      - nvfp4_ds_mla KV dtype + b12x（SparkInfer）MXFP4 MoE → flashinfer_b12x
#      - dspark MTP 投机解码（v0.26.0 主流内建）
#    Python 层覆盖（overlay/vllm/，若存在则在 build 前 rsync 进源码）由构建
#    期烘培进 wheel，运行期不再 apply。
#
#  版本全部来自 versions.conf（scripts/build.sh 以 --build-arg 显式传入）。
#  升级 vLLM：改 versions.conf 外，还需同步 overlay/vllm/ 源码覆写（漂移硬失败提示）。
#  需要 Docker BuildKit（mount=type=cache/bind）。GB10 arm64 构建机。
# =============================================================================

# ---- 顶层 ARG（默认值与 versions.conf 保持一致，脚本会显式覆盖）----
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG GPU_ARCH=12.1a
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG BUILD_JOBS=16
ARG FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git
ARG FLASHINFER_REF=0472b9b3f2fba11b463f8526f390297d52a8aad7
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.26.0
ARG SPARKINFER_REPO=https://github.com/lukealonso/b12x.git
ARG SPARKINFER_REF=7dc6fb8fcc6446ea093537d1657df81985fa5f43
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ARG CUTLASS_DSL_VERSION=4.6.0
ARG APACHE_TVM_FFI_VERSION=0.1.10
ARG CUDA_PYTHON_VERSION=13.3.1


# ============================ 阶段 1: base ==============================
FROM ${CUDA_IMAGE} AS base

ARG BUILD_JOBS
ARG TORCH_VERSION
ARG TORCH_INDEX_URL
ARG NCCL_NVCC_GENCODE
ARG GPU_ARCH
ARG APACHE_TVM_FFI_VERSION

# 编译并行度 + 运行期 JIT 一致性（与 runner 保持一致）
ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=0 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    TORCH_CUDA_ARCH_LIST=${GPU_ARCH} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PATH=/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

# 编译依赖（ccache 加速增量编译；rsync 供 overlay 合入源码；devscripts 等用于 NCCL deb 打包）
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates vim git wget rsync \
        build-essential cmake ninja-build ccache pkg-config \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
        python3-dev python3-pip \
        libibverbs1 libibverbs-dev rdma-core \
        devscripts debhelper fakeroot \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# PyTorch（cu130 索引 / aarch64）——编译期与运行期共用同一 torch，避免 ABI 漂移
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
    uv pip install "torch==${TORCH_VERSION}" triton \
        --index-url "${TORCH_INDEX_URL}"

# NVIDIA/Ryzen AI 运行时附加包（仅在 PyPI）：nvshmem cu13 / Apache TVM FFI
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install \
        nvidia-nvshmem-cu13 "apache-tvm-ffi==${APACHE_TVM_FFI_VERSION}" \
        filelock pynvml requests tqdm

# NCCL 从源码编译（sm_121 gencode；打成 deb 供 base 编译与 runner 运行共用）
WORKDIR /workspace/nccl
RUN git clone https://github.com/NVIDIA/nccl.git . \
    && make -j "${MAX_JOBS}" src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" \
    && make pkg.debian.build \
    && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        ./build/pkg/deb/*.deb \
    && rm -rf /var/lib/apt/lists/*


# ========================== 阶段 2: flashinfer ===========================
FROM base AS flashinfer

ARG FLASHINFER_REPO
ARG FLASHINFER_REF
ARG GPU_ARCH

ENV FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH}

WORKDIR /workspace
# 优先浅克隆指定分支/tag；不存在（如具体 commit）则整克隆后 checkout
RUN set -eux; \
    if ! git clone --recursive --depth 1 --branch "${FLASHINFER_REF}" "${FLASHINFER_REPO}" flashinfer 2>/dev/null; then \
        git clone --recursive "${FLASHINFER_REPO}" flashinfer; \
        git -C flashinfer checkout --detach "${FLASHINFER_REF}"; \
    fi

WORKDIR /workspace/flashinfer
# flashinfer-python：主 wheel。
# 注：uv build 对旧版 PEP 621 license 字段的兼容性修复，若上游已修复可移除。
RUN sed -i \
        -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' \
        -e '/license-files/d' pyproject.toml || true

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    uv pip install --no-deps packaging build hatchling wheel \
    && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels

# flashinfer-cubin / flashinfer-jit-cache：预编译 cubin 包，构建期要从 NVIDIA 制品库
# 顺序下载 ~2.3 万个 cubin（RTT 受限，外网带宽低时可达数小时）。
# v0.26.0 的 sparse-MLA/DeepSeek 路径走 flashinfer-python 运行时 JIT（runner 内
# 有 ptxas + 12.1a），cubin 包非必需；默认 0=跳过。需要预编译 cubin 时置 1（很慢）。
ARG FW_FLASHINFER_CUBINS=0
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    set -eux; \
    if [ "${FW_FLASHINFER_CUBINS:-0}" = "1" ]; then \
        for sub in flashinfer-cubin flashinfer-jit-cache; do \
            if [ -d "$sub" ]; then \
                (cd "$sub" && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels); \
            fi; \
        done; \
    else \
        echo "FW_FLASHINFER_CUBINS=0：跳过 flashinfer-cubin/jit-cache（sparse-MLA 走运行时 JIT）。"; \
    fi


# ============================= 阶段 3: vllm ==============================
FROM base AS vllm

ARG VLLM_REPO
ARG VLLM_REF
ARG DEEPGEMM_REPO
ARG DEEPGEMM_REF
ARG GPU_ARCH

ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH}

# Rust 工具链（vLLM Rust 前端）+ protobuf
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path

ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

# vLLM 源码（主流 v0.26.0）
WORKDIR /workspace
RUN set -eux; \
    if ! git clone --recursive --depth 1 --branch "${VLLM_REF}" "${VLLM_REPO}" vllm 2>/dev/null; then \
        git clone --recursive "${VLLM_REPO}" vllm; \
        git -C vllm checkout --detach "${VLLM_REF}"; \
    fi

# DeepGEMM（DeepSeek 内核，DS 模型 llama/deepseek MoE 计算路径）
RUN set -eux; \
    if ! git clone --recursive --depth 1 --branch "${DEEPGEMM_REF}" "${DEEPGEMM_REPO}" DeepGEMM 2>/dev/null; then \
        git clone --recursive "${DEEPGEMM_REPO}" DeepGEMM; \
        git -C DeepGEMM checkout --detach "${DEEPGEMM_REF}"; \
    fi

# ★ 应用 Python 层 overlay（overlay/vllm/，Anemll 配方定向移植）。
#    存在才合入（rsync -a：新增文件 + 覆盖同名文件），构建期烘培进 wheel。
#    注：与 Anemll「运行时再 COPY 一份到 dist-packages」不同，本仓库在 wheel 内
#    烘焙，运行期无需 apply。
WORKDIR /workspace
COPY overlay/ /workspace/overlay/
RUN set -eux; \
    if [ -d /workspace/overlay/vllm ] && [ -n "$(ls -A /workspace/overlay/vllm)" ]; then \
        rsync -a /workspace/overlay/vllm/ /workspace/vllm/vllm/; \
        echo "overlay applied:"; find /workspace/overlay/vllm -type f | sed 's#^/workspace/##'; \
    else \
        echo "overlay empty: no source overlay applied (mainline stock v0.26.0)."; \
    fi

# 复用 base 已装的 torch；FlashInfer 不在此阶段安装（用我们自己的 wheel）
WORKDIR /workspace/vllm

# CUDA 13 的 vLLM 构建默认把请求的 12.1a 折叠成通用 12.0(sm_120)，导致 GB10 专属
# FP4(e2m1x2) cvt 指令无法通过 ptxas。把 12.1 加入 CUDA_SUPPORTED_ARCHS，
# 使 CMake 交集运算保留 12.1a 目标（DGX Spark 专属；v0.26.0 CUDA>=13 默认串已确认相同）。
ARG VLLM_PRESERVE_SM12X_TARGET=1
RUN VLLM_PRESERVE_SM12X_TARGET="${VLLM_PRESERVE_SM12X_TARGET}" python3 - <<'PY'
import os
from pathlib import Path

if os.environ.get("VLLM_PRESERVE_SM12X_TARGET") not in {
    "1", "true", "TRUE", "yes", "YES"
}:
    print("SM12x target preservation not requested; skipping")
    raise SystemExit(0)

target = Path("CMakeLists.txt")
old = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
new = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")'
text = target.read_text()
if new in text:
    print("CUDA 13 SM12x subarchitecture allow-list already present; skipping")
elif text.count(old) == 1:
    target.write_text(text.replace(old, new, 1))
    print("Enabled selected SM12x target preservation for CUDA 13 vLLM build")
else:
    raise SystemExit(
        "SM12x preservation: expected CUDA 13 CUDA_SUPPORTED_ARCHS was not found exactly once"
    )
PY

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py \
    && sed -i "/flashinfer/d" requirements/cuda.txt \
    && sed -i "/^triton\b/d" requirements/test/cuda.txt \
    && sed -i "/^fastsafetensors\b/d" requirements/test/cuda.txt \
    && uv pip install -r requirements/build/cuda.txt "setuptools-rust>=1.9.0"

# 编译 vLLM wheel（Rust 前端；cargo/ccache 缓存加速迭代构建）
# 注：triton_kernels 由 vLLM CMake FetchContent 全量克隆
# （triton-lang/triton@v3.5.1）后源码构建（需 MLIR，由构建环境提供）；
# 曾尝试 TRITON_KERNELS_SRC_DIR 本地源绕开，但子项目 find_package(MLIR)
# 无法满足，回退默认 fetch（M2/M3 已验证可构建；偶发网络慢则重试构建即可）。
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=cargo-git,target=/root/.cargo/git \
    --mount=type=cache,id=vllm-rust-target,target=/workspace/vllm/vllm/target \
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS="${MAX_JOBS}" \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels


# ============================= 阶段 4: runner ============================
FROM ${CUDA_IMAGE} AS runner

ARG TORCH_VERSION
ARG TORCH_INDEX_URL
ARG BUILD_JOBS
ARG GPU_ARCH
ARG SPARKINFER_REPO
ARG SPARKINFER_REF
ARG SPARKINFER_CACHEBUST=1
ARG CUTLASS_DSL_VERSION
ARG APACHE_TVM_FFI_VERSION
ARG CUDA_PYTHON_VERSION

# 与 base 一致的编译期变量：运行期 vLLM/内核仍会做 JIT，需要 ptxas 与 arch 一致
ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=0 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    FLASHINFER_DISABLE_VERSION_CHECK=1

# 运行时系统依赖 + 安装 base 编译的 NCCL deb（避免运行期再编译）
COPY --from=base /workspace/nccl/build/pkg/deb /workspace/nccl-pkg
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev curl git wget \
        libcudnn9-cuda-13 \
        libibverbs1 libibverbs-dev rdma-core \
        libxcb1 earlyoom \
    && (cd /workspace/nccl-pkg && apt-get install -y --no-install-recommends \
        --allow-downgrades --allow-change-held-packages ./*.deb) \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# tiktoken 词表（路由 lineage 加载）
RUN mkdir -p /workspace/vllm/tiktoken_encodings \
    && wget -q -O /workspace/vllm/tiktoken_encodings/o200k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" \
    && wget -q -O /workspace/vllm/tiktoken_encodings/cl100k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# PyTorch 运行期（与 base 同版本同索引；torchvision 为 vLLM 内核 warmup
# 传导依赖（如 MiniMax-M3 warmup 模块 import torchvision.transforms）所需）
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install "torch==${TORCH_VERSION}" triton torchvision \
        --index-url "${TORCH_INDEX_URL}"

# 附加运行时包（PyPI）：nvshmem cu13 / Apache TVM FFI / cuda-python / CuTe DSL
# （cutlass-dsl 与 v0.26.0 wheel 依赖同版本，显式 pin 防解析漂移）
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install \
        nvidia-nvshmem-cu13 "apache-tvm-ffi==${APACHE_TVM_FFI_VERSION}" \
        "cuda-python==${CUDA_PYTHON_VERSION}" \
        "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
        "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}" \
        "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}"

# 安装编译产物 wheel（--override 固定 torch/fastapi，防止依赖解析把 cu130 torch 换掉）
COPY --from=flashinfer /workspace/wheels /workspace/wheels/flashinfer
COPY --from=vllm /workspace/wheels /workspace/wheels/vllm
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH="$(python3 -c 'import torch; print(torch.__version__)')" \
    && printf 'torch==%s\nfastapi[standard]>=0.115.0,<0.137.0\n' "${PINNED_TORCH}" \
        > /tmp/wheel-override.txt \
    && uv pip install /workspace/wheels/vllm/*.whl \
        --override /tmp/wheel-override.txt

# ★ 用本地源码编译的 flashinfer wheel 覆盖 vLLM wheel 依赖锁定的 0.6.14
#   （v0.26.0 pin）为 Anemll 验证的 0.6.15(git 0472b9b)——sparse-MLA 数据路径
#   需要；FLASHINFER_DISABLE_VERSION_CHECK=1 避免版本横幅校验失败。
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install --force-reinstall --no-deps /workspace/wheels/flashinfer/*.whl

# 推理运行时额外依赖：fastsafetensors / instanttensor（InstantTensor 加载器）/ ray
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH="$(python3 -c 'import torch; print(torch.__version__)')" \
    && printf 'torch==%s\nfastapi[standard]>=0.115.0,<0.137.0\n' "${PINNED_TORCH}" \
        > /tmp/torch-override.txt \
    && uv pip install 'ray[default]' fastsafetensors instanttensor \
        --override /tmp/torch-override.txt

# SparkInfer（B12X MXFP4 MoE 内核包 lukealonso/b12x）从源码安装；内核 JIT 首次使用编译
# --no-deps：vLLM 已安装并锁定运行依赖（如 nvidia-cutlass-dsl 版本）
# SPARKINFER_REPO 为空则跳过；REF 为 commit 时必须显式 fetch（--depth 1 浅克隆 +
# detach 无法触达裸 commit），不能退化为默认分支（master 已是 sparkinfer 1.0.1，
# import 名不同会导致 b12x_mxfp4_moe 的 `from b12x...` 导入失败）。
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    set -eux; \
    if [ -n "${SPARKINFER_REPO}" ]; then \
        echo "Refreshing SparkInfer source (cache key: ${SPARKINFER_CACHEBUST})"; \
        rm -rf /tmp/sparkinfer-source; \
        git init -q /tmp/sparkinfer-source; \
        git -C /tmp/sparkinfer-source remote add origin "${SPARKINFER_REPO}"; \
        if [ -n "${SPARKINFER_REF}" ]; then \
            git -C /tmp/sparkinfer-source fetch --depth 1 origin "${SPARKINFER_REF}" >/dev/null 2>&1 \
                || { echo "WARN: fetch ${SPARKINFER_REF} 失败，使用默认分支 HEAD"; \
                     git -C /tmp/sparkinfer-source fetch --depth 1 origin >/dev/null 2>&1; }; \
            git -C /tmp/sparkinfer-source checkout --detach FETCH_HEAD; \
        else \
            git -C /tmp/sparkinfer-source fetch --depth 1 origin >/dev/null 2>&1; \
            git -C /tmp/sparkinfer-source checkout --detach FETCH_HEAD; \
        fi; \
        uv pip install --reinstall --no-deps /tmp/sparkinfer-source; \
        rm -rf /tmp/sparkinfer-source; \
    else \
        echo "SPARKINFER_REPO 为空，跳过 SparkInfer 安装。"; \
    fi

# NCCL 加载顺序修复：torch 自带的 nvidia-nccl 缺 sm_121 cubin，
# 统一软链到我们源码编译（sm_121 gencode）的系统 libnccl
RUN rm -f /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 \
    && ln -sf /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
        /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

# 运行期环境
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH} \
    FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PATH=/workspace/vllm:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

# 无 ENTRYPOINT：启动命令由 Fireworks 配方（compose command）提供
WORKDIR /workspace
