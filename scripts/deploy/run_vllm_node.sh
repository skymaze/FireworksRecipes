#!/usr/bin/env bash
# =============================================================================
# scripts/deploy/run_vllm_node.sh —— 在单个 DGX Spark 节点启动 vLLM 容器
#
# 用法（由 deploy_v0260.sh 调用，或手动）：
#   ./scripts/deploy/run_vllm_node.sh <ip> <rank> <headless:0|1> [image]
#
# 与 recipes/<id>/fireworks.recipe.json 的 v0.26.0 命令保持一致：
#   kv=fp8_ds_mla / --moe-backend flashinfer_b12x / dspark MTP / instanttensor /
#   mp 分布式 / RoCE 4×100G env。容器直接以 `vllm serve` 作为 CMD（不经 bash -lc），
#   规避上文 README 记录的 JSON 参数引号截断问题。
# =============================================================================
set -euo pipefail

NODE_IP="${1:?<ip>}"
RANK="${2:?<rank>}"
HEADLESS="${3:-0}"
IMG="${4:-fireworks-models/deepseek-v4-flash-0731:0.3.0}"

# 分布式 master 地址：默认占位；正式部署请导出 MASTER_ADDR（head 节点 IP）
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR not set}"
MASTER_PORT="${MASTER_PORT:-25000}"
PORT="${VLLM_PORT:-8000}"
MODEL="${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
SERVED="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
MAX_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_SEQS="${MAX_NUM_SEQS:-6}"
MAX_BATCH="${MAX_NUM_BATCHED_TOKENS:-8192}"
UTIL="${GPU_MEMORY_UTILIZATION:-0.88}"
BLOCK="${BLOCK_SIZE:-256}"
CAPTURE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-36}"
SPEC_N="${NUM_SPECULATIVE_TOKENS:-5}"
KV="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
HCA="${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}"
GID="${NCCL_IB_GID_INDEX:-3}"

declare -a ARGS=(vllm serve "$MODEL"
  --served-model-name "$SERVED"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 2
  --kv-cache-dtype "$KV"
  --block-size "$BLOCK"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$MAX_BATCH"
  --max-cudagraph-capture-size "$CAPTURE"
  --gpu-memory-utilization "$UTIL"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --enable-flashinfer-autotune
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --load-format instanttensor
)
# MoE 后端：默认 auto（v0.26.0 原生优先级 → DeepGemmFP4Experts，真机验证 prefill
# 2200+ tok/s）；可设 flashinfer_b12x（decode 更优但 v0.26.0 集成下 prefill ~88 tok/s
# 不可用），或其它受支持值。
MOE_BACKEND="${MOE_BACKEND:-auto}"
ARGS+=(--moe-backend "${MOE_BACKEND}")
# 对照实验开关：SPEC_OFF=1 时不传 --speculative-config（隔离 dspark 对 prefill 的影响）
if [ "${SPEC_OFF:-0}" != "1" ]; then
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_N,\"draft_sample_method\":\"probabilistic\"}")
fi
ARGS+=(
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
)
[ "$HEADLESS" = "1" ] && ARGS+=(--headless)

# 数组 → 经 ssh 传送到节点执行。为免 JSON 在多层转义中损坏，这里把整条
# docker run 命令序列化到节点上的临时脚本再执行。
CMD_FILE="/tmp/fw_vllm_run.sh"
cat > /tmp/fw_vllm_run.sh <<EOF
set -euxo pipefail
docker rm -f vllm-ds 2>/dev/null || true
# JIT 缓存种子灌入（0.3.0+ 镜像 bake 在 /opt/fw/vllm-cache-seed）：
# 幂等——cp -an（archive + no-clobber）只补缺失文件，宿主已有缓存不被覆盖。
if docker run --rm --entrypoint sh "${IMG}" -c 'test -d /opt/fw/vllm-cache-seed' 2>/dev/null; then
  docker run --rm -v /home/spark/.cache/vllm-cache:/cache/vllm-cache \\
    --entrypoint sh "${IMG}" -c \\
    'mkdir -p /cache/vllm-cache && cp -an /opt/fw/vllm-cache-seed/. /cache/vllm-cache/'
  echo "[deploy] vllm-cache seed seeded from image"
else
  echo "[deploy] no seed in image (pre-0.3.0); skip"
fi
docker run -d --name vllm-ds \\
  --gpus all --network host --ipc host --shm-size 64gb \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  --device /dev/infiniband:/dev/infiniband \\
  -v /home/spark/.cache/huggingface:/cache/huggingface \\
  -v /home/spark/.cache/vllm-cache:/cache/vllm-cache \\
  -v /home/spark/.cache/dspark-tmp:/tmp \\
  -e HF_HOME=/cache/huggingface \\
  -e HF_HUB_OFFLINE=1 \\
  -e HF_HUB_DISABLE_XET=1 \\
  -e VLLM_CACHE_ROOT=/cache/vllm-cache \\
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \\
  -e TRITON_CACHE_DIR=/cache/vllm-cache/triton \\
  -e TILELANG_CACHE_DIR=/cache/vllm-cache/tilelang \\
  -e TILELANG_CLEANUP_TEMP_FILES=1 \\
  -e DG_JIT_CACHE_DIR=/tmp/deep-gemm \\
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  -e VLLM_HOST_IP=${NODE_IP} \\
  -e NCCL_IB_HCA=${HCA} \\
  -e NCCL_IB_GID_INDEX=${GID} \\
  -e NCCL_IB_GID_AUTO=1 \\
  -e NCCL_IB_ADDR_FAMILY=AF_INET \\
  -e NCCL_IB_ROCE_VERSION_NUM=2 \\
  -e NCCL_IB_DISABLE=0 \\
  -e NCCL_CROSS_NIC=1 \\
  -e NCCL_NVLS_ENABLE=0 \\
  -e NCCL_CUMEM_ENABLE=0 \\
  -e NCCL_IGNORE_CPU_AFFINITY=1 \\
  -e NCCL_DEBUG=WARN \\
EOF
# 对照实验开关：FW_B12X_OFF=1 时覆盖镜像内 VLLM_USE_B12X_MOE=1（MoE 走原生路径）
if [ "${FW_B12X_OFF:-0}" = "1" ]; then
  cat >> /tmp/fw_vllm_run.sh <<'EOF2'
  -e VLLM_USE_B12X_MOE=0 \
EOF2
fi
# 对照实验开关：BREAKABLE_CUDAGRAPH=0 → regular CUDA graph（ref1 配置；v0.26.0 默认开 breakable）
if [ -n "${BREAKABLE_CUDAGRAPH:-}" ]; then
  cat >> /tmp/fw_vllm_run.sh <<EOF2
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=${BREAKABLE_CUDAGRAPH} \\
EOF2
fi
cat >> /tmp/fw_vllm_run.sh <<EOF
  "${IMG}" \\
EOF
# 把 vllm serve 参数逐个以 shell 安全形式追加进脚本（引号保留于节点侧）
{
  for a in "${ARGS[@]}"; do
    printf '  %q \\\n' "$a"
  done
  printf '\n'
} >> /tmp/fw_vllm_run.sh

echo "[deploy] sending run script to ${NODE_IP} (rank=${RANK} headless=${HEADLESS})"
scp -q -o BatchMode=yes /tmp/fw_vllm_run.sh "spark@${NODE_IP}:/tmp/"
ssh -o BatchMode=yes "spark@${NODE_IP}" "bash /tmp/fw_vllm_run.sh"
echo "[deploy] started vllm-ds on ${NODE_IP} (rank=${RANK})"
