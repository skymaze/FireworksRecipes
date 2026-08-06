#!/usr/bin/env bash
# =============================================================================
# scripts/deploy/deploy_v0260.sh —— 双节点(TP=2) 部署 vLLM v0.26.0 专属镜像
#
# 流程（与 README 真机验证一致）：
#   1) 停止两节点旧服务容器（ds4f-test-rank0/1 与 vllm-ds，若存在）
#   2) worker(.112, rank1 --headless) 先起 → head(.111, rank0) 后起
#   3) 健康检查轮询 head :8000/v1/models
# 用法：
#   ./scripts/deploy/deploy_v0260.sh [IMAGE]     # 含长上下文 JIT 预热（约 10~20 分钟）
#   WARMUP=0 ./scripts/deploy/deploy_v0260.sh     # 跳过预热，尽快对外服务
#   WARMUP_LENS=65536,131072 ./scripts/deploy/deploy_v0260.sh  # 自定义预热长度
#   之后手动跑 bench：见脚本末尾注释。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${1:-fireworks-models/deepseek-v4-flash-0731:0.3.0}"
# 节点地址（环境变量可覆盖，默认占位；正式部署请导出 HEAD_IP/WORKER_IP）
H1="${HEAD_IP:?HEAD_IP not set (head rank0 IP)}"   # head rank0
H2="${WORKER_IP:?WORKER_IP not set (worker rank1 IP)}"   # worker rank1

log(){ printf '\033[1;35m[deploy]\033[0m %s\n' "$*"; }

log "1) 停止旧服务容器（${H1} / ${H2}）"
for h in "$H1" "$H2"; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "spark@$h" \
    'docker rm -f ds4f-test-rank0 ds4f-test-rank1 vllm-ds 2>/dev/null || true; echo "[$h] old containers stopped; running now:"; docker ps --format "{{.Names}}"' || true
done

log "2) 镜像存在于两节点？"
for h in "$H1" "$H2"; do
  ssh -o BatchMode=yes "spark@$h" "docker image inspect ${IMG} >/dev/null 2>&1 && echo '[$h] ${IMG} present' || echo '[$h] ${IMG} MISSING'"
done

log "3) 启动 worker（${H2} rank1 --headless）"
bash scripts/deploy/run_vllm_node.sh "$H2" 1 1 "$IMG"
sleep 5

log "4) 启动 head（${H1} rank0）"
bash scripts/deploy/run_vllm_node.sh "$H1" 0 0 "$IMG"

log "5) 健康检查轮询 http://${H1}:8000/v1/models"
ok=""
for i in $(seq 1 90); do
  r=$(curl -s -m3 "http://${H1}:8000/v1/models" 2>/dev/null | head -c 200)
  if [ -n "$r" ]; then
    ok=1
    echo "   [+] READY at $((i*15))s: $r"
    break
  fi
  # 探活失败继续；若 head 容器退出则报错
  st=$(ssh -o BatchMode=yes "spark@$H1" "docker inspect -f '{{.State.Status}}' vllm-ds 2>/dev/null || echo gone" 2>/dev/null || true)
  [ "$st" != "running" ] && [ "$st" != "restarting" ] && echo "   [!] head vllm-ds status=$st (logs below)" && \
    ssh -o BatchMode=yes "spark@$H1" "docker logs --tail 30 vllm-ds 2>&1 | tail -30" && break
  sleep 15
  printf '   [%s] waiting… (head=%s)\n' "$((i*15))s" "${st:-unknown}"
done
if [ -z "$ok" ]; then
  echo "   [!] 未就绪；查看两节点日志："
  for h in "$H1" "$H2"; do echo "--- $h ---"; ssh -o BatchMode=yes "spark@$h" "docker logs --tail 40 vllm-ds 2>&1 | tail -40" || true; done
  exit 1
fi

log "6) 长上下文 JIT 形状预热（v0.3.0+ 已 bake 缓存于镜像，此步骤仅为补充/兜底）"
# 说明：0.3.0 镜像 bake 了全量 JIT 缓存（/opt/fw/vllm-cache-seed，含 32K..1M 形状），
# 启动 warmup（fw-warmup 补丁）已编译全部变体 → 本步默认执行仅覆盖"未 bake 的新形状"
# 与依赖真实 token 序列触发的路径；也可 WARMUP=0 直接跳过（种子命中时秒级可用）。
if [ "${WARMUP:-1}" = "1" ]; then
  WARMUP_LENS="${WARMUP_LENS:-32768,65536,131072,262144,524288,1048576}"
  log "    预热长度: ${WARMUP_LENS}（跳过: WARMUP=0）"
  python3 scripts/deploy/warmup_longctx.py \
    --base-url "http://${H1}:8000/v1" \
    --model deepseek-v4-flash-0731 \
    --lens "$WARMUP_LENS" || \
    log "    [!] 预热未完成（服务仍可用；新形状首次触达时可能慢/占内存）"
else
  log "    跳过（WARMUP=0）"
fi

log "7) 部署就绪。基准命令（在构建机/控制机执行；<head-ip> 替换为 head 节点地址）："
cat <<'EOF'
  HEAD_URL="http://<head-ip>:8000/v1"
  python3 scripts/bench/bench_decode.py  --base-url "$HEAD_URL" --model deepseek-v4-flash-0731 --concurrency 1,2,4 --max-tokens 128
  python3 scripts/bench/bench_prefill.py --base-url "$HEAD_URL" --model deepseek-v4-flash-0731 --sizes 1024,2048,4096,8192,16384
EOF
log "部署完成。"
