#!/usr/bin/env bash
# =============================================================================
# render-model-dockerfile.sh —— 从模板渲染 recipes/<id>/Dockerfile.model
#
# 用法：
#   ./scripts/render-model-dockerfile.sh <model>        # 生成并写入
#   ./scripts/render-model-dockerfile.sh --check <model> # 校验生成==已提交（CI）
#
# 数据来源 = recipes/<id>/build.conf（与 scripts/build.sh 同源 source）：
#   MODEL_PATCH_DIRS : 烘培进镜像的补丁目录列表（templates/patches/<dir>.inc）
#   MODEL_ID         : HF 模型 id（ENV FW_MODEL_ID / LABEL fw.model）
#   MODEL_LABEL_TITLE: 镜像 OCI title（缺省 "Fireworks ${MODEL_NAME}"）
#   MODEL_LABEL_DESC : 镜像 OCI description（缺省 "Dedicated serving image for ${MODEL_ID}..."）
#   MODEL_EXTRA_ENV  : 追加到通用调优 ENV 之后的模型专属 ENV（可选）
#
# 渲染产物要求与已提交文件逐字节一致；漂移时 --check 退出非 0（防止手改/漏改）。
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMPL="$REPO/templates/Dockerfile.model"

usage() { echo "用法: $0 [--check] <model>" >&2; exit 2; }

CHECK=false
MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=true; shift ;;
    -*) usage ;;
    *) MODEL="$1"; shift ;;
  esac
done
[ -n "$MODEL" ] || usage

MODEL_DIR="$REPO/recipes/$MODEL"
BC="$MODEL_DIR/build.conf"
[ -f "$BC" ] || { echo "错误: 缺少 $BC" >&2; exit 2; }

# 与 build.sh 一致：source build.conf（纯赋值，构建主链已有先例）
MODEL_PATCH_DIRS=()
MODEL_EXTRA_ENV=()
set -a
# shellcheck disable=SC1090
source "$BC"
set +a

: "${MODEL_NAME:?build.conf 需定义 MODEL_NAME}"
: "${MODEL_ID:?build.conf 需定义 MODEL_ID}"
MODEL_TITLE="${MODEL_TITLE:-$MODEL_NAME}"
MODEL_LABEL_TITLE="${MODEL_LABEL_TITLE:-Fireworks ${MODEL_TITLE}}"
MODEL_LABEL_DESC="${MODEL_LABEL_DESC:-Dedicated serving image for ${MODEL_ID} (mainline vLLM v0.26.0 + GB10 overlay)}"

# 拼接补丁块（块与块之间一个空行；$(cat) 会剥尾换行，故以 \n\n 连接）
PATCHES=""
sep=""
for d in "${MODEL_PATCH_DIRS[@]}"; do
  inc="$REPO/templates/patches/$d.inc"
  [ -f "$inc" ] || { echo "错误: 缺少补丁片段 $inc（build.conf MODEL_PATCH_DIRS 需对应 templates/patches/）" >&2; exit 2; }
  PATCHES+="${sep}$(cat "$inc")"
  sep=$'\n\n'
done

# 模型专属 ENV：非空时在通用 ENV 后另起一个 ENV 续行块
EXTRA_ENV=""
if [ "${#MODEL_EXTRA_ENV[@]}" -gt 0 ]; then
  EXTRA_ENV+=$'\nENV '
  for line in "${MODEL_EXTRA_ENV[@]}"; do
    EXTRA_ENV+="${line} \\"
    EXTRA_ENV+=$'\n'
  done
  EXTRA_ENV+=" "
fi

# python 做安全替换（令牌无注入风险；多行变量交给 -c 传入环境）
render() {
  PATCHES="$PATCHES" EXTRA_ENV="$EXTRA_ENV" \
  MODEL_TITLE="$MODEL_TITLE" MODEL_ID="$MODEL_ID" \
  MODEL_LABEL_TITLE="$MODEL_LABEL_TITLE" MODEL_LABEL_DESC="$MODEL_LABEL_DESC" \
  python3 - "$TMPL" <<'PY'
import os, sys
t = open(sys.argv[1], encoding="utf-8").read()
t = t.replace("__TITLE__", os.environ["MODEL_TITLE"])
t = t.replace("__FW_MODEL_ID__", os.environ["MODEL_ID"])
t = t.replace("__LABEL_TITLE__", os.environ["MODEL_LABEL_TITLE"])
t = t.replace("__LABEL_DESC__", os.environ["MODEL_LABEL_DESC"])
t = t.replace("__PATCHES__", os.environ["PATCHES"])
t = t.replace("__EXTRA_ENV__", os.environ["EXTRA_ENV"])
if not t.endswith("\n"):
    t += "\n"
sys.stdout.write(t)
PY
}

OUT="$MODEL_DIR/Dockerfile.model"
# $(...) 会剥掉尾部换行，用 += 在变量上补回一个（Dockerfile 以单个换行结尾）
RENDERED="$(render)"
RENDERED+=$'\n'

if [ "$CHECK" = true ]; then
  if [ ! -f "$OUT" ]; then echo "错误: $OUT 不存在（--check）" >&2; exit 1; fi
  if ! printf '%s' "$RENDERED" | diff -q - "$OUT" >/dev/null; then
    echo "漂移: $OUT 与模板渲染不一致，请运行 $0 $MODEL 重新生成" >&2
    diff <(printf '%s' "$RENDERED") "$OUT" | head -40 >&2 || true
    exit 1
  fi
  echo "OK: $OUT 与模板一致"
else
  printf '%s' "$RENDERED" > "$OUT"
  echo "已生成 $OUT（模板: $TMPL）"
fi
