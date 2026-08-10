# 1.5) 烘培 fw-warmup 补丁（NVIDIA/DGX Spark mHC warmup no-op 修复 + sparse MLA 覆盖）
#      根因（2026-08-05 长上下文 OOM）：deepseek_v4_mhc_warmup 的属性检查要求
#      hc_pre/hc_post（仅 AMD/XPU 有），NVIDIA 层没有 → 启动从不预热首层 broadcast
#      变体（mhc_pre_big_fuse_broadcast_with_norm_tilelang）→ 首个长请求推理期
#      JIT 编译 → 峰值内存 OOM。补丁让 NVIDIA 层直接调 tilelang 内核预热全部变体，
#      并把 _SPARSE_MLA_MIXED_WARMUP_TOKENS 16→8192。纯文件级 Python 补丁（AST 校验
#      + 幂等），锚点漂移时拒绝打补丁（构建失败）而非静默打坏。
COPY patches/fw-warmup /opt/fw/patches/fw-warmup
RUN set -eux; \
    cd /opt/fw/patches/fw-warmup && bash run.sh; \
    rm -rf /opt/fw/patches/fw-warmup
