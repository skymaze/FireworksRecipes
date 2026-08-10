# 1) 烘培 vLLM 混合加载补丁
#    --load-format instanttensor 时，同 checkpoint 内的 dspark 投机草稿改走
#    lazy safetensors（避免对整份权重做第二次 InstantTensor 流式拷贝）。
#    run.sh 为纯文件级 Python 补丁：AST 校验 + 幂等，构建期安全。
#    锚点已对齐 v0.26.0 的 get_model（签名逐字节一致），--check 通过即可复用。
COPY patches/hybrid-draft-loader /opt/fw/patches/hybrid-draft-loader
RUN set -eux; \
    cd /opt/fw/patches/hybrid-draft-loader && bash run.sh; \
    rm -rf /opt/fw/patches/hybrid-draft-loader
