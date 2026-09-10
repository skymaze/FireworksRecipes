# Derived serving image: spark-vllm-b12x + InstantTensor hybrid speculative-draft loader
#
# Base is pinned to the nightly the vendored patch was verified against. The mod
# (mods/instanttensor-hybrid-draft-loader, MIT, from eugr/spark-vllm-docker@60dee5b) is a
# runtime workaround that is compatible with one specific vLLM get_model() shape; the
# patcher fails the build hard if the base image's model_loader no longer matches, so a
# nightly bump without re-verifying will not silently produce a broken image.
#
# Maintenance when upstream spark-vllm-docker advances:
#   1. re-pin FROM to the new nightly (or digest) and rebuild;
#   2. if the patch no longer applies, update mods/.../patch_model_loader.py to the new
#      upstream revision (the GET_MODEL_ANCHOR must match the image's model_loader) and
#      bump the baked image tag;
#   3. remove the bake entirely once vLLM supports a per-draft load format
#      (see mods/README.md "Scope and limitations").
FROM eugr/spark-vllm-b12x@sha256:4f5482a24a3d966a4f1a942102232568cad438431cbdc2d17eeabcac33ede5a2

LABEL org.opencontainers.image.title="spark-vllm-b12x + instanttensor-hybrid-draft-loader" \
      org.opencontainers.image.description="DGX Spark vLLM b12x distribution image with the InstantTensor hybrid speculative-draft loader baked in (FireworksRecipes bake)" \
      org.opencontainers.image.source="https://github.com/eugr/spark-vllm-docker" \
      org.opencontainers.image.licenses="MIT" \
      com.fireworks.bake.base="eugr/spark-vllm-b12x:nightly-20260909" \
      com.fireworks.bake.upstream_commit="60dee5b47ab14201381d486c6e994b5cf0e648a1"

# Vendored mod sources (MIT) from eugr/spark-vllm-docker; see mods/README.md.
COPY mods/instanttensor-hybrid-draft-loader /opt/spark-vllm-mods/instanttensor-hybrid-draft-loader

# Bake the patch at build time: the draft loaded for speculative decoding switches to
# lazy safetensors (auto mode, same checkpoint path) while the target keeps InstantTensor.
# VLLM_SITE_PACKAGES is detected from the image's own vllm package layout.
RUN PYROOT="$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")" \
    && INSTANTTENSOR_DRAFT_LOADER=auto VLLM_SITE_PACKAGES="${PYROOT}" \
       bash /opt/spark-vllm-mods/instanttensor-hybrid-draft-loader/run.sh
