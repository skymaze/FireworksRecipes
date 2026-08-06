#!/usr/bin/env python3
"""fw-warmup —— 自研补丁：修复 DGX Spark (NVIDIA) 路径 mHC warmup no-op + sparse MLA warmup 覆盖不足。

背景（2026-08-05 长上下文 OOM 崩溃根因之一）：
- ``deepseek_v4_mhc_warmup`` 的 ``_find_first_mhc_layer`` 要求层上有
  ``hc_pre``/``hc_post`` 属性（仅 AMD/XPU 模型有）；NVIDIA ``DeepseekV4DecoderLayer``
  只有 ``hc_attn_fn``/``hc_ffn_fn``/``hc_attn_fn_broadcast`` 等 → 启动 warmup 是 no-op，
  首层 broadcast 变体（``mhc_pre_big_fuse_broadcast_with_norm_tilelang``）从不预热，
  首个长请求在推理路径内触发 TileLang JIT 编译（峰值内存 → OOM）。
- ``_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16``：sparse MLA mixed prefill+decode autotune
  只覆盖 16 token，远小于 ``max_num_batched_tokens``。

补丁内容：
  A) ``vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py``：
     ``_SPARSE_MLA_MIXED_WARMUP_TOKENS`` 16 → 8192
  B) ``vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py``：
     - ``_find_first_mhc_layer``：属性检查兼容 NVIDIA 层（hc_attn_fn/hc_ffn_fn/
       hc_attn_fn_broadcast 等，不再要求 hc_pre/hc_post）
     - ``_warmup_layer_mhc``：新增 NVIDIA 分支 —— 直接调 ``mhc_pre_broadcast_tilelang``
       (2D 首层 broadcast) / ``mhc_pre_tilelang`` (3D) / ``mhc_fused_post_pre_tilelang``
       (attn+ffn)，token sizes 用现有候选集 {1..16384}+cudagraph sizes
     - ``_warmup_hc_head``：新增 NVIDIA 分支 —— 直接调 ``hc_head_fused_kernel_tilelang``

幂等（MARKER 标记）/ AST 形状校验 / 安全降级：与 hybrid-draft-loader 同风格。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# fw mod: fw-warmup v1"

# ---- patch A: flashinfer_sparse_mla_warmup.py -------------------------------------
SPARSE_ANCHOR = "_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16"
SPARSE_REPLACEMENT = MARKER + "\n_SPARSE_MLA_MIXED_WARMUP_TOKENS = 8192"


def patch_sparse_mla(text: str) -> str:
    if MARKER in text:
        if text.count(MARKER) != 1:
            raise ValueError("fw-warmup 标记出现多次（flashinfer_sparse_mla_warmup）")
        compile(text, "<patched flashinfer_sparse_mla_warmup.py>", "exec")
        return text
    if text.count(SPARSE_ANCHOR) != 1:
        raise ValueError(
            f"锚点 {SPARSE_ANCHOR!r} 匹配数 != 1（当前 vLLM 版本与本补丁不匹配）"
        )
    patched = text.replace(SPARSE_ANCHOR, SPARSE_REPLACEMENT, 1)
    compile(patched, "<patched flashinfer_sparse_mla_warmup.py>", "exec")
    return patched


# ---- patch B: deepseek_v4_mhc_warmup.py -------------------------------------------
# 1) _find_first_mhc_layer：NVIDIA 兼容的属性检查
FIND_LAYER_OLD = """def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None"""

FIND_LAYER_NEW = MARKER + "\n" + """def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        # NVIDIA 路径（DGX Spark）：层有 hc_attn_fn/hc_ffn_fn/hc_attn_fn_broadcast，
        # 但没有 AMD/XPU 才有的 hc_pre/hc_post 属性。兼容两者。
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ) and (
            hasattr(module, "hc_pre")
            or hasattr(module, "hc_attn_fn_broadcast")
        ):
            return module
    return None"""

# 2) _warmup_layer_mhc：NVIDIA 分支（直接调 tilelang 内核）
WARMUP_LAYER_OLD = """def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)"""

WARMUP_LAYER_NEW = MARKER + "\n" + """def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device

    if not hasattr(layer, "hc_pre"):
        # NVIDIA 路径（DGX Spark）：直接调 tilelang 内核，编译全部变体。
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_fused_post_pre_tilelang,
            mhc_pre_broadcast_tilelang,
            mhc_pre_tilelang,
        )

        attn_norm_weight = layer.attn_norm.weight.data
        attn_norm_eps = layer.attn_norm.variance_epsilon
        ffn_norm_weight = layer.ffn_norm.weight.data
        ffn_norm_eps = layer.ffn_norm.variance_epsilon
        x3d = torch.zeros(
            max_tokens,
            hc_mult,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        x2d = torch.zeros(
            max_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        for size in token_sizes:
            # 首层 2D broadcast 变体（mhc_pre_big_fuse_broadcast_with_norm_*）
            # hc_attn_fn_broadcast 由 finalize_mhc_broadcast_weights 填充；
            # 若尚未 finalize 则跳过（首次真实请求会触发，但概率极低）。
            if layer.hc_attn_fn_broadcast is not None:
                mhc_pre_broadcast_tilelang(
                    x2d[:size],
                    layer.hc_attn_fn,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    layer.rms_norm_eps,
                    layer.hc_eps,
                    layer.hc_eps,
                    layer.hc_post_alpha,
                    layer.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                    fn_broadcast=layer.hc_attn_fn_broadcast,
                )
            # 3D pre 变体（mhc_pre_big_fuse_*）
            mhc_pre_tilelang(
                x3d[:size],
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
            # fused post-pre 变体（mhc_fused_post_pre_*，attn 与 ffn）
            post_mix = torch.zeros(
                size, hc_mult, dtype=torch.float32, device=device
            )
            comb_mix = torch.zeros(
                size, hc_mult, hc_mult, dtype=torch.float32, device=device
            )
            for fn, scale, base, nw, ne in (
                (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                 attn_norm_weight, attn_norm_eps),
                (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                 ffn_norm_weight, ffn_norm_eps),
            ):
                mhc_fused_post_pre_tilelang(
                    x2d[:size],
                    x3d[:size],
                    post_mix,
                    comb_mix,
                    fn,
                    scale,
                    base,
                    layer.rms_norm_eps,
                    layer.hc_eps,
                    layer.hc_eps,
                    layer.hc_post_alpha,
                    layer.hc_sinkhorn_iters,
                    n_splits=1,
                    tile_n=1,
                    norm_weight=nw,
                    norm_eps=ne,
                )
        return

    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)"""

# 3) _warmup_hc_head：NVIDIA 分支
HC_HEAD_OLD = """    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )"""

HC_HEAD_NEW = MARKER + "\n" + """    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is not None:
        for size in token_sizes:
            hc_head_op(
                hidden_states[:size],
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.rms_norm_eps,
                model.hc_eps,
            )
        return

    # NVIDIA 路径（DGX Spark）：hc_head 是自由函数 hc_head_fused_kernel_tilelang。
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
    )

    for size in token_sizes:
        hc_head_fused_kernel_tilelang(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )"""


def _required_symbols() -> set[str]:
    return {
        "deepseek_v4_mhc_warmup",
        "_find_first_mhc_layer",
        "_find_deepseek_v4_model",
        "_warmup_layer_mhc",
        "_warmup_hc_head",
        "_select_mhc_warmup_token_sizes",
    }


def _defined_symbols(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def patch_mhc_warmup(text: str) -> str:
    tree = ast.parse(text)
    defined = _defined_symbols(tree)
    missing = _required_symbols() - defined
    if missing:
        raise ValueError("deepseek_v4_mhc_warmup.py 形状不符，缺少: " + ", ".join(missing))

    if MARKER in text:
        if text.count(MARKER) != 3:
            raise ValueError("fw-warmup 标记出现次数 != 3（deepseek_v4_mhc_warmup）")
        compile(text, "<patched deepseek_v4_mhc_warmup.py>", "exec")
        return text

    for anchor, replacement, what in (
        (FIND_LAYER_OLD, FIND_LAYER_NEW, "_find_first_mhc_layer"),
        (WARMUP_LAYER_OLD, WARMUP_LAYER_NEW, "_warmup_layer_mhc"),
        (HC_HEAD_OLD, HC_HEAD_NEW, "_warmup_hc_head"),
    ):
        if text.count(anchor) != 1:
            raise ValueError(
                f"锚点 {what} 匹配数 != 1（当前 vLLM 版本与本补丁不匹配）"
            )
        text = text.replace(anchor, replacement, 1)

    compile(text, "<patched deepseek_v4_mhc_warmup.py>", "exec")
    return text


def validate_shape(path: Path, patched: bool) -> None:
    tree = ast.parse(path.read_text())
    defined = _defined_symbols(tree)
    missing = _required_symbols() - defined
    if missing:
        raise ValueError("模型加载模块形状不符，缺少: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("warmup_dir", type=Path,
                        help="vllm/model_executor/warmup 目录")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sparse = args.warmup_dir / "flashinfer_sparse_mla_warmup.py"
    mhc = args.warmup_dir / "deepseek_v4_mhc_warmup.py"
    if not sparse.is_file() or not mhc.is_file():
        print(f"[fw-warmup ERROR] warmup 文件缺失: {sparse} / {mhc}",
              file=sys.stderr)
        return 1

    results = []
    for path, patcher in ((sparse, patch_sparse_mla), (mhc, patch_mhc_warmup)):
        original = path.read_text()
        try:
            patched = patcher(original)
        except (SyntaxError, ValueError) as exc:
            print(f"[fw-warmup ERROR] 拒绝补丁 {path}: {exc}", file=sys.stderr)
            return 1

        state = "已补丁" if patched == original else "兼容（未补丁）"
        if args.check:
            print(f"[fw-warmup] {path.name}: {state}。")
            continue
        if patched == original:
            print(f"[fw-warmup] {path.name}: 已存在补丁，跳过。")
            continue
        tmp = path.with_suffix(path.suffix + ".fw-warmup.tmp")
        tmp.write_text(patched)
        tmp.replace(path)
        results.append(path.name)
        print(f"[fw-warmup] 已补丁 {path.name}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
