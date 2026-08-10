#!/usr/bin/env python3
"""hybrid-draft-loader —— 自研补丁：InstantTensor（目标） / lazy safetensors（投机草稿）混合加载。

把 DDSPark/MTP 这类「同 checkpoint 内嵌草稿」的投机模型在 `--load-format instanttensor`
场景下的草稿加载改为 lazy safetensors，避免对整份权重做第二次流式加载；目标模型保持不变。

改动点：vllm/model_executor/model_loader/__init__.py 的 get_model()
- 幂等：带 MARKER 标记，重复执行安全
- 校验：AST 形状检查 + compile() 语法检查 + 锚点精确匹配（vLLM 引用漂移时报错而非打坏文件）
- 安全降级：运行时任何异常都返回原加载配置（草稿也走 InstantTensor，仅更慢）
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# fw mod: hybrid-draft-loader v1"

# 目标 vLLM 版本（本仓库 versions.conf 锁定 VLLM_REF）中 get_model 的锚点形态。
# 若升级 vLLM 后此处不再精确匹配，`--check` 会报“锚点匹配数 != 1”，此时需同步更新。
GET_MODEL_ANCHOR = """def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""


HELPER = (
    MARKER
    + """
def _fw_hybrid_draft_load_config(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    load_config: LoadConfig | None,
) -> LoadConfig:
    \"\"\"为单个模型（目标或投机草稿）解析加载配置。

    仅当有效加载方式是 instanttensor 且 model_config 正是投机草稿时生效。
    模式（环境变量 FW_HYBRID_DRAFT_LOADER，兼容 INSTANTTENSOR_DRAFT_LOADER）：
      auto          —— 默认：草稿与目标同源（同 model/revision）时启用 lazy safetensors
      safetensors   —— 所有草稿一律 lazy safetensors
      instanttensor —— 草稿也走 InstantTensor（等价于不启用本补丁）
    运行时异常一律安全降级：返回原加载配置。
    \"\"\"
    import logging
    import os

    from vllm.config import replace

    log = logging.getLogger("fw.hybrid_draft_loader")

    try:
        mode = (
            os.environ.get("FW_HYBRID_DRAFT_LOADER")
            or os.environ.get("INSTANTTENSOR_DRAFT_LOADER")
            or "auto"
        ).strip().lower()
        if mode not in ("auto", "safetensors", "instanttensor"):
            log.warning("FW_HYBRID_DRAFT_LOADER 未知值 %r，按 auto 处理", mode)
            mode = "auto"

        effective = load_config or vllm_config.load_config
        load_format = getattr(effective.load_format, "value", effective.load_format)
        if mode == "instanttensor" or str(load_format).lower() != "instanttensor":
            return effective

        speculative_config = getattr(vllm_config, "speculative_config", None)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        if draft_model_config is None or model_config is not draft_model_config:
            return effective

        if mode == "auto":
            target_model_config = (
                getattr(speculative_config, "target_model_config", None)
                or vllm_config.model_config
            )
            draft_source = (
                getattr(draft_model_config, "model", None),
                getattr(draft_model_config, "revision", None),
            )
            target_source = (
                getattr(target_model_config, "model", None),
                getattr(target_model_config, "revision", None),
            )
            if draft_source != target_source:
                return effective

        log.info(
            "Hybrid draft loading: speculative draft -> lazy safetensors, "
            "target keeps InstantTensor (mode=%s).",
            mode,
        )
        return replace(
            effective,
            load_format="safetensors",
            safetensors_load_strategy="lazy",
        )
    except Exception as exc:  # noqa: BLE001 —— 补丁必须安全降级，不可阻断推理
        log.warning("Hybrid draft loading disabled: %s", exc)
        return load_config or getattr(vllm_config, "load_config")


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    if model_config is None:
        model_config = vllm_config.model_config
    resolved_load_config = _fw_hybrid_draft_load_config(
        vllm_config, model_config, load_config
    )
    loader = get_model_loader(resolved_load_config)
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""
)


def validate_shape(text: str, *, patched: bool) -> None:
    """校验模块形状：收集「定义的函数」+「导入的名字」，要求关键符号俱在。"""
    tree = ast.parse(text)

    def names(node):
        for child in ast.iter_child_nodes(node):
            yield from names(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child.name
            elif isinstance(child, ast.Import):
                for a in child.names:
                    yield a.asname or a.name.split(".")[0]
            elif isinstance(child, ast.ImportFrom):
                for a in child.names:
                    yield a.asname or a.name

    available = set(names(ast.Module(body=tree.body, type_ignores=[])))
    # 顶层定义的函数（not 导入），保证我们注入的 helper 是真正的 def
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {"get_model", "get_model_loader"}
    if patched:
        required.add("_fw_hybrid_draft_load_config")
    missing = sorted(required - available)
    if missing:
        raise ValueError("模型加载模块形状不符，缺少: " + ", ".join(missing))
    if patched and "_fw_hybrid_draft_load_config" not in defined:
        raise ValueError("helper 必须是模块内实际定义的函数")


def patched_text(text: str) -> str:
    validate_shape(text, patched=MARKER in text)

    if MARKER in text:
        if text.count(MARKER) != 1:
            raise ValueError("hybrid-draft-loader 标记出现多次")
        if "_fw_hybrid_draft_load_config(" not in text:
            raise ValueError("已有标记但 get_model 未使用补丁 helper")
        compile(text, "<patched model_loader/__init__.py>", "exec")
        return text

    count = text.count(GET_MODEL_ANCHOR)
    if count != 1:
        raise ValueError(
            f"锚点 get_model 匹配数 {count}（需恰好 1 个）——当前 vLLM 版本与 "
            "本补丁不匹配，请对照 versions.conf 锁定版本的源码更新 GET_MODEL_ANCHOR"
        )

    patched = text.replace(GET_MODEL_ANCHOR, HELPER, 1)
    validate_shape(patched, patched=True)
    compile(patched, "<patched model_loader/__init__.py>", "exec")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅校验兼容性，不落盘",
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"[hybrid-draft-loader ERROR] 目标不存在: {args.target}", file=sys.stderr)
        return 1

    original = args.target.read_text()
    try:
        patched = patched_text(original)
    except (SyntaxError, ValueError) as exc:
        print(
            f"[hybrid-draft-loader ERROR] 拒绝补丁 {args.target}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        state = "已补丁" if patched == original else "兼容（未补丁）"
        print(f"[hybrid-draft-loader] {args.target}: {state}。")
        return 0

    if patched == original:
        print("[hybrid-draft-loader] 已存在补丁，跳过。")
        return 0

    temporary = args.target.with_suffix(args.target.suffix + ".fw-hybrid-draft.tmp")
    temporary.write_text(patched)
    temporary.replace(args.target)
    print(f"[hybrid-draft-loader] 已补丁 {args.target}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
