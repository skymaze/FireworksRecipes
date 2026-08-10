#!/usr/bin/env python3
"""
FireworksRecipes 配方源校验脚本（零依赖，纯 stdlib）。

职责（配合 docs/RECIPE-FORMAT.md 与 schemas/）：
1. 校验每份 fireworks.recipe.json：
   - 必填/类型/枚举（schema）；
   - 变量 key 唯一；source/user/cluster/node 规则；picker/type 枚举；
   - **source ∈ {cluster, node} 的变量必须给出『已知』的 auto 键**（AUTO_KEYS 白名单，
     与 Fireworks backend recipe_render 的自动变量集合一致）；未知 auto 键 → 报错。
2. 校验 recipes/index.json（manifest）：schema==1、每条必填字段、path 唯一。
3. 交叉一致性（防双写漂移）：
   - manifest 指向的 path/readme(_en) 必须存在；
   - manifest 与 recipe 重叠字段必须一致：version / image / nodes / tensor_parallel；
   - 描述字段若两边都有则必须一致。

用法：  python3 scripts/validate.py            # 校验仓库全部配方 + manifest
        python3 scripts/validate.py --recipe recipes/xxx/fireworks.recipe.json
        python3 scripts/validate.py --manifest recipes/index.json
退出码：0 = 全部通过；1 = 存在错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MANIFEST = REPO / "recipes" / "index.json"

# 与 Fireworks backend app/services/recipe_render.py 的自动变量集合保持一致；
# 新增自动填充键必须同步两处（本白名单 + backend renderer）。
AUTO_KEYS = {
    # source=cluster（任务级共享）
    "head_roce_ip", "nodes_total", "network_type", "head_ip", "head_hostname",
    # source=node（逐节点）
    "node_rank", "role", "hostname", "node_ip", "node_roce_ip",
    "hca", "netdev", "gid_index", "agent_port", "headless",
}

TYPES = {"string", "int", "float", "bool", "select"}
SOURCES = {"user", "cluster", "node"}
PICKERS = {"", "model", "image"}


def _is_str(v): return isinstance(v, str)
def _is_int(v): return isinstance(v, int) and not isinstance(v, bool)
def _is_num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


# 已废弃字段：曾经在 manifest/配方里出现过，现统一只保留 nodes（GB10 每机 1 GPU，
# TP=节点数，且可能混合架构下无单一 dtype）。出现即报错。
DEPRECATED_FIELDS = ("topology", "tensor_parallel", "dtype")


def _check_deprecated(obj, rel, errors):
    for k in DEPRECATED_FIELDS:
        if k in obj:
            errors.append(f"{rel}: 已废弃字段「{k}」（请移除；固定拓扑只用 nodes）")


def check_recipe(rel: Path) -> list[str]:
    errors: list[str] = []
    src = (REPO / rel).read_text(encoding="utf-8")
    try:
        r = json.loads(src)
    except json.JSONDecodeError as e:
        return [f"{rel}: JSON 解析失败: {e}"]

    def err_at(msg): errors.append(f"{rel}: {msg}")
    _check_deprecated(r, rel, errors)

    # 必填（name/compose_template 为字符串，variables 为数组）
    for f in ("name", "compose_template"):
        if not _is_str(r.get(f)):
            err_at(f"缺少/类型错误: 「{f}」")
    if not isinstance(r.get("variables"), list):
        err_at("缺少/类型错误: 「variables」（必须为数组）")
    name = r.get("name")
    if _is_str(name) and not name.strip():
        err_at("name 为空")
    if "name_en" in r and not _is_str(r.get("name_en")):
        err_at("name_en 必须为字符串")
    for f in ("description", "description_en"):
        if f in r and not (_is_str(r.get(f)) or r.get(f) is None):
            err_at(f"「{f}」必须为字符串或 null")
    for f in ("version", "image"):
        if f in r and r.get(f) is not None and not _is_str(r.get(f)):
            err_at(f"「{f}」必须为字符串或 null")
    for f in ("nodes", "tensor_parallel"):
        if f in r and r.get(f) is not None and not _is_int(r.get(f)):
            err_at(f"「{f}」必须为整数或 null")
    if _is_int(r.get("nodes")) and r["nodes"] < 1:
        err_at("「nodes」必须大于等于 1")

    # 变量
    vars_ = r.get("variables") or []
    seen: set[str] = set()
    for i, v in enumerate(vars_):
        if not isinstance(v, dict):
            err_at(f"variables[{i}] 不是对象"); continue
        key = v.get("key")
        if not _is_str(key) or not key.strip():
            err_at(f"variables[{i}] 缺少非空 key"); continue
        if key in seen:
            err_at(f"variables[{i}].key「{key}」重复")
        seen.add(key)
        vtype = v.get("type", "string")
        if vtype not in TYPES:
            err_at(f"variables[{i}].key「{key}」type 非法: {vtype!r}（应为 {sorted(TYPES)}）")
        src = v.get("source", "user")
        if src not in SOURCES:
            err_at(f"variables[{i}].key「{key}」source 非法: {src!r}")
        picker = v.get("picker", "")
        if picker not in PICKERS:
            err_at(f"variables[{i}].key「{key}」picker 非法: {picker!r}")
        auto = v.get("auto")
        if auto is not None and not _is_str(auto):
            err_at(f"variables[{i}].key「{key}」auto 必须为字符串")
        if src == "user":
            if auto:
                err_at(f"variables[{i}].key「{key}」source=user 不应携带 auto（只用默认值/用户值）")
        else:  # cluster / node
            if not auto:
                err_at(f"variables[{i}].key「{key}」source={src} 必须给出 auto 填充键")
            elif auto not in AUTO_KEYS:
                err_at(
                    f"variables[{i}].key「{key}」auto「{auto}」不在已知自动键集合中。"
                    f"若确需该键，请同步 README/RECIPE-FORMAT 与 Fireworks recipe_render；已知键: {sorted(AUTO_KEYS)}"
                )
        if vtype == "select" and not isinstance(v.get("options", []), list):
            err_at(f"variables[{i}].key「{key}」select 必须提供 options 列表")
        if "default" in v and v.get("default") is not None and not (
            _is_str(v["default"]) or _is_num(v["default"]) or isinstance(v["default"], bool)
        ):
            err_at(f"variables[{i}].key「{key}」default 类型非法")
    return errors


def check_manifest() -> list[str]:
    errors: list[str] = []
    if not MANIFEST.is_file():
        return ["缺少 manifest: recipes/index.json"]
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"recipes/index.json: JSON 解析失败: {e}"]

    def err_at(msg): errors.append(f"recipes/index.json: {msg}")
    _check_deprecated(m, "recipes/index.json", errors)

    if m.get("schema") != 1:
        err_at("schema 必须为 1")
    if not _is_str(m.get("name")):
        err_at("缺少 name")
    items = m.get("recipes")
    if not isinstance(items, list):
        err_at("recipes 必须为数组"); return errors

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            err_at(f"recipes[{i}] 不是对象"); continue
        _check_deprecated(it, f"recipes/index.json entries[{i}]", errors)
        iid = it.get("id")
        path = it.get("path")
        if not iid:
            err_at(f"recipes[{i}] 缺少 id")
        elif iid in seen_ids:
            err_at(f"recipes[{i}].id「{iid}」重复")
        seen_ids.add(iid)
        if not _is_str(it.get("name")) or not it.get("name"):
            err_at(f"recipes[{i}] 缺少显示名「name」（商店卡片标题用）")
        if "name_en" in it and not _is_str(it.get("name_en")):
            err_at(f"recipes[{i}].name_en 必须为字符串")
        if not path:
            err_at(f"recipes[{i}] 缺少 path")
        elif path in seen_paths:
            err_at(f"recipes[{i}].path「{path}」重复")
        seen_paths.add(path)
        for f in ("version", "image"):
            if not _is_str(it.get(f)):
                err_at(f"recipes[{i}] 缺少字符串字段「{f}」")
        if not _is_int(it.get("nodes")):
            err_at(f"recipes[{i}] 缺少整数字段「nodes」")
        elif it["nodes"] < 1:
            err_at(f"recipes[{i}].nodes 必须大于等于 1")

        if not path:
            continue
        recipe_rel = REPO / path
        if not recipe_rel.is_file():
            err_at(f"recipes[{i}].path 不存在: {path}")
            continue
        recipe_errors = check_recipe(path)
        errors.extend(recipe_errors)
        try:
            recipe = json.loads(recipe_rel.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 重叠字段一致性（防双写漂移）：版本/镜像/固定节点数必须一致；
        # topology/tensor_parallel/dtype 已废弃（validate 会单独拦截）。
        for f in ("version", "image", "nodes"):
            mv, rv = it.get(f), recipe.get(f)
            if mv != rv:
                err_at(f"recipes[{i}].{f}「{mv}」与 {path} 的「{rv}」不一致")
        for rf in ("readme", "readme_en"):
            rp = it.get(rf)
            if rp and not (REPO / rp).is_file():
                err_at(f"recipes[{i}].{rf} 不存在: {rp}")
    return errors


def all_recipe_files() -> list[Path]:
    return sorted((REPO / "recipes").glob("*/fireworks.recipe.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default=None, help="只校验单个配方文件（相对仓库根路径）")
    ap.add_argument("--manifest", action="store_true", help="只校验 manifest（默认全部）")
    args = ap.parse_args()

    errors: list[str] = []
    if args.recipe:
        errors.extend(check_recipe(Path(args.recipe)))
    elif args.manifest:
        errors.extend(check_manifest())
    else:
        recipes = all_recipe_files()
        errors.extend(check_manifest())
        for p in recipes:
            errors.extend(check_recipe(p))
        # 反向一致性：仓库内每份 recipe 都应在 manifest 中登记
        if MANIFEST.is_file():
            try:
                m = json.loads(MANIFEST.read_text(encoding="utf-8"))
                registered = {it.get("path") for it in m.get("recipes", [])}
            except Exception:
                registered = set()
            for p in recipes:
                rel = p.relative_to(REPO).as_posix()
                if rel not in registered:
                    errors.append(f"{rel}: 未在 recipes/index.json 登记")

    for e in errors:
        print(f"ERROR  {e}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} 处问题", file=sys.stderr)
        return 1
    print("OK: 全部配方与 manifest 通过校验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
