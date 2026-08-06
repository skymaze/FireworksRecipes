#!/usr/bin/env python3
"""warmup_longctx.py —— 长上下文 JIT 形状预热（防推理期 OOM）

背景：vLLM 的 tilelang/flashinfer 内核按「请求形状」惰性编译；长上下文（>max_num_batched_tokens
8192）的 prefill 大块形状（mhc_pre_big_fuse_*）不会被启动 warmup 覆盖，首次触达时在推理路径里
编译，峰值内存 + vLLM 已预留内存 → DGX Spark 统一内存全局耗尽 → OOM killer 杀 Worker（实测
2026-08-05 246K prompt 崩溃，head 上 global_oom，进程 VLLM::Worker_TP）。

本脚本在服务就绪后触发两类形状编译（产物落盘到持久化的 TILELANG_CACHE_DIR，之后请求命中缓存）：
  A) 全量随机头  —— 打破 prefix-caching，编译「连续 block 布局」大块 prefill 形状
  B) 同前缀+短尾 —— 命中前缀缓存，编译「非连续 block 布局」形状（真实长系统提示 + 变化后缀场景）
对每个目标长度各发一次 A 和 B（max_tokens=1，只做 prefill + 1 token）。

用法：
  python3 scripts/deploy/warmup_longctx.py \
      --base-url http://<head-ip>:8000/v1 \
      --model deepseek-v4-flash-0731 \
      [--lens 32768,65536,131072,262144] [--only-hit] [--timeout 3600]
"""
import argparse
import random
import time

import requests

WORD = "fireworks "
TOK_PER_WORD = 1.0


def prompt_of_tokens(target: int, *, random_head: bool) -> tuple[str, int]:
    """构造约 target token 的 prompt，返回 (text, 头部随机串)。

    random_head=True 时头部注入随机字符（打破前缀缓存，测得真实全量 prefill）；
    False 时保持确定性头部（与同目标长度的 random_head 请求共享前缀）。
    """
    n = max(1, int(target / TOK_PER_WORD))
    body = " ".join([WORD.rstrip()] * n)
    if random_head:
        head = "".join(random.Random().choices(
            "abcdefghijklmnopqrstuvwxyz", k=64))
        return head + " " + body, head
    return body, ""


def one(base_url: str, model: str, prompt: str, *, timeout: int):
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 1,
              "temperature": 0.0, "stream": False},
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    return dt, usage.get("prompt_tokens", 0), usage.get(
        "completion_tokens", 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--lens", default="32768,65536,131072,262144,524288,1048576")
    ap.add_argument("--only-hit", action="store_true",
                    help="只做 B（命中缓存）形状，跳过全量 A")
    ap.add_argument("--max-ctx", type=int, default=1048576,
                    help="服务端 max-model-len（A 的 prompt 与 B 的 prompt+tail 都不能越过）")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    # B 的 tail 实际 token 数 ≈ 2050（"What is the capital of France? "×256）；
    # 预留 64 token 缓冲，保证 A/B 都不触发 max-model-len 校验（prompt+output≥max 时 400）。
    TAIL_TOKENS, SAFETY = 2100, 64
    lens = [int(x) for x in args.lens.split(",") if int(x) > 0]
    total_t0 = time.perf_counter()
    for target in lens:
        # A) 全量随机头：连续 block 布局（同时为 B 建立前缀缓存）
        a_len = min(target, args.max_ctx - 1 - SAFETY)
        prompt, _ = prompt_of_tokens(a_len, random_head=True)
        if not args.only_hit:
            try:
                dt, pt, ct = one(args.base_url, args.model, prompt,
                                 timeout=args.timeout)
                print(f"[warmup A] len≈{target}: prompt_tokens={pt} "
                      f"completion_tokens={ct} wall={dt:.1f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 —— 预热失败不阻断部署
                print(f"[warmup A] len≈{target} FAILED: {exc}", flush=True)
        # B) 复用 A 的前缀 + ~2K tail：命中缓存 → 非连续 block 布局（崩溃场景）
        #    接近 max-ctx 时截短 base 为 A prompt 的前缀子串（按字符比例近似 token
        #    数；tail 有 2100 token 缓冲，误差不会触顶），前缀缓存仍命中。
        b_len = min(a_len, max(args.max_ctx - TAIL_TOKENS - SAFETY, 1024))
        if b_len < a_len:
            prefix = prompt[: int(len(prompt) * b_len / a_len)]
        else:
            prefix = prompt
        tail = (" What is the capital of France? " * 256).strip()
        try:
            dt, pt, ct = one(args.base_url, args.model, prefix + " " + tail,
                             timeout=args.timeout)
            print(f"[warmup B] len≈{target}+tail: prompt_tokens={pt} "
                  f"completion_tokens={ct} wall={dt:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup B] len≈{target}+tail FAILED: {exc}", flush=True)
    print(f"[warmup] done in {time.perf_counter() - total_t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
