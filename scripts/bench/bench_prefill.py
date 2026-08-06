#!/usr/bin/env python3
"""bench_prefill.py —— vLLM prefill 吞吐/首包基准（对齐 Anemll「server tokens/s」口径）

POST /v1/completions，output 仅 1 token；以响应 usage.prompt_tokens 为准，
server_tokens_per_s = prompt_tokens / TTFT(墙钟到首个输出 token)。TTFT 约等于
prefill 计算时长（kernel 时间），与 Anemll benchmark_prefill.py 口径一致。

用法:
  python3 scripts/bench/bench_prefill.py \
      --base-url http://<head-ip>:8000/v1 \
      --model deepseek-v4-flash-0731 \
      --sizes 1024,2048,4096,8192,16384
"""
import argparse
import json
import random
import time

import requests


# 近似 token 密度：每 "fireworks" 一词≈1 token（BPE）。用大量重复词拼目标长度。
WORD = "fireworks "
TOK_PER_WORD = 1.0


def prompt_of_tokens(target: int) -> str:
    n = max(1, int(target / TOK_PER_WORD))
    rnd = random.Random()
    # 混合少量变化字符避免超长连续重复被模板截断，但保持强烈周期性便于精确计数
    parts = []
    for _ in range(n):
        parts.append(WORD.rstrip())
    # ★ 随机头：从 token 0 起与历史请求不同，彻底打破服务端 prefix-caching，
    #   测得真实 prefill（尾随机只能破坏末段，主体仍命中缓存）。
    head = "".join(rnd.choices("abcdefghijklmnopqrstuvwxyz", k=64))
    return head + " " + " ".join(parts)


def one(base_url: str, model: str, prompt: str):
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 1,
              "temperature": 0.0, "stream": False},
        timeout=1800,
    )
    ttft = time.perf_counter() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    return ttft, usage.get("prompt_tokens", 0)


def warm(base_url: str, model: str):
    for _ in range(2):
        try:
            one(base_url, model, "hi")
            break
        except Exception as e:  # noqa: BLE001
            time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--sizes", default="1024,2048,4096,8192", help="目标输入 token 数，逗号分隔")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    print(f"== bench_prefill base={args.base_url} model={args.model} sizes={sizes} ==")
    warm(args.base_url, args.model)

    out = []
    for target in sizes:
        prompt = prompt_of_tokens(target)
        ttft, ptoks = one(args.base_url, args.model, prompt)
        rate = ptoks / ttft if ttft > 0 else 0.0
        print(f"  target={target:<6} prompt_tokens={ptoks:<6} "
              f"TTFT={ttft:7.2f}s  server={rate:8.1f} tok/s")
        out.append({"target": target, "prompt_tokens": ptoks,
                    "ttft_s": round(ttft, 3),
                    "server_tokens_per_s": round(rate, 1)})
    if args.json:
        print(json.dumps({"bench": "prefill", "model": args.model, "results": out},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
