#!/usr/bin/env python3
"""bench_decode.py —— vLLM decode 吞吐基准（对齐 Anemll「server tokens/s」口径）

非流式 POST /v1/completions，统计并发请求期间的**聚合输出 token 数 / 总墙钟**，
并给出每请求平均/中位延迟。同一 harness 复用于新旧镜像对比，保证同口径。

用法:
  python3 scripts/bench/bench_decode.py \
      --base-url http://<head-ip>:8000/v1 \
      --model deepseek-v4-flash-0731 \
      --concurrency 1,2,4 --max-tokens 128 --trials 1
"""
import argparse
import concurrent.futures
import json
import statistics
import time

import requests


def once(base_url: str, model: str, prompt: str, max_tokens: int, temperature: float):
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=900,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    return elapsed, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)


def warm(base_url: str, model: str, prompt: str):
    for _ in range(2):
        try:
            once(base_url, model, prompt, max_tokens=8, temperature=0.0)
            break
        except Exception as e:  # noqa: BLE001
            time.sleep(3)
    print(f"  warmup done (model={model})")


def run_conc(base_url, model, prompt, max_tokens, conc, trials, temperature):
    results = []
    for t in range(trials):
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(once, base_url, model, prompt, max_tokens, temperature)
                    for _ in range(conc)]
            t0 = time.perf_counter()
            done = [f.result() for f in futs]
            wall = time.perf_counter() - t0
        lat = [d[0] for d in done]
        toks = sum(d[1] for d in done)
        results.append((wall, toks, lat))
    wall, toks, lat = results[-1]
    agg_rate = toks / wall
    avg_lat = statistics.mean(lat)
    med_lat = statistics.median(lat)
    print(
        f"  conc={conc:<2} trial={trials:<1} wall={wall:6.2f}s "
        f"out_tokens={toks:<5} agg={agg_rate:7.2f} tok/s   "
        f"lat avg/med = {avg_lat:.2f}/{med_lat:.2f}s"
    )
    return dict(concurrency=conc, wall_s=round(wall, 3), out_tokens=toks,
                agg_tokens_per_s=round(agg_rate, 2),
                avg_lat_s=round(avg_lat, 3), med_lat_s=round(med_lat, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--prompt",
                    default="请用中文写一个关于海滨城市冬天的短故事，约一百字。")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    print(f"== bench_decode base={args.base_url} model={args.model} "
          f"max_tokens={args.max_tokens} ==")
    warm(args.base_url, args.model, args.prompt)

    out = []
    for conc in [int(x) for x in args.concurrency.split(",")]:
        out.append(run_conc(args.base_url, args.model, args.prompt, args.max_tokens,
                            conc, args.trials, args.temperature))
    if args.json:
        print(json.dumps({"bench": "decode", "model": args.model,
                          "max_tokens": args.max_tokens, "results": out},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
