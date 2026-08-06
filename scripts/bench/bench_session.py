#!/usr/bin/env python3
"""bench_session.py —— per-session 解码/ITL 基准（对齐 dgx-spark-bench / ref1 口径）

单个会话流式输出，测：
- per-session tok/s（总输出 token / 总墙钟，含 TTFT）
- decode tok/s（排除首 token 的 TTFT 后）
- ITL p50/p90（token 间隔，流式 chunk 间）
- 固定输出或自然停止（max_tokens 大 + 自然 EOS）

用法:
  python3 scripts/bench/bench_session.py --base-url http://<head-ip>:8000/v1 \
      --model deepseek-v4-flash-0731 --prompt "Count from 1 to 300" --max-tokens 1024
"""
import argparse
import json
import statistics
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--prompt", default="Count from 1 to 300, separated by commas.")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    first_ts = None
    itls = []
    prev = None
    out_tokens = 0
    with requests.post(
        f"{args.base_url}/completions",
        json={"model": args.model, "prompt": args.prompt,
              "max_tokens": args.max_tokens, "temperature": args.temperature,
              "stream": True,
              "stream_options": {"include_usage": True}},
        stream=True, timeout=1800,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload.strip() == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            usage = d.get("usage")
            if usage is not None and usage.get("completion_tokens"):
                # usage chunk（stream_options.include_usage）无 choices
                out_tokens = usage["completion_tokens"]
                continue
            choices = d.get("choices")
            if not choices:
                continue
            ch = choices[0]
            piece = ch.get("text") or ch.get("delta", {}).get("content")
            if piece:
                if first_ts is None:
                    first_ts = now
                else:
                    itls.append(now - prev)
                prev = now
    wall = time.perf_counter() - t0
    ttft = (first_ts - t0) if first_ts else wall
    decode_wall = max(wall - ttft, 1e-9)
    session_rate = out_tokens / wall
    decode_rate = out_tokens / decode_wall
    itl_p50 = statistics.median(itls) if itls else float("nan")
    itl_p90 = sorted(itls)[int(len(itls) * 0.9) - 1] if itls else float("nan")
    print(f"out_tokens={out_tokens} wall={wall:.2f}s TTFT={ttft:.2f}s "
          f"session={session_rate:.1f} tok/s decode={decode_rate:.1f} tok/s "
          f"ITL p50/p90={itl_p50 * 1000:.1f}/{itl_p90 * 1000:.1f} ms")
    if args.json:
        print(json.dumps({"out_tokens": out_tokens, "wall_s": round(wall, 2),
                          "ttft_s": round(ttft, 2),
                          "session_tokens_per_s": round(session_rate, 1),
                          "decode_tokens_per_s": round(decode_rate, 1),
                          "itl_p50_ms": round(itl_p50 * 1000, 1),
                          "itl_p90_ms": round(itl_p90 * 1000, 1)}))


if __name__ == "__main__":
    main()
