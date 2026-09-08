# Qwen3.8-Flash-Next · single-node TP=1 · Fireworks recipe (1× DGX Spark)

Run **Qwen3.8-Flash-Next** (176.9B) vLLM serving on **one** DGX Spark (GB10, 128 GiB
unified memory) with Fireworks: **TP=1**, PLE lookup table offloaded from NVMe, MTP
speculation with reduced-vocabulary drafting, native 262,144 / YaRN 524,288 context,
text + image + video multimodal. Ported from
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
(AGPL-3.0).

## Model & mechanism

- Checkpoint: `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (NVFP4, ~99 GB on disk)
- **PLE-table offload is what makes this fit one Spark**: the n-gram/PLE lookup table
  (26.82 GiB) is never computed - it is served by a CPU offload worker from a
  **memory-mapped** packed table (`VLLM_PLE_CPU_OFFLOAD=1`). The mapping is advised
  `MADV_RANDOM` plus per-batch `POSIX_FADV` prefetch, cutting disk read per decoded token
  from ~1,366 KiB to **57 KiB (−24×)** and freeing ~2 GiB of unified memory otherwise
  wasted on readahead
- Weights on GPU ~71.75 GiB + ~5.6 GiB runtime overhead; KV sized by
  `GPU_MEMORY_UTILIZATION`, **capped from the host side by `HOST_RESERVE_GIB=26`**
  (aligned with upstream **09-06 shipped**: `KV_TARGET_GIB=20` -> GMU 0.786 = 95.60 GiB
  budget -> 16.67 GiB KV; ~15.98 GiB fp8 KV = ~1,132,586-token pool after vLLM profiling,
  ~4.3x a full 262k request, pool varies ~10% across restarts)
- **BF16 GDN/SSM recurrent state** (`MAMBA_SSM_CACHE_DTYPE=bfloat16`, upstream 09-06 shipped):
  halves the per-step state traffic (~0.23 GB/seq/step) and the mamba page (attention block
  3,200->1,664 tokens); +8.5% aggregate decode at 8 streams, needles 15/15 unchanged; empty =
  the checkpoint's float32
- **V2 model runner pinned** (`VLLM_USE_V2_MODEL_RUNNER=1`): the MTP draft copy is not in the
  V2 default set, falls back to V1, and mutates its shared `compilation_config` - the 09-05
  dynamic-K failure was cudagraph_mode silently becoming PIECEWISE; pinned on every 09-06 launch
- **Image**: `registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.2.0`
  (official `vllm/vllm-openai:qwen38-flash-next` base + **all five** upstream MiaAI patches
  baked in: PLE-layer prefetch, PLE offload host handshake, MTP draft vocab, ModelOpt MXFP8
  BF16 fallback, QSA FP8-KV). The upstream repo rewrites vLLM files at launch; one-click
  Fireworks deploy uses this baked image - the stock image deadlocks PLE offload and breaks
  fp8 KV. The 09-06 changes are runtime env/args only - no new image needed
- Port default `8888`; served-model-name `qwen3.8-flash-next`; OpenAI-compatible API
- **Upstream 09-07/09-08 `ABLIT` (gated Keys abliterated checkpoint,
  `drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only`) is intentionally not adopted**: it removes safety refusals and
  needs you to accept gated HF terms and fetch a separate ~99 GiB checkpoint — an optional
  experiment, not the shipped default; this recipe pins stock Mia NVFP4 (ABLIT=0). The other
  new commits are checkpoint-state resolution, docs and sweep tooling — no shipped-config change

## Pre-deploy prerequisites (prepare on the node)

1. **Model**: distribute `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` to the node HF cache (~99 GB).
2. **Packed PLE table** (one-time, ~27 GB): on the host, run the upstream repo's
   `files/build_ple_packed_table.py` against the distributed checkpoint once; output lands in
   `~/.cache/vllm/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/*.packed_u8`
   (this recipe mounts that dir into the container and the launch checks for it).
3. **Image**: confirm you run the v1.2.0 baked variant (see above).
4. **(optional) MTP reduced draft vocab**: generate a token-id-per-line file with
   `files/build_draft_vocab.py` (65,536 rows at the 256k tier, ~0.16 GiB) and set
   `MTP_DRAFT_VOCAB` to its host path to enable reduced-vocabulary drafting (+25% decode).

## Variables at a glance

| Variable | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | 262144 | context at YARN=0 (native cap 262,144; above is refused) |
| `YARN` | 0 | 0 = native rope; 1 = YaRN -> `YARN_MAX_MODEL_LEN` (512k, factor 2.0) |
| `YARN_MAX_MODEL_LEN` | 524288 | length served at YARN=1; 1M fails the single-Spark budget at BF16, don't raise |
| `KV_CACHE_DTYPE` | fp8 | fp8 ~1.85x KV pool (2.73x a full 512k req); auto = BF16 |
| `MAMBA_SSM_CACHE_DTYPE` | bfloat16 | GDN/SSM state dtype; bf16 +8.5% decode at 8 streams (09-06 shipped); empty = float32 |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | pin V2 runner; 0 = unpinned (MTP draft copy may fall back to V1 and flip the graph mode) |
| `GPU_MEMORY_UTILIZATION` | 0.786 | aligned with 09-06 shipped (KV_TARGET_GIB 20 -> GMU 0.786 = 95.60 GiB budget); raising it must keep host headroom |
| `MAX_NUM_SEQS` | 4 | measured at 1/2/3/4 streams; KV pool ~4.3x (long context is the concurrency limit, not a throughput knob) |
| `MTP` | 3 | trained-draft spec k=3 (0 off, saves ~1.49 GiB); multimodal requests fall back |
| `MTP_DRAFT_VOCAB` | empty | host path to a token-id file; set = reduced-vocabulary drafting (+25% decode) |
| `MAX_NUM_BATCHED_TOKENS` | 2048 | chunked-prefill batch cap; 8192 buys ~11% prefill for ~3% of the KV pool |
| `CUDAGRAPH_CAPTURE_SIZES` | auto | every buildable verify-batch width (1+K)xS; vLLM's own list misses 3/5-seq batches at MTP=3 |
| `PORT` | 8888 | comfy-h3.service watches this port (see safety) |
| `CUDAGRAPH_MODE` | FULL_DECODE_ONLY | NONE for eager debugging |
| `VLLM_CACHE` | `${HOME}/.cache/vllm` | host dir of the packed PLE table (mounted at `/root/.cache/vllm`) |

## Performance (upstream 2026-09-05/09-06, measured with sparkDash)

**decode** (prose, idle; shipped tier: 262k, FP8, MTP 3, `CUDAGRAPH_CAPTURE_SIZES=auto`,
`MTP_DRAFT_VOCAB` 65,536 tokens all active) — 09-05 baseline:

| streams | TTFT | aggregate | per stream | vs 09-04 |
|---|---:|---:|---:|---:|
| 1 | 270 ms | 46.3 tok/s | 46.3 tok/s | +25.5% |
| 2 | 466 ms | 73.0 tok/s | 36.5 tok/s | +27.2% |
| 3 | 355 ms | 91.9 tok/s | 31.2 tok/s | — |
| 4 | 346 ms | 108.1 tok/s | 27.7 tok/s | +25.8% |

- Almost all of it is the **reduced draft vocabulary**: slicing the drafter's lm_head to
  65,536 rows saves 2.61 GiB per draft step, and decode here is close to the memory-bandwidth
  wall, so bytes removed convert almost one-for-one into time. The target model verifies every
  token, so output is unchanged (MGSM: EN 94.8% vs 93.6%, ZH 86.4% vs 86.4% - flat/ahead)
- MTP mean acceptance length ~2.1 of a possible 4; **K=3 stays optimal at every concurrency**
  (09-06 static sweep; nothing to schedule)

**09-06 sweep** (512k YaRN, FP8, MTP 3, bf16 GDN state, V2 pinned, every verify width captured
as a FULL decode graph):

| streams | ms/engine step | tokens/step | aggregate | per stream |
|---|---:|---:|---:|---:|
| 1 | 61.5 | 3.00 | **48.7 tok/s** | 48.7 tok/s |
| 4 | 96.2 | 2.84 | **113.7 tok/s** | 28.4 tok/s |
| 8 | 131.0 | 2.81 | **162.9 tok/s** | 20.4 tok/s |

- the 8-stream +8.5% (151.6 -> 164.5 tok/s) is the **bf16 GDN state**; K=1 loses 8-14%, and the
  static K sweep (K=0/1/2/3 at S=1/2/4/8) found no crossover — K=3 stays optimal everywhere

**prefill** (fp8, bf16 state): **2,200 @8k / 2,304 @16k / 2,314 @32k** / 2,257 @64k /
2,146 @128k / 1,944 @256k tok/s (+~1.7% vs 09-05; the +8.5% from bf16 sits in decode, prefill
is rate-limited the same way)

- 512k YaRN + FP8 KV: **KV pool ~1,431,164-1,502,014 tokens (2.73-2.86x a full request)**
- Host headroom (shipped 262k tier, KV_TARGET 20): 15.7 GiB after launch, 15.5-16.4 GiB over 40
  idle minutes, 14.26 GiB low with five concurrent ~60k prompts, `NV_ERR_NO_MEMORY` 0
  (reproduced over the ten 09-06 launches)

## Multimodal

- The 27-layer vision tower is already counted in "weights on GPU", so images/video cost no
  extra GPU budget and are on by default
- Verified upstream: 336x336 three-band PNG named in order; 4 s / 16-frame clip named by
  temporal order
- Standard OpenAI content parts: `image_url` / `video_url` (`http(s)://` or `data:` URI)
- Note: **MTP automatically degrades on multimodal requests** (the draft model cannot take
  embeddings; vLLM logs `using text-only draft inputs instead`); video is token-hungry, and
  long-video-at-512k is untested

## Reasoning is on by default

`--reasoning-parser qwen3` returns the thinking block in a separate `reasoning` field. To turn
it off **per request** (no restart): pass `"chat_template_kwargs":{"enable_thinking":false}`.
With a small `max_tokens` an empty `content` is normal (the answer is still inside reasoning);
budget ~400+ tokens or disable thinking.

## Safety rules (upstream pitfalls)

- **Cap the GPU budget from the host side**: GMU x MemTotal <= MemTotal - `HOST_RESERVE_GIB`
  (26). The first scheme (KV_TARGET_GIB 22, GMU 0.83) lost three servers on 2026-09-04 when
  the host MemAvailable was drained; since 09-05 the budget is capped from the host. Raising
  GMU means re-doing the upstream `start.sh` budget arithmetic first - keep host MemAvailable
  >= ~10 GiB, exhausting the unified pool hangs the kernel (no OOM, no logs)
- **comfy-h3.service must be disabled**: it polls `127.0.0.1:8888` and launches ComfyUI (a
  GPU co-tenant) as soon as anything answers there. While active, 8888 would trigger it;
  `sudo systemctl disable --now comfy-h3.service` or use another port
- **Never set `PLE_OFFLOAD=false`** (TP=1): 99 GB of weights through UVM hangs the host
- **The baked image includes the QSA FP8-KV patch** (v1.2.0); without it `KV_CACHE_DTYPE=fp8`
  silently serves garbage instead of erroring
- **Don't raise `YARN_CEILING_MODEL_LEN` past 524288** (BF16): 1M needs ~28.8 GiB KV and
  drives the container cap past the 105 GiB hard ceiling

## Reference upstream

- [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  (AGPL-3.0-or-later): source of every parameter, the memory-budget arithmetic, the patch
  list and the measured numbers; full memory / offload-handshake / watchdog logic lives in its
  `start.sh` (since 09-05 a dual-threshold `MemAvailable` / `MemFree` watchdog)
- [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) ·
  [Qwen](https://qwen.ai) (model & PLE writeups)
- FP8-KV idea credited to [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)
  (Apache-2.0)

See root [`NOTICE.md`](../../NOTICE.md) for full sources and derivations.
