# DeepSeek-V4-Flash-Vision-Exp · 4-node TP=4 · Fireworks recipe

Serve DeepSeek-V4-Flash-Vision-Exp on **4** DGX Spark (head + 3 workers, direct
RoCE 10.1.0.0/24) at **TP=4**. Based on the **hotfix8** image, whose entrypoint now
parametrizes `--tensor-parallel-size` / `--nnodes` via `DSPARK_TP` / `NODES_TOTAL`,
so the same image serves 2-node or 4-node.

## Model & mechanics

- Model `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (305B incl. ViT/Aligner, NVFP4
  DS-MLA; ~157 GiB node cache snapshot `86f746b3`)
- KV: **nvfp4_ds_mla** (padded `page_block_size=64`); **~51 GiB ≈ 7.57M token** KV pool/rank
  (far beyond 1M context)
- Speculation: **dspark k=3** (DSpark draft; **n_predict=3 tier requires k divisible by 3**)
- Image `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix8`
  (hotfix7 + TP-parametrized entrypoint; full hotfix chain and all DSPARK_* flags intact)
- Distributed: mp backend, `--nnodes 4 --node-rank N --master-addr <head RoCE IP>`

## On-hardware benchmarks (2026-09-08, 4xGB10, prose decode, reasoning off)

| Spec k | ws=1 stream | ws=2 agg | ws=4 agg | ws=6 agg | ws=8 agg |
|---|---:|---:|---:|---:|---:|
| **k=3** | **54.9** t/s | 88.1 | 113.5 | **160.3** (peak) | 129.3 |
| k=6 | 46.5 t/s | 70.2 | 103.8 | 126.0 | 109.7 |

- **k=3 beats k=6 everywhere** (+18% single-stream, +27% at ws=6): the draft's
  n_predict=3 tier gives the highest acceptance at k=3
- **ws=6 is the throughput sweet spot** (matches default `MAX_NUM_SEQS=6`); ws>=8 gets uneven latency
- Long generation (1024t, ws=6): **147.1 t/s aggregate**, ~25/stream
- reasoning=low short answer: ~76 t/s, tiny TTFT
- **TTFT / prefill** (short-token responses): 1.7k→0.26 s, 8.8k→0.22 s, 17.6k→0.29 s,
  36.3k→0.34 s
- Unified memory: 121G budget nearly exhausted (`MemAvailable 2-3G` stable, no OOM);
  ~41.5 GiB weights/rank + ~51 GiB KV + CUDA/JIT
- Cold start ~2.5-3 min to healthy (weight load ~10-20s cache-hit + KV conversion)

## ⚠️ TP8 (8 nodes) cannot deploy

**Root cause:** FlashInfer's DSV4 sparse-MLA **prefill** kernel
(`sparse_mla_sm120_prefill.cu`) only instantiates `num_heads ∈ {16, 32, 64, 128}`
(MG dispatch `dispatch_dsv4_single/dual`). The model has `num_attention_heads = 64`,
so TP8 drops each rank to 8 heads -> no template -> startup fails
`Unsupported sparse-MLA prefill configuration: num_heads=8`. TP6 (64/6 not integer)
is likewise impossible. **TP4 is the largest usable topology on an 8-node cluster**
until FlashInfer ships an 8-head template.

## Deployment notes

- One GPU per node; **must set `--ulimit nofile=1048576`** — NCCL multi-node hits
  `Too many open files` at the default 1024 and rank handshakes fail (hit in testing)
- head (rank0) `HEADLESS` empty; workers `HEADLESS=1` (`${HEADLESS:+--headless}`
  misjudges 0/non-empty as headless — fixed in hotfix8)
- Use GMU 0.835; lower it if startup reports `Free memory < desired`
- RoCE HCA `rocep1s0f0,rocep1s0f1` (100G EDR), netdev `enp1s0f0np0`

## Upstream references

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  @`957890ac` (hotfix7 snapshot); TP4-ization built on this repo's `dspark-image-build` hotfix8
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) (base image 0.1.1)
- FlashInfer `sparse_mla_sm120` sparse-MLA kernels (the TP ceiling's root cause)
