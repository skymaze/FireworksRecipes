# DeepSeek-V4-Flash-Vision-Exp · 4 节点 TP=4 · Fireworks 配方

在 **4 台** DGX Spark（head + 3 worker，RoCE 10.1.0.0/24 直连）上以 **TP=4** 服务
DeepSeek-V4-Flash-Vision-Exp。基于 **hotfix8** 镜像——入口脚本已将
`--tensor-parallel-size` / `--nnodes` 参数化（`DSPARK_TP` / `NODES_TOTAL`），
同一镜像同时支持 2 节点 / 4 节点。

## 模型与机制

- 主模型 `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`（305B 含 ViT/Aligner，NVFP4
  DS-MLA，回收 ~157 GiB/节点缓存快照 `86f746b3`）
- KV：**nvfp4_ds_mla**（padded `page_block_size=64`）；每 rank KV 池 **~51 GiB ≈
  7.57M token**（远超 1M 上下文）
- 投机：**dspark k=3**（DSpark 草稿，**n_predict=3 档位需 k 被 3 整除**）
- 镜像 `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix8`
  （hotfix7 + 入口 TP 参数化；热修复链 100% 保留，DSPARK_* 开关全可用）
- 分布式：mp 后端，`--nnodes 4 --node-rank N --master-addr <head RoCE IP>`

## 实机基准（2026-09-08，4×GB10，散文 decode，reasoning off）

| 投机 k | ws=1 单流 | ws=2 聚合 | ws=4 聚合 | ws=6 聚合 | ws=8 聚合 |
|---|---:|---:|---:|---:|---:|
| **k=3** | **54.9** t/s | 88.1 | 113.5 | **160.3**（峰值） | 129.3 |
| k=6 | 46.5 t/s | 70.2 | 103.8 | 126.0 | 109.7 |

- **k=3 全面优于 k=6**（单流 +18%，ws=6 +27%）：草稿模型 n_predict=3，k=3 接受率最高
- **ws=6 是吞吐甜点**（配默认 `MAX_NUM_SEQS=6`）；ws≥8 每流延迟不均（排队）
- 长生成（1024t，ws=6）：聚合 **147.1 t/s**，每流 ~25
- reasoning=low 短回答：~76 t/s，TTFT 极小
- **TTFT / prefill**（短 token 响应）：1.7k→0.26s、8.8k→0.22s、17.6k→0.29s、
  36.3k→0.34s（prefill 大 batch 高效）
- 统一内存：121G 预算几乎吃满（`MemAvailable 2–3G` 稳定、无 OOM）；权重 ~41.5 GiB/rank + KV ~51 GiB + CUDA/JIT
- 冷启动：权重加载（缓存命中后 ~10–20s）+ KV 化，健康就绪总耗时 ~2.5–3 min

## ⚠️ 无法部署 TP8（8 节点）

**根因**：FlashInfer DSV4 稀疏-MLA **预填**内核（`sparse_mla_sm120_prefill.cu`）的
TVM 模板只实例化了 `num_heads ∈ {16, 32, 64, 128}`。模型 `num_attention_heads = 64`，
TP8 时每 rank 降到 8 头 → 无模板 → 启动失败
`Unsupported sparse-MLA prefill configuration: num_heads=8`。TP6（64÷6 非整除）同样不可行。
**TP4 是本模型在 8 台集群上的最大可用拓扑**，除非 FlashInfer 上游新增 8 头模板。

## 部署注意

- 每节点 1 GPU；**必须设 `--ulimit nofile=1048576`**——NCCL 多节点在默认 1024 时
  `Too many open files`，rank 握手失败（实踩）
- head（rank0）`HEADLESS` 留空；worker 设 `HEADLESS=1`（`${HEADLESS:+--headless}`
  在值为 0/非空时会误判为 headless，见 hotfix8 修复）
- 建议 GMU 0.835；如启动报 `Free memory < desired` 降 GMU
- RoCE HCA `rocep1s0f0,rocep1s0f1`（100G EDR），网卡 `enp1s0f0np0`

## 参考上游

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  @`957890ac`（hotfix7 快照）；本配方的 TP4 化基于仓库内 `dspark-image-build` 的 hotfix8
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)（基镜像 0.1.1）
- FlashInfer `sparse_mla_sm120` 稀疏-MLA 内核（TP 上限的根因）
