# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 900K · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，CX7 直连 RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next`）的 **EXL3/TR3 路线** 服务——与仓库内 NVFP4 配方（marlin）**不同镜像、不同 lane**：

- 权重：**[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)**
  （uniform-K4 EXL3/TR3 routed-experts，4 bpw，~164 GiB，120 分片）——4bpw 在 KLD 上与官方 FP8
  持平（~1.00×）却只要 **54% 的字节**；
- KV：**fp8 · packed `fp8_ds_mla`**（NoPE MLA 零填充进 SM12x 唯一的 sparse-MLA 后端）；
- 投机：**DFlash2 k=7**（[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)，
  ~2.3 GiB BF16，drafter 只留 rank 0）；
- 上下文：**900,000**；Vision 默认开启（image×4 / video×1）。

> **不要** `--moe-backend marlin` / NVFP4 权重 / `kv-cache-dtype nvfp4|bf16` /
> `attention_backend=TRITON_ATTN`（本镜像内 causal-in-block，会压塌后段接受率）。
> 也不要拉 `glm53-flash-sm121:v8`——那是旧的 NVFP4/Ray 内核。

## 实测数据（上游 2026-08-28，sparkDash decode bench）

DFlash2 k=7 · Structured/Code 同一高接受档 · temp 0 · thinking off · 400 tokens · CUDA graphs · 融合 EXL3 MoE：

| 并发 | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

实验室 `tests/bench_decode.py`（C1，median 5×400）：Structured **61.7** tok/s（0.918 accept /
6.43 per step）；Prose 26.9（0.332/2.33）；长上下文混合（~60–100k KV）24–27；MTP k=2 基线 ~24.6。
KV 池 util 0.87 下 **982,612** token（~15.67 GiB fp8 MLA），为满 900k 请求的 **1.09×**。

## 快速开始（发布前就绪）

- 集群：恰好 **2 台**节点（head + 1 worker），CX7 直连（NCCL 不能走 10.0.0.x loopback 别名）。
- 镜像：`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`（GHCR **公开**，无需登录）。
- 模型：**主模型 + DFlash2 drafter** 都由 Fireworks 以 `picker=model` 变量分发到各节点
  HF 缓存（`HF_HOME=/root/.cache/huggingface`，按 repo id 离线解析）。
- **NCCL**：HCA / 网卡 / GID index 由 Fireworks 自动键按节点填充。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53EXL3_IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | overlay 镜像（NoPE-MLA SM121 + exllamav3_ext + DFlash2/EAGLE3/video 烘焙） |
| `GLM53EXL3_MODEL_PATH` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | **EXL3 主模型**（勿换 NVFP4） |
| `GLM53EXL3_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（务必分发到节点缓存） |
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | 对外服务名 |
| `VLLM_PORT` | `8888` | API 端口 |
| `MAX_MODEL_LEN` | `900000` | **上游生产档**（原生 1M 分配不出来） |
| `MAX_NUM_SEQS` | `4` | decode 批（上游 pin） |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | **上游 pin**（8192 撑爆 GB10 indexer smem） |
| `GPU_MEMORY_UTILIZATION` | `0.87` | KV 池 982,612 token |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`；bf16/nvfp4 无 sparse 核 |
| `GLM53EXL3_DFLASH_TOKENS` | `7` | DFlash2 投机 token（trained block 8） |
| `GLM53EXL3_DFLASH_DRAFT_TP` | `1` | drafter 只留 rank 0（不过 CX7） |
| `LANGUAGE_MODEL_ONLY` | `0` | 0=加载视觉塔（默认）；1=仅文本 |
| `SKIP_MM_PROFILING` | `1` | 保持 1（MM dummy profile 会 OOM UMA） |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template.jinja` | 镜像内 MM 模板 |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意

- 任务名即 docker compose 项目名：只允许**小写字母/数字/`-`/`_`，不能含点 `.`**（节点 Docker
  Compose v5 硬性限制）。带点任务名会发布失败（502）。建议如 `glm53-exl3-tp2`。
- **Thinking 默认开启**：要关就在请求顶层带
  `"chat_template_kwargs": {"enable_thinking": false}`（`extra_body` 是 SDK 选项，别裸发 HTTP 嵌套对象）。
- vision 请求走 MM 模板（已默认）；`usage.prompt_tokens_details.cached_tokens` 可核对前缀缓存
  命中（已开 `--enable-prefix-caching` + `--enable-prompt-tokens-details`；OpenAI API 无状态，
  只有 block-aligned 前缀才算命中）。

## 部署注意（源自源仓库实机踩坑）

- **EXL3 ≠ NVFP4**：`--quantization exl3` 固定写死；权重、KV、镜像三者必须配套，混用必挂。
- DFlash2 草稿 KV 是 `auto`/bf16、TP=1（dense 草稿用不了目标的 `fp8_ds_mla`）；目标权重仍 `fp8`。
- 冷启动慢（权重加载 + warmup），健康检查 start_period 已设 900s；CUDA graphs 已开
  （capture `1 2 4 8 16 24 32`），不要 `--enforce-eager`。
- NCCL 必须用 CX7 直连口（HCA/网卡由 auto 键按节点填），否则 `ncclCommInitRank` 悬挂。
- 上游按头拉取镜像再 docker save|load 到 worker；Fireworks 侧各节点分别拉取即可（镜像公开）。

## 参考来源

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)：
  参数级参考（.env / start.sh / overlay / Dockerfile）
- [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)：
  EXL3/TR3 uniform-K4 4bpw 权重（ShapleyMCG License 1.0）
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿模型
  （CC BY-NC-ND 4.0）
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)：基座模型
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)：EXL3 格式/内核
