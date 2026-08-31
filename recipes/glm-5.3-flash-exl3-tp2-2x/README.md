# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 1M · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，CX7 直连）上以 **EXL3/TR3 路线** 服务
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)（320B / A18B MoE），
与仓库内 NVFP4（marlin）配方为不同镜像、不同 lane：

- 权重：**[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
  （[brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) 快照 `5ab363a8…` 的公开镜像；
  4bpw 在 KLD 上与官方 FP8 持平 ~1.00×，仅 **54%** 字节；镜像不完整时回退 brandonmusic）；
- KV：**fp8 · packed `fp8_ds_mla`**；
- 投机：**DFlash2 k=7**（[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)，drafter 只留 rank 0；上游 2026-08-30 已把默认改为跨 TP 分片，**须待镜像重建后跟随**），
  草稿与 MLA **padded slot-share** 共管 KV 页；
- 上下文：**1,000,000**（padded slot-share 使 1M 分配得出）；Vision 默认开（image×4 / video×1）。

> 勿用 `--moe-backend marlin` / NVFP4 权重 / `kv-cache-dtype nvfp4|bf16` /
> `attention_backend=TRITON_ATTN`；勿拉 `glm53-flash-sm121:v8`（旧 NVFP4/Ray 内核）。

## 实测数据（上游 2026-08-28，sparkDash decode bench）

DFlash2 k=7 · Structured/Code 高接受档 · temp 0 · thinking off · 400 tokens · CUDA graphs · 融合 EXL3 MoE：

| 并发 | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| ×1 | 719 ms | **62.9** | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | **146.5** |

实验室（C1，median 5×400）：Structured **61.7** tok/s（0.918 accept）；Prose 26.9；长上下文（~60–100k KV）24–27；
MTP k=2 基线 ~24.6。**2026-08-29 上游 P1 阶梯把 `MAX_NUM_BATCHED_TOKENS` pin 到 2048**（8k cold TTFT 10.36s/772 → 8.93s/895、100k 947→975、decode 无退化；3584/4096 被 LinearEXL3 胖 expert 税吃掉、已 revert；**8192 撑爆 GB10 indexer smem，永远别上**）。
**2026-08-30 上游又把 `DFLASH_DRAFT_TP` 默认改为 2**（drafter 跨 TP 分片，实验室 structured 65.1 tok/s）——该验收出自上游新 overlay 构建，**当前 `:exl3`（2026-08-28 烘焙）尚未验证 TP=2，本配方默认保持已实机验证的 1**，等镜像重建后再切 2。
上游 1M serve（util 0.87，同池 1,754,237 token / **1.75×** / 690 blocks / **18.67 GiB**）；
KV 余量随 boot 浮动——本机 0.87 曾只余 11.77 GiB（< 1M 所需 14.61 GiB）；先核对节点 `:exl3` 与上游同一构建，
差一口气时优先把 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 设为 0（去掉 CUDA-graph 显存预留，
还 KV ~2.6 GiB，graphs 仍开；上游 2026-08-29 CG_ESTIMATE 旋钮），仍不够再调高 util（≥0.90）。
前缀缓存 block-aligned（3584-token）命中，~7.7k 后续轮 93% 命中、TTFT 9.7s → 1.17s。

## 快速开始

- 集群：恰好 **2 台**节点（head + 1 worker），CX7 直连（NCCL 不能走 10.0.0.x loopback 别名）。
- 镜像：`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`（GHCR 公开，无需登录；节点不支持 pull，镜像由集群镜像仓库分发到各节点，本地已有即用）。
  若 KV 池异常偏小，先核对节点上的 `:exl3` 与上游为同一构建。
- 模型：**主模型 + DFlash2 drafter** 由 Fireworks 以 `picker=model` 分发到各节点 HF 缓存
  （`HF_HOME=/root/.cache/huggingface`，按 repo id 离线解析）。
- **NCCL**：HCA / 网卡 / GID index 由 Fireworks 自动键按节点填充。
- 冷启动后首个请求会触发一次 JIT 编译（上游有 boot 预热钩子，Fireworks 侧无），略慢属正常；
  Triton/TileLang 缓存已挂持久卷，重建不丢。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53EXL3_IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | overlay 镜像 |
| `GLM53EXL3_MODEL_PATH` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | **EXL3 主模型**（勿换 NVFP4） |
| `GLM53EXL3_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（须分发到节点缓存） |
| `MAX_MODEL_LEN` | `1000000` | 上游生产档 1M（勿降 256k） |
| `GPU_MEMORY_UTILIZATION` | `0.87` | 上游实测值（池 ~18.67 GiB）；本机若 <14.61 GiB 先核对镜像构建，再调 ≥0.90 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | 0 = 去掉 CUDA-graph 显存预留还 KV ~2.6 GiB（graphs 仍开；KV 池不足时先试它） |
| `MAX_NUM_SEQS` | `4` | decode 批（上游 pin） |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | 上游 2026-08-29 P1 keep（8k −16% TTFT、100k +3%；3584/4096 revert；**8192 撑爆 GB10 indexer smem**） |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`；别用 bf16/nvfp4 |
| `GLM53EXL3_DFLASH_TOKENS` | `7` | DFlash2 投机 token（trained block 8） |
| `GLM53EXL3_DFLASH_DRAFT_TP` | `1` | drafter 只留 rank 0（当前镜像上唯一实机验证档）。2 = 跨 TP 分片（上游 2026-08-30 keep）**待 :exl3 重建后启用** |
| `EXL3_FUSED_MOE` | `1` | 每层融合 `exl3_moe`；0 = 逐 expert 循环 |
| `GLM53_MIXED_PREFILL_CHUNK` | `skip` | decode 步不混入 peer prefill（上游 pin） |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | thinking 内客户端 stop 保持休眠 |
| `LANGUAGE_MODEL_ONLY` | `0` | 0=加载视觉塔；1=仅文本（更快） |
| `LIMIT_MM` | `{"image":4,"video":1}` | 每请求多模态上限 |
| `SKIP_MM_PROFILING` | `1` | 保持 1（profiling OOM UMA） |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template.jinja` | 镜像内 MM 模板 |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*` 由 Fireworks
自动填充；另有 `SERVED_MODEL_NAME`（默认 `GLM-5.3-Flash-EXL3`）、`VLLM_PORT`（默认 8888）可调。

## 发布注意

- 任务名即 compose 项目名：只允许小写字母/数字/`-`/`_`，**不能含点 `.`**（否则发布 502）。建议 `glm53-exl3-tp2`。
- **Thinking 默认开**：请求顶层带 `"chat_template_kwargs": {"enable_thinking": false}` 关闭
  （`extra_body` 是 SDK 选项，别裸发 HTTP 嵌套对象）。
- `usage.prompt_tokens_details.cached_tokens` 可核对前缀缓存命中（`--enable-prefix-caching` +
  `--enable-prompt-tokens-details` 已开；OpenAI API 无状态，仅 block-aligned 前缀命中）。
- **EXL3 ≠ NVFP4**：`--quantization exl3` 固定；权重、KV、镜像必须配套。草稿 KV 为 `auto`/bf16、
  TP=1（dense DFlash2 用不了目标的 `fp8_ds_mla`）；目标仍 `fp8`。
- **KV 余量因机而异**：boot 日志的 `Available KV cache memory` 若低于 14.61 GiB（1M 硬需求），
  调高 `GPU_MEMORY_UTILIZATION`（≥0.90）；过高会对 MM/长 prefill 峰值失去余量。
- 冷启动慢，健康检查 start_period 900s；CUDA graphs 已开，勿 `--enforce-eager`。
- 容器启动会先执行镜像内运行时 overlay 补丁（含禁用 GB10 `persistent_topk`、xgrammar 投机解码终止修复等），
  与上游 start.sh 一致；补丁一律 `if [ -f ]` 哨兵执行，缺文件自动跳过。
- **⚠️ 已发布 `:exl3` 镜像落后上游 main（2026-08-31 核实）**：GHCR 三个 tag 均为 2026-08-28 07:46Z
  同一构建；上游现行 Dockerfile 烘焙的 5 个运行时补丁在此之后才入仓——
  `patch_suppress_stops_in_reasoning.py`（08-28 14:12Z）、`patch_scheduler_decode_floor.py`（08-28 16:34Z）、
  `patch_hybrid_prefix_hit.py`（08-28 23:18Z）、`patch_xgrammar_termination.py`（08-29）、
  `patch_kpool_tail_slotmap.py`（08-30）——**当前镜像上这五项静默不生效**（thinking 内 stop 抑制、
  decode 步隔离混合 prefill、MLA 前缀命中保留、xgrammar 终止修复、kpool tail 修复），且镜像内
  `overlay/exl3.py` 为旧版（`THE_TEMP_ROWS` 等环境旋钮不存在，但 MNBT=2048 行为语义等价）。
  **镜像不重发则这五项功能缺失**：等上游重建 GHCR tag，或按仓库惯例由镜像仓库
  （glm53-flash-sm121 同款流程）按上游现行 Dockerfile 烘焙 ACR 镜像后改 `GLM53EXL3_IMAGE`。
- NCCL 必须用 CX7 直连口（auto 键按节点填），否则 `ncclCommInitRank` 悬挂。

## 参考来源

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)：
  参数级参考（.env / start.sh / overlay / Dockerfile；2026-08-29：1M 上下文 + padded slot-share + hybrid prefix hit +
  MNBT=2048 P1 keep；2026-08-30：`DFLASH_DRAFT_TP=2` keep + kpool-tail slotmap 修复（`overlay/patch_kpool_tail_slotmap.py`，
  长生成时通用 paged kernel 会越过 tail group 单 block-table 条目、可崩溃或静默损坏 indexer——修复 pin 在单 block 环形 scratch，
  随下次镜像烘焙生效）+ per-rank GID（与我们按节点 gid_index 自动键同思路））
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)：
  EXL3/TR3 4bpw 权重镜像（ShapleyMCG License 1.0）· [原始权重](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿（CC BY-NC-ND 4.0）
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)
