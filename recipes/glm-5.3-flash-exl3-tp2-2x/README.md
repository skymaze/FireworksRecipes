# GLM-5.3-Flash EXL3 · TP=2 · DFlash2 · 1M · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，CX7 直连）上以 **EXL3/TR3 路线** 服务
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)（320B / A18B MoE），
与仓库内 NVFP4（marlin）配方为不同镜像、不同 lane：

- 权重：**[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
  （[brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) 快照 `5ab363a8…` 的公开镜像；
  4bpw 在 KLD 上与官方 FP8 持平 ~1.00×，仅 **54%** 字节；镜像不完整时回退 brandonmusic）；
- KV：**fp8 · packed `fp8_ds_mla`**；
- 投机：**DFlash2 k=7**（[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)，drafter 跨 TP 分片），
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
**2026-08-30 上游又把 `DFLASH_DRAFT_TP` 默认改为 2**（drafter 跨 TP 分片，实验室 structured 65.1 tok/s）；
**2026-09-01 上游合并胖 expert prefill 加速（#77）：`EXL3_FAT_KERNEL=1` 默认 + `MAX_NUM_BATCHED_TOKENS` 默认 2048→**7168**（E2 内核下本机最优单发值；旧 3584/4096 被胖 expert 税吃掉是 no-fat-kernel 时代的结论），另加数值化自旋（`GLM53_SPINWAIT_MS`）与 indexer workspace 右尺寸（`GLM53_INDEXER_WORKSPACE=rightsize`，可选省 ~5 GiB KV）。上游还合并了实验性 TP4 线（`start-tp4.sh` / `.env.tp4`，4× GB10，本配方仍为 TP=2）。**镜像已按上游现行 overlay 重建（ACR `glm53-flash-exl3:v1.1.0`，@c707598），本配方已跟随默认 2**。
上游 1M serve（util 0.87，同池 1,754,237 token / **1.75×** / 690 blocks / **18.67 GiB**）；
KV 余量随 boot 浮动——本机 0.87 曾**只余 ~11.9 GiB（< 1M 所需 ~14.5 GiB，v1.1.0 实测直接启动失败）**；
v1.5.1 起配方默认已把两个旋钮改为**还 KV**：
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`（不扣 CUDA-graph 显存估计，+~2.6 GiB，graphs 仍开）+
`GLM53_INDEXER_WORKSPACE=rightsize`（1M 下 indexer workspace 从锁死 ~5 GiB 缩到按需，+~4.9 GiB）；
仍不够再调高 util（≥0.90）。
前缀缓存 block-aligned（3584-token）命中，~7.7k 后续轮 93% 命中、TTFT 9.7s → 1.17s。

## 快速开始

- 集群：恰好 **2 台**节点（head + 1 worker），CX7 直连（NCCL 不能走 10.0.0.x loopback 别名）。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.1.0`（ACR 按上游现行 Dockerfile 烘焙，
  @c707598 / 2026-09-01；节点不支持 pull，镜像由集群镜像仓库分发到各节点，本地已有即用）。
  若 KV 池异常偏小，先核对节点镜像是否为本 ACR tag（旧的 GHCR `:exl3` 是 2026-08-28 构建、缺 5 个运行时补丁）。
- 模型：**主模型 + DFlash2 drafter** 由 Fireworks 以 `picker=model` 分发到各节点 HF 缓存
  （`HF_HOME=/root/.cache/huggingface`，按 repo id 离线解析）。
- **E2 内核**：`EXL3_FAT_KERNEL=1`（默认）走镜像内构建期编译的 exl3_fat_gemm；要回退旧胖 expert 路径设 0（无需重建镜像）。
- **NCCL**：HCA / 网卡 / GID index 由 Fireworks 自动键按节点填充。
- 冷启动后首个请求会触发一次 JIT 编译（上游有 boot 预热钩子，Fireworks 侧无），略慢属正常；
  Triton/TileLang 缓存已挂持久卷，重建不丢。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53EXL3_IMAGE` | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.1.0` | overlay 镜像（ACR 烘焙 @c707598，含 exl3_fat_gemm E2 内核） |
| `GLM53EXL3_MODEL_PATH` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | **EXL3 主模型**（勿换 NVFP4） |
| `GLM53EXL3_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（须分发到节点缓存） |
| `MAX_MODEL_LEN` | `1000000` | 上游生产档 1M（勿降 256k） |
| `GPU_MEMORY_UTILIZATION` | `0.87` | 上游实测值（池 ~18.67 GiB）；本机若 <14.61 GiB 先核对镜像构建，再调 ≥0.90 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `0` | 0（默认）= 不扣 CUDA-graph 显存估计，还 KV ~2.6 GiB（graphs 仍开）——1M 需 ~14.5 GiB KV，默认 1 时本机只有 ~11.9 GiB 会启动失败 |
| `MAX_NUM_SEQS` | `4` | decode 批（上游 pin） |
| `MAX_NUM_BATCHED_TOKENS` | `7168` | 上游 2026-09-01 E2 默认（`EXL3_FAT_KERNEL=1` 下本机最优单发；旧 2048 是 08-29 老内核 keep。**8192 撑爆 GB10 indexer smem，永远别上**） |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`；别用 bf16/nvfp4 |
| `GLM53EXL3_DFLASH_TOKENS` | `7` | DFlash2 投机 token（trained block 8） |
| `GLM53EXL3_DFLASH_DRAFT_TP` | `2` | drafter 跨 TP 分片（上游 2026-08-30 keep，structured 65.1 tok/s；镜像重建后已跟随）。1 = 只留 rank 0 |
| `EXL3_FUSED_MOE` | `1` | 每层融合 `exl3_moe`；0 = 逐 expert 循环 |
| `EXL3_FAT_KERNEL` | `1` | E2 fat-expert prefill 内核（v1.1.0 镜像已编译，上游 2026-09-01 默认）；0 = 旧胖 expert 路径 |
| `GLM53_SPINWAIT_MS` | `stock` | stock = vLLM 默认 1s 自旋；数字 = 毫秒（上游数值化自旋补丁；16ms 档单流略快） |
| `GLM53_INDEXER_WORKSPACE` | `rightsize` | rightsize（默认）= 1M 下把锁死的 ~5 GiB indexer workspace 缩到按需，还回 KV（本机必须）；`stock` = 上游默认（锁 ~5 GiB） |
| `ABLIT` | `0` | 0 = stock 权重；1 = o_proj 正交化（dealign，15-45 层含 MTP 块）。`patch_ablit.py` 每次启动应用（与上游 start.sh 一致） |
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
  （dense DFlash2 用不了目标的 `fp8_ds_mla`）；`DFLASH_DRAFT_TP=2`（默认）跨 TP 分片；目标仍 `fp8`。
- **KV 余量因机而异**：boot 日志的 `Available KV cache memory` 若低于 ~14.5 GiB（1M 硬需求），
  先确认镜像为 v1.1.0 + 配方 v1.5.1 默认（`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` +
  `GLM53_INDEXER_WORKSPACE=rightsize`，合计还 KV ~7 GiB）；仍不足再调高
  `GPU_MEMORY_UTILIZATION`（≥0.90）；过高会对 MM/长 prefill 峰值失去余量。
- 冷启动慢，健康检查 start_period 900s；CUDA graphs 已开，勿 `--enforce-eager`。
- 容器启动会先执行镜像内运行时 overlay 补丁（含禁用 GB10 `persistent_topk`、xgrammar 投机解码终止修复、
  spinwait/indexer-workspace/ablit 等），与上游 start.sh 一致；补丁一律 `if [ -f ]` 哨兵执行，缺文件自动跳过。
  `ABLIT` 默认 0（stock 权重），`patch_ablit.py` 幂等、每次启动都会应用。
- **镜像侧记录（2026-09-02 ACR 重建）**：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-exl3:v1.1.0`
  按上游现行 Dockerfile 烘焙（`@c707598` / 2026-09-01，digest `sha256:06035a0d…`），运行时补丁
  `suppress_stops_in_reasoning` / `scheduler_decode_floor` / `hybrid_prefix_hit` / `xgrammar_termination` /
  `kpool_tail_slotmap` / `spinwait` / `indexer_workspace` 全部入镜像并在构建期跑了自检（每个 `ok`）；
  **exl3_fat_gemm（E2 内核）构建期编译进 exllamav3_ext**（构建断言 `exl3_fat_gemm`/`exl3_fat_gemm_scatter`
  存在）；exllamav3_ext 以 `sm_121a` cubins 编译。旧 ACR `:v1.0.0`（@493cb88）与 GHCR `:exl3`
  （2026-08-28 07:46Z 构建）都缺 E2 内核 + spinwait/indexer 补丁，勿再作为部署镜像。
- **KV per-token 自检**：正常部署每 token 目标 KV ≈ 9.3 KB（fp8_ds_mla 656 B/token/层 × 11 DSA 层 + 草稿 ~2 KB），
  1M 约需 9–12 GiB。若 boot 报错按 786,432 token 需要 16.34 GiB 这类量级（≈22.3 KB/token）反推，
  说明部署的镜像/`--kv-cache-dtype fp8` 没生效（或走了 bf16/旧版镜像），先按上面核对镜像再调 util。
- NCCL 必须用 CX7 直连口（auto 键按节点填），否则 `ncclCommInitRank` 悬挂。

## 参考来源

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)：
  参数级参考（.env / start.sh / overlay / Dockerfile；2026-08-29：1M 上下文 + padded slot-share + hybrid prefix hit +
  MNBT=2048 P1 keep；2026-08-30：`DFLASH_DRAFT_TP=2` keep + kpool-tail slotmap 修复（`overlay/patch_kpool_tail_slotmap.py`，
  长生成时通用 paged kernel 会越过 tail group 单 block-table 条目、可崩溃或静默损坏 indexer——修复 pin 在单 block 环形 scratch，
  随镜像烘焙生效）+ per-rank GID（与我们按节点 gid_index 自动键同思路）；2026-09-01：`EXL3_FAT_KERNEL=1` + MNBT=7168
  （#77 E2 fat-expert prefill 内核，构建期编译）+ spinwait 数值化（#96）+ indexer workspace 右尺寸（#86，`rightsize`
  省 ~5 GiB KV）+ 实验性 TP4 线（#105，4× GB10，本配方仍 TP=2））
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)：
  EXL3/TR3 4bpw 权重镜像（ShapleyMCG License 1.0）· [原始权重](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿（CC BY-NC-ND 4.0）
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)
