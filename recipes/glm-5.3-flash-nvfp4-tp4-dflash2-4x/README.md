# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上跑 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next`）的 **TP=4** 服务——**上游当前默认配置**：fp8 KV + **DFlash2 k=7 块扩散投机**，
**1M 上下文、3,895,606-token KV 池（3.72× 满 1M 请求）**：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`（ACR）
  = 按上游公开 GHCR 链 `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` 烘焙（day-0
  `glm53-flash-arm64-cu130` + **v1→v9 九层 patch 栈** + **DFlash2 overlay**）：NoPE-MLA FA2
  SM121 unlock、FlashInfer 0.6.18 nightly、NCCL 2.30.7 / cutlass-dsl 4.6.2 回钉、PDL 关闭、
  indexer top-k 初始化、fp8 KV smem tile、InstantTensor（未启用）、DFlash2 drafter/aux 捕获/
  KV 组 slot-share。另在镜像内烘焙 **SM121 `sparse_attn_indexer_kpool` 模块**（上游由 launcher
  运行时 bind-mount 的同一文件，平台无法分发主机文件，故打入镜像）与 **mm chat 模板**。
- 主模型（**默认**）：[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
  ——**compressed-tensors** 量化，修复 ModelOpt 构建的**间歇性 token 损坏**（vLLM #54150）；
  drop-in、零 flag 改动、加载快 ~2×（11 大分片 vs 120 小分片）。ModelOpt 版仍可用但带损坏：
  censored `LibertAIDAI/GLM-5.3-Flash-NVFP4`、uncensored
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- 草稿模型：`incoai/GLM-5.3-Flash-DFlash2`（单个 `model.safetensors` 2.34 GB，qwen3 架构 5 层
  SWA，`block_size=8` / `selector_rank=256` / `target_layer_ids [5,14,24,33,42]`）
- 并行：**TP=4**（`--tensor-parallel-size 4`）、`mp` 后端、4 台每机 1 GPU
- KV/形状：**fp8_e4m3**、每 rank **24 GiB 预算**（`--kv-cache-memory 25769803776` =
  **3,895,606-token 池**）、`--block-size 2304`、gmu 0.85、`--max-num-seqs 6`、
  **`--max-num-batched-tokens 8192`**（不设会从投机设置推导 2048 并告警 suboptimal）、
  **1,048,576 上下文**、端口 `8000`
- 投机：`--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  （**必须是 7 = block_size−1**）；drafter 层与 MLA 张量 **slot-share，KV 池成本 ~0**
- 实测（上游 2026-08-29 gate 套件通过）：**单流 54.5 tok/s**（n=1：408 tokens/7.5s，code
  prompt，temp 0，thinking off；**引用前请带 prompt**——接受率由内容决定，结构化/代码 ~0.70+
  vs 自由文本 ~0.33）、**prefill 4,141.8 tok/s**（warmed，单样本；冷首 prefill ~467 tok/s，
  核 JIT）；gate = 2× ~41K 深度解码（392/399 tokens）+ 3× 并发 32,879-token prefill + vision
  + `/health` 全程 200；余量：head 15 GiB、worker 19–20 GiB

> **无条件 flusher 是这套 24 GiB 配方的全部戏法。** 上游曾以为是 GB10 的 "phantom KV
> backing"（>16 GiB 预留成功、实载下触碰即故障）——**其实是阈值触发式 page-cache 刷新的锅**。
> 无条件 flusher（每节点、启动前起、全程跑）让同一个 24 GiB pin 直接通过 gate 套件，
> 池 +54.8%（2,516,582 → 3,895,606 tokens）。**发布前每节点必须跑**（见部署注意）。

## 快速开始（发布前就绪）

- 集群：**4 台**节点（head + 3 worker），双 rail RoCEv2 已配置测试。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`（ACR，
  已推送；上游 GHCR 原版约 31 GiB，Fireworks 拉取后分发，节点别并发四台各拉）。
- 模型：主模型 + drafter **两个**都经 `picker=model` 分发到节点 HF 缓存（drafter 仅
  2.34 GB）；容器内 `HF_HUB_OFFLINE=1` 按 repo id 离线解析（`HF_HOME=/cache/huggingface`）。
- **NCCL**：HCA / 网卡 / GID index 全部由 Fireworks 自动键按节点填充。
- **启动顺序**：Fireworks 发布时自动 worker-first、head 最后。
- **内存仪式（每节点，host 侧）**：`vm.swappiness=0`（写进 `/etc/sysctl.d/`，重启不丢）、
  `swapoff -a && swapon -a`、`sync; echo 3 > /proc/sys/vm/drop_caches`，然后
  `setsid nohup ./flusher-unconditional.sh > flusher.log 2>&1 &`（需要无密码 sudo）**整个启动
  期间一直跑**，服务就绪后 `pkill -f flusher-unconditional.sh`。
- **健康签名**：启动日志应有 `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
  与 `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`；
  KV 行应显示 **`GPU KV cache size: 3,895,606 tokens, Maximum concurrency for
  1,048,576 tokens per request: 3.72x`**（各机数字会有出入，见部署注意的"测你自己的上限"）。

> 上游 Quickstart 里还有两个"必须在每节点存在"的源仓库文件，本配方已改为镜像内烘焙：
> **(c) SM121 indexer 模块**（`docker/sparse_attn_indexer_kpool_sm121.py`）与 **(d) mm 模板**
> （`chat_template_mm.jinja`）。其余（主权重、drafter）由平台分发。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v11-dflash2` | 平台镜像（九层 patch + DFlash2 overlay + SM121 indexer 模块已烘焙） |
| `GLM53_MODEL_PATH` | `RedHatAI/GLM-5.3-Flash-NVFP4` | 主模型（compressed-tensors 默认；ModelOpt censored/uncensored drop-in 可选但带 token 损坏；或缓存内 snapshot 绝对路径） |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（第二个模型，务必分发到节点缓存） |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | 对外服务名（drop-in 名） |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 模型原生 1M；降低（如 300000）更跟手，须 64 对齐 |
| `MAX_NUM_SEQS` | `6` | 与 Lane A 一致 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 与固定 `KV_CACHE_MEMORY` 搭配 |
| `KV_CACHE_MEMORY` | `25769803776` | 每 rank fp8 KV 预算（**24 GiB = 3,895,606-token 池**，上游当前默认、gate-passed）；无条件 flusher 前提；加大必须真实长 prefill 把关 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 上游 pin；取消会让 vLLM 推导 2048 并告警 |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **必须 = block_size−1** |
| `CHAT_TEMPLATE` | （空） | 容器内模板路径；镜像已烘焙 mm 模板（填 `/opt/glm53/chat_template_mm.jinja` 开 Vision） |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 4）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意（任务/项目命名）

任务名即 docker compose 项目名：只允许**小写字母/数字/`-`/`_`，不能含点 `.`**（节点
Docker Compose v5 硬性限制）。`glm5.3-flash-nv` 这类带点任务名会在发布时报
`invalid project name ...`（502）。请用如 `glm53-flash-dflash2-4x` 的无点任务名。

## 部署注意（源自源仓库实机踩坑）

- **无条件 flusher，没有替代**：阈值触发的 flusher（`Cached > 40GiB` 才 drop）会低于阈值还
  一直饿着 NVRM 分配器——这正是同一 pin 时好时坏的原因。必须每节点、先于 launcher 启动、
  全程运行到服务就绪。**这也是 24 GiB 默认值成立的前提**。
- **测你自己的上限，别直接抄 24 GiB**：这是上游机队的落点（head 余量仅 15 GiB、worker
  19–20 GiB；其他运营者同硬件跑 28/32 GiB）。**head 永远是最紧约束**（API server + 引擎核心
  在 shard 之上）；启动空闲内存各机差几个 GiB。每加一档都用真实长 prefill + 并发把关，
  **能 boot 并能答短 prompt ≠ 能干活**。
- **checkpoint 别用 ModelOpt 版做生产**（token 损坏，vLLM #54150：工具调用块内坏 token 会让
  `glm47` parser 失步、生成旋进重复锁；同机探测 U+FFFD 4/9/8 vs RedHatAI 0/0/0）。要
  uncensored 目前只有 ModelOpt ablit 可选（带损坏，等 compressed-tensors ablit 出现）。
- **`num_speculative_tokens` 必须 = block_size−1 = 7**：drafter 按 8 的 block 训练，末位是
  目标模型已验证 token；填 8 会草稿一个模型从未学过的位置。
- **drafter 只做文本草稿**：vision 请求仍可用，只是不投机（日志提示 "does not support
  external multimodal embeddings"）。
- **冷启动 JIT 需预热**：首个请求 JIT 编译 `_prepare_dflash_inputs_kernel` 与
  `mhc_pre_big_fuse_with_norm_tilelang`，冷测 ~10 tok/s 偏低，量测请先跑热。
- **健康签名**：启动日志应有 `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
  与 `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`；
  acceptance（`spec_decode_num_accepted_tokens_total ÷ num_draft_tokens_total`）应 ~0.6–0.8，
  掉到 ~0.15 说明 aux 捕获或 mHC 收缩错了（**静默退化、不崩溃**）。单流 tok/s 是**对 prompt
  的陈述**——结构化/代码 ~0.70+、自由文本 ~0.33；引用 54.5 时请带上 prompt。
- **SM121 indexer 模块（disease 1）**：没有它任意 decode 过 ~24K 上下文即崩
  （`persistent_topk` smem 墙，SM121 只有 99KB/block 而 `FilteredTopK` 需 128KB）；本配方
  已烘焙进镜像（上游 launcher 运行时挂 `sparse_attn_indexer_kpool.py`，同一文件）。
- **不要改**：`--block-size 2304`、`--moe-backend marlin`、`--kv-cache-dtype fp8_e4m3`、
  `--enforce-eager`、`KV_CACHE_MEMORY` 默认（GB10 上更大池并发 prefill 会 NVRM OOM，每次
  加大必须用真实长 prefill 把关）、`MAX_NUM_BATCHED_TOKENS` 默认 8192。
- **起停纪律**（源 hard-won rules）：先全部 teardown 再重启任一 rank（新 rank 与垂死 rank
  rendezvous 会挂死）；每次发布**核对各节点 image sha256 一致**（tag 相同不代表镜像相同，
  四台各自构建就是四台不同镜像）；先 `docker logs` 后 `docker rm -f`；gate 必须长 prompt +
  **长回答**（≥100 token，prompt 每次换，否则前缀缓存把 30s gate 变成 2s 空转）。
- **自愈**：上游 `fleet_watchdog.sh`（3 连败 teardown → 内存仪式 → 重启），恢复 ~15 min，
  忙时端点再考虑接。

## 参考来源

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：
  本配方（上游当前默认）的源部署：`launch-glm53-tp4-24g.sh`（worker-first 3→2→1→head 0）、
  `flusher-unconditional.sh`、`fleet_watchdog.sh`、`docker/`（v1→v9 patch 栈）、
  `overlay-dflash2/`、`docs/DFLASH2-SPECULATIVE-DECODING.md`、`docs/GB10-KV-MEMORY-LADDER.md`、
  `docs/SM121-CRASH-FORENSICS-2026-08-27.md`（gate 套件）
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)：
  compressed-tensors NVFP4 主模型（默认，修复 ModelOpt token 损坏）
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿模型
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎（`glm5_next` 支持 PR #53906、
  DFlash2 上游 PR #52816、token 损坏 issue #54150）
