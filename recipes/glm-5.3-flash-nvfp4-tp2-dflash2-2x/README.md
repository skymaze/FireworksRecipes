# GLM-5.3-Flash NVFP4 · TP=2 · DFlash2 · 262K · Fireworks 配方（2× DGX Spark）

在 **2 台** DGX Spark（head + 1 worker，双 rail RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next`）的 **fp8 KV + DFlash2 块扩散投机解码** 服务——上游姊妹仓库
`GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` 的 proven & reproducible 档位（README 标
「This is the config to copy」）。

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2`
  （ACR，**与 TP4 DFlash2 配方同一镜像**）：按上游公开 GHCR 链
  `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`（day-0 + **v1→v9 九层 patch 栈** +
  **DFlash2 overlay**）烘焙，另烘焙 **SM121 `sparse_attn_indexer_kpool` 模块**（上游 launcher
  运行时 bind-mount 的同一文件，平台无法分发主机文件，故打入镜像）与 **mm chat 模板**。
- 主模型（**默认**）：[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
  ——**compressed-tensors**，修复 ModelOpt 构建的**间歇性 token 损坏**（vLLM #54150）；
  drop-in、零 flag 改动、加载快 ~2×。ModelOpt 版仍可用但带损坏：censored
  `LibertAIDAI/GLM-5.3-Flash-NVFP4`（120 分片 ~182 GiB）、uncensored
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock`
- 草稿模型：`incoai/GLM-5.3-Flash-DFlash2`（2.34 GB，qwen3 架构 5 层 SWA）
- 拓扑：**2 节点 · TP=2**（`--tensor-parallel-size 2`）、`mp` 后端
- 上下文：**262,144（262K）**（TP2 每 rank ~97 GiB 权重，勿上 1M；要 1M 用 4x 配方）
- KV：**fp8_e4m3 · 让 profiler 定池（不传 `--kv-cache-memory`）= 581,040-token 池**（上游验证）
- 批处理：**`--max-num-batched-tokens 8192`**（上游 2026-08-30 pin，与 TP4 同档：投机模式下
  vLLM 会静默推导 2048 限并发吞吐，8192 在 TP4 实测 C4 并发 63→99 tok/s（+57%）且单流
  不变；flag 与拓扑无关，TP2 同享）
- 投机：`--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  （必须 7 = block_size−1）
- 实测（上游 2026-08-28，warm）：**单流 46.9 tok/s · 74.1% 接受率**（结构化输出
  **54–61 tok/s**）；C1–C6 并发扫描**零失败**：聚合 35.1 / 41.6 / 40.6 / 47.5 / **56.2**(C5) /
  47.7；= **2.15×** MTP-4（21.8 tok/s）

> **TP2 这条线的硬教训：完全别 pin `--kv-cache-memory`。** 传了之后 vLLM 仍然跑 profile 但
> **从不扣测得的激活峰值**（`--gpu-memory-utilization` 失效）——分配、预热、短答全过，然后
> **第一个长 prompt 没有任何地方放得下激活，引擎当场死**（上游在 4 个不同 pin 上复现）。
> 老 launcher 仍带着 3 GiB pin（`3221225472`）是**滞后残留**；本配方默认
> 让 profiler 定池（DFlash2 + fp8 KV @262K = **581,040-token 池**，28,818-token 深 prompt
> 存活、引擎健康）。

## 快速开始（发布前就绪）

- 集群：恰好 **2 台**节点（head + 1 worker），双 rail RoCEv2 已配置测试。
- 镜像：`…/glm53-flash-sm121:v11-dflash2`（ACR，已推送；与 TP4 DFlash2 配方同镜像）。
- 模型：主模型 + drafter 两个都由 Fireworks 以 `picker=model` 变量分发到节点 HF 缓存
  （drafter 仅 2.34 GB）；容器内 `HF_HUB_OFFLINE=1` 按 repo id 离线解析。
- **NCCL**：HCA / 网卡 / GID index 由 Fireworks 自动键按节点填充。
- **启动顺序**：Fireworks 发布时自动 worker-first、head 最后。
- **内存仪式（host 侧，每节点）**：`vm.swappiness=0`（写进 `/etc/sysctl.d/` 持久化；swap
  可以存在但**绝对不能有 swappiness**，否则 UVM 驱动活锁冻死分片加载）、
  `sync; echo 3 > /proc/sys/vm/drop_caches` 启动前两节点都做一遍。
- **健康签名**：启动日志应有 `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
  与 `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`；KV
  行应显示 **`GPU KV cache size: 581,040 tokens`**（各机数字会有出入；池取各 rank **最小值**，
  `Available KV cache memory` 只有 rank 0 打日志——**每台都读**）。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v11-dflash2` | 平台镜像（与 TP4 DFlash2 同镜像；九层 patch + DFlash2 overlay + SM121 indexer 模块已烘焙） |
| `GLM53_MODEL_PATH` | `RedHatAI/GLM-5.3-Flash-NVFP4` | 主模型（compressed-tensors 默认；ModelOpt censored/uncensored drop-in 可选但带 token 损坏；或缓存内 snapshot 绝对路径） |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（务必分发到节点缓存） |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | 对外服务名 |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `262144` | **上游已验证档位**（TP2 每 rank ~97 GiB 权重，勿上 1M） |
| `MAX_NUM_SEQS` | `6` | 与上游一致 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 上游 2026-08-30 pin（与 TP4 同档）：不设会静默推导 2048 限并发；TP4 实测 C4 63→99 tok/s（+57%）、单流不变 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 上游生产值（0.78–0.80 在 131K+ 饿死 KV 池）；配合 profiler 定池 |
| `KV_CACHE_MEMORY` | （空） | **留空 = 不传 flag，profiler 定池（581,040-token）——上游推荐**；填字节数才传 `--kv-cache-memory`（勿 pin，见上） |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **必须 = block_size−1**；K=7 最优别扫 |
| `CHAT_TEMPLATE` | `/opt/glm53/chat_template_mm.jinja` | 镜像内烘焙 mm 模板，Vision 默认可用（不投机）；填空 = text-only |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 2）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意（任务/项目命名）

任务名即 docker compose 项目名：只允许**小写字母/数字/`-`/`_`，不能含点 `.`**（节点
Docker Compose v5 硬性限制）。带点任务名（如 `glm5.3-flash-2x`）会发布失败（502）。
建议用如 `glm53-dflash2-tp2` 的无点任务名。

## 部署注意（源自源仓库实机踩坑）

- **KV 定池纪律（TP2 头号教训）**：让 profiler 定池，别 pin——pin 后激活峰值不扣、首个长
  prompt 即 NVRM OOM（上游 4 个不同 pin 都复现；本仓库 launcher 的 3 GiB pin 是滞后残留）。
- **drafter 的隐藏成本**：DFlash2 吃掉 ~4.8 GiB KV headroom（远超其 2.2 GiB 权重），换来
  ~+91% decode、−40% 池——按负载取舍；不带 drafter（MTP-4 v8 镜像）池 965,166-token。
- **读 KV 数字的坑**：`GPU KV cache size` 头衔数字 = `max_concurrency × max_model_len`，
  随上下文膨胀、不反映真实字节；跨配置只能比 `blocks × block_size`。`Available KV cache
  memory` 只由 rank 0 打日志，但池从各 rank **最小值**构建——每台都读。TP worker 比 head
  少 4–5 GiB KV headroom（上游未解释）。
- **checkpoint 别用 ModelOpt 版做生产**（token 损坏，vLLM #54150）；要 uncensored 目前只有
  ModelOpt ablit 可选（带损坏）。
- `num_speculative_tokens` 必须 = 7（block_size−1）；K=7 最优别扫（末位仍 0.94 条件接受率）。
- drafter 只做文本草稿（vision 请求仍可用、不投机）。
- 冷启动 JIT 需预热（`_prepare_dflash_inputs_kernel`、`mhc_pre_big_fuse_with_norm_tilelang`）；
  健康签名 `Using Eagle3 auxiliary layers … (6,15,25,34,43)`、acceptance 0.6–0.8
  （~0.15 = aux 捕获/mHC 收缩错了，静默退化）。
- **`temperature: 0` 白嫖吞吐（+13–21%）**；`enable_thinking: false` 也更快（+8%）——注意
  thinking 关掉时 GLM 会把未标注的推理散文写进 `content`，某些 agent 解析器会误读。
- 不要改：`--block-size 2304`、`--moe-backend marlin`、`--kv-cache-dtype fp8_e4m3`、
  `--enforce-eager`、`GPU_MEMORY_UTILIZATION` 默认（0.78–0.80 饿死 KV）、
  `MAX_NUM_BATCHED_TOKENS` 默认 8192（同 TP4 上游 pin）。
- 内存纪律：`vm.swappiness=0` **强制**且**重启不保留**（要持久化）；swap 完全关掉会让 worker
  在 MoE marlin repack 时被杀、默认 swappiness 会让 UVM 驱动活锁。
- 起停纪律：先全部 teardown 再重启任一 rank；发布核对各节点 `IMAGE` 一致；`docker logs`
  先于 `rm -f`；对存活探测用 `/health`，**别用 `/v1/models`**（配置阶段就返回 200）。
- **SM121 indexer 模块（disease 1）**：没有它任何 decode 过 ~24K 上下文即崩；本配方镜像已
  烘焙（上游 launcher 运行时挂 `sparse_attn_indexer_kpool.py`，同一文件）。
- C1–C6 是上游 TP2 实测；Fireworks 侧建议真机复测后回填自己的数字。

## 参考来源

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)：
  姊妹仓库（TP2 变体）——`launch-glm53-vllm-tp2-dflash2.sh`、`docs/BENCH-C1-C6-DFLASH2.md`、
  `docs/GB10-KV-MEMORY-LADDER.md`、`docs/SM121-CRASH-FORENSICS-2026-08-27.md`、
  `docs/KV-HUNT-672K-TP2-RECORD.md`、`docs/OPEN-PROBLEMS.md`
- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：4x 主仓库（同镜像、TP4/1M）
- [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)：compressed-tensors 主模型（默认）
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿模型
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎（`glm5_next` PR #53906、DFlash2 上游 PR #52816、token 损坏 issue #54150）
