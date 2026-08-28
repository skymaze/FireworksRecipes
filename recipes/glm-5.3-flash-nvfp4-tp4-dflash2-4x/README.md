# GLM-5.3-Flash NVFP4 · TP=4 · DFlash2 · Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上跑起 **GLM-5.3-Flash**
（[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，320B / A18B MoE，
`glm5_next`）的 **Lane A（fp8 KV）+ DFlash2 块扩散投机解码** TP=4 服务：
上游在 GB10 上首个跑通的 **DFlash2** 部署：**TP4 单流 68.5 tok/s（0.641 接受率）**，
ladder = MTP TP2 21.8 → MTP TP4 35.7 → DFlash2 TP2 46.9 → **DFlash2 TP4 68.5**。

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2`
  = 上游 sm121-v8 构建链（day-0 + 8 层 patch，即 Lane A 镜像）+ **4-patch DFlash2 overlay**：
  1. `qwen3_dflash2.py` + `dflash2/` + registry/selection wiring（把上游 PR #52816 的
     DFlash2 drafter/选择器移植进本 vLLM 树，注册 `DFlash2DraftModel`）
  2. GLM aux hidden-state 捕获（`SupportsEagle3` 接口 + 5 个 tap 层 `(6,15,25,34,43)` + mHC 收缩）
  3. drafter KV 组**并入 GLM 自定义快速路径**，5 个 `SlidingWindowSpec` 层 slot-share MLA 张量
     （KV 池成本 ~0；通用路径页大小互斥会让单条 262K 请求膨胀 ~13×）
  4. `patch_kv_page_lcm2.py`（no-op 占位）
- 双模型（都按 HF repo id 分发到节点缓存、离线解析）：
  - 主模型 `LibertAIDAI/GLM-5.3-Flash-NVFP4`（120 分片 ~182 GiB，censored）
  - **DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2`**（单个 `model.safetensors` 2.34 GB，
    qwen3 架构 5 层 SWA，`block_size=8` / `selector_rank=256` / `target_layer_ids [5,14,24,33,42]`；
    默认加不加权；可换 uncensored 主模型 drop-in）
- 投机：`--speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}'`
  （**必须是 7 = block_size−1**）
- KV/形状：**fp8_e4m3**、每 rank **16 GiB 预算**（上游 TP4 DFlash2 实测 pin）、`--block-size 2304`、
  gmu 0.85、**1,048,576 上下文**、`--max-num-seqs 6`、端口 `8000`
- 实测（上游 2026-08-28）：
  - **TP4/1M，warm，code prompt：单流 68.5 tok/s · 0.641 接受率 · KV 池 2,622,494 token
    （2.5× 完整 1M 请求）**· 28.8K 深解码通过 · vision on（ablit 权重）
  - TP4 C1–C6 并发扫描（42 请求零失败）：聚合 55.2 / 52.3 / 59.7 / 84.4 / 85.2 / **100.1** tok/s
    （TP4 计算余量足以吸收草稿校验，聚合到 C6 仍爬升，不像 TP2 在 C5 见顶）
  - TP2/262K（姊妹仓库）：单流 46.9 tok/s · 74.1% 接受率；C5 聚合峰值 56.2 tok/s

> 与 Lane A（MTP-4）对比：DFlash2 单流约 **1.9×**（TP4：35.7 → 68.5 tok/s），且 KV 池成本 ~0
> （drafter 层与 MLA 张量 slot-share）。上游 TP4 DFlash2 扫描用 **16 GiB/rank KV pin** 达成
> 2,622,494-token 池，本配方默认即为其值；如确需更大 KV 池请用真实长 prefill 把关。

## 快速开始（发布前就绪）

- 集群：**4 台**节点（head + 3 worker），双 rail RoCEv2 已配置测试。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8-dflash2`（ACR，已推送）。
- 模型：主模型 + drafter **两个**都由 Fireworks 以 `picker=model` 变量分发到节点 HF 缓存
  （drafter 仅 2.34 GB，几分钟传完）；容器内 `HF_HUB_OFFLINE=1` 按 repo id 离线解析。
- **NCCL**：HCA / 网卡 / GID index 全部由 Fireworks 自动键按节点填充。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_IMAGE` | `…/glm53-flash-sm121:v8-dflash2` | 平台镜像（8 层 patch + DFlash2 overlay 已烘焙） |
| `GLM53_MODEL_PATH` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 主模型（HF hub 离线解析；或缓存内 snapshot 绝对路径） |
| `GLM53_DRAFT_PATH` | `incoai/GLM-5.3-Flash-DFlash2` | **DFlash2 drafter**（第二个模型，务必分发到节点缓存） |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | 对外服务名（drop-in 名） |
| `VLLM_PORT` | `8000` | API 端口 |
| `MAX_MODEL_LEN` | `1048576` | 模型原生 1M；降低（如 300000）更跟手，须 64 对齐 |
| `MAX_NUM_SEQS` | `6` | 与 Lane A 一致 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 与固定 `KV_CACHE_MEMORY` 搭配 |
| `KV_CACHE_MEMORY` | `17179869184` | 每 rank fp8 KV 预算（16 GiB = 2,622,494-token 池，上游 TP4 DFlash2 实测 pin）；drafter KV 成本 ~0，无需为此削减 |
| `DFLASH2_NUM_SPECULATIVE_TOKENS` | `7` | **必须 = block_size−1** |
| `CHAT_TEMPLATE` | （空） | 容器内模板路径；填 mm 模板可开 Vision（见 Lane A 说明） |
| `MASTER_PORT` | `29521` | 分布式主端口 |

`NODES_TOTAL`（固定 4）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。

## 发布注意（任务/项目命名）

任务名即 docker compose 项目名：只允许**小写字母/数字/`-`/`_`，不能含点 `.`**（节点
Docker Compose v5 硬性限制）。`glm5.3-flash-nv` 这类带点任务名会在发布时报
`invalid project name ...`（502）。请用如 `glm53-flash-dflash2-4x` 的无点任务名。

## 部署注意（源自源仓库实机踩坑）

- **缓存命中可见**：已开 `--enable-prefix-caching`（hybrid 模型默认开启）与
  `--enable-prompt-tokens-details`；API 返回 `usage.prompt_tokens_details.cached_tokens`
  可核对前缀缓存命中 token 数（深会话/agentic 场景命中率 ~100×，同前缀二次请求比对）。
**`num_speculative_tokens` 必须 = block_size−1 = 7**：drafter 按 8 的 block 训练，末位是
  目标模型已验证 token；填 8 会草稿一个模型从未学过的位置。
- **drafter 只做文本草稿**：vision 请求仍可用，只是不投机（日志会提示
  "does not support external multimodal embeddings"）。
- **冷启动 JIT 需预热**：首个请求 JIT 编译 `_prepare_dflash_inputs_kernel` 与
  `mhc_pre_big_fuse_with_norm_tilelang`，冷测 ~10 tok/s 偏低，量测请先跑热。
- **健康签名**：启动日志应有 `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
  与 `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`；
  acceptance（`spec_decode_num_accepted_tokens_total ÷ num_draft_tokens_total`）应 ~0.6–0.8，
  掉到 ~0.15 说明 aux 捕获或 mHC 收缩错了（**静默退化、不崩溃**）。
- 不要改：`--block-size 2304`、`--moe-backend marlin`、`--kv-cache-dtype fp8_e4m3`、
  `--enforce-eager`、`KV_CACHE_MEMORY` 默认（GB10 上更大池并发 prefill 会 NVRM OOM，
  每次加大必须用真实长 prefill 把关）。
- 起停纪律（源 hard-won rules）：先全部 teardown 再重启任一 rank；每次发布核对各节点
  `IMAGE` 一致；`docker logs` 在 `rm -f` 之前抓。
- TP2 上 KV 曾收紧到 3 GiB 保内存余量；**TP4 不要照抄**——本配方默认 16 GiB 即上游 TP4
  DFlash2 实测 pin（2,622,494-token 池）；drafter 仅 2.34 GB/rank，计入显存即可。

## 参考来源

- [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)：
  `docs/DFLASH2-SPECULATIVE-DECODING.md`（方法 + 九次启动失败阶梯）、`docs/BENCH-C1-C6-DFLASH2.md`、
  `overlay-dflash2/`（4-patch overlay 可直接复现）
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)：DFlash2 草稿模型
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：NVFP4 量化主模型
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎（DFlash2 上游 PR #52816）
