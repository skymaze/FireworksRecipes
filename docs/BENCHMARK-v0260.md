# Benchmark —— FireworksRecipes v0.26.0 主路径 · 真机双节点（2026-08-05）

## 环境
- 2× DGX Spark（GB10 / sm_121a / 128GB 统一内存 / aarch64）：2 节点 head rank0 + worker rank1（内网段，地址略）
- 镜像：`fireworks-models/deepseek-v4-flash-0731:0.3.0`（主流 vLLM v0.26.0 + Anemll 式 GB10 overlay + flashinfer 0.6.15 + b12x 0.15.3 + fw-warmup 补丁 + JIT 缓存 bake）
- 部署：TP=2 / mp 分布式 / dspark 投机 MTP=5 / KV=`nvfp4_ds_mla` / block=256 / `--load-format instanttensor` / max_model_len=1048576 / gpu-mem-util=0.88 / RoCE 4×100G
- 口径：**同 harness**（`scripts/bench/`）——decode 聚合输出 token/墙钟；prefill server tokens/s=prompt_tokens/TTFT，**随机头破前缀缓存**后为真实 prefill

## 结果

### decode（128 output tokens/请求，聚合 tok/s）

| 并发 | 基线 0.1.0（fork 栈，同 harness） | **v0.26.0 auto(DeepGEMM)** | v0.26.0 b12x(flashinfer_b12x) | Anemll 公开（参考口径） |
|---|---|---|---|---|
| 1 | 46.4 | 33.8–40.9 | 37.1–41.7 | 48.5 |
| 2 | 65.6 | 44.4–55.6 | 48.2–54.4 | 70.4 |
| 4 | 75.0 | 63–85* | 83–85.4 | 103.5 |

*conc4 部分请求自然 EOS 提前结束（temp=0 短故事）导致 token 数不足 128，读数波动。

### prefill（真实 prefill，破缓存；server tok/s = prompt_tokens/TTFT）

| 输入 | 基线 0.1.0（首次/波动） | **v0.26.0 auto(DeepGEMM)** | v0.26.0 b12x | Anemll 公开 |
|---|---|---|---|---|
| 2K | 3859 | 1622–2252 | 69 | 2252 |
| 8K | 4659 | 804–2296 | 86–88 | 2184 |
| 16K | 4480 | 2109–2374 | 88 | 2204 |

**关键结论：b12x(flashinfer_b12x) MoE 在 v0.26.0 集成下的 prefill 路径不可用（~88 tok/s，GPU 96% 满载仍慢）；auto → DeepGemmFP4Experts，prefill 稳定 2200+ tok/s（与 Anemll 持平）。**

### 端到端
- `/v1/models` 健康检查 200；中文 chat（thinking）真实生成 ✓；长时运行 0 崩溃（护栏修复后）
- 首次 chat/decode 正常；`--enable-flashinfer-autotune` 使 SM120 sparse-MLA decode 自动调优（缓存复用）

## 与验收对照
- **≥ Anemll 70%**（decode ≥34 @conc1）：✓（33.8–41.7）
- **prefill 靠近标杆**：✓（2200+ vs Anemll 2000–2300）
- 相对 fork 基线（46.4/65.6/75.0）decode 略低（~10–25%），prefill 同量级——v0.26.0 主路径与 Anemll 参考处于同一性能包络；fork 是此前性能峰值，主路径刚完成首个真机验证周，仍有调优空间（dspark 参数、capture size、DeepGEMM decode 配置等）

## 调试历程（记录在案）
1. 原生 v0.26.0 fp8_ds_mla 在 block=256 时 SM120 decode 崩（页块矛盾）→ flashinfer 0.6.15 + Anemll attention overlay（git 三方合入 0 冲突）
2. 部署成功后在 decode 首请求遇空段 `[0,-1]` reshape 崩溃 → overlay 空 chunk 护栏（`query_end<=query_start: continue`）
3. b12x prefill 88 tok/s（GPU 满载）→ 隔离实验（dspark off / b12x off / auto）→ 定为 b12x prefill 路径问题 → 最终配置 auto(DeepGEMM)

---

## 追加：1M / NVFP4 KV 配置（2026-08-06，参考 jvr0x/dgx-spark-bench + tonyd2wild 双节点配方）

**配置 A（最终定案）**：`--kv-cache-dtype nvfp4_ds_mla --max-model-len 1048576
--max-num-seqs 6 --max-cudagraph-capture-size 36 --gpu-memory-utilization 0.88
--block-size 256` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（regular）+ MoE=auto(DeepGEMM)
+ dspark MTP=5。真机 TP=2 同口径。

| 指标 | 配置 A（nvfp4+1M） | 此前（fp8+350K） | 参考：ref1(Anemll0.25+b12x) | 参考：ref2(v0.24+Patch4) |
|---|---|---|---|---|
| decode 512-token 固定 @1/2/4 | **41.2 / 94.4 / 151.2** | 34–41 / 44–56 / 63–85 | 46.9@1(强制128) / 123.7 agg@6 | 55.4 mean / 66.1 peak |
| per-session 数数 | **54.8** | — | 95.5（自然停止峰值） | 55.4 |
| per-session 中文 prose | **34.1** | — | — | 37.8 |
| prefill 2K/8K/16K | 1322–2342 | 2200+ | — | 2639@100K |
| prefill 128K / 256K | **1566 / 1289**（实测可用） | 不支持 | 1M 池 | 1.5M 池 |
| KV pool / 上下文 | **1M 上下文就绪** | 350K | 203 万 token | 154 万 token |

要点：
- **nvfp4_ds_mla 数据路径由 Anemll attention overlay 提供**（"padded nvfp4_ds_mla KV
  cache format"），dtype 枚举/归组此前已打通，本次直接启用。
- 1M 上下文需 `GPU_MEMORY_UTILIZATION≥0.88`（DeepGEMM 栈权重占用；ref1/ref2 的 0.80
  是在 b12x 栈上）。
- **DSpark shared-expert 丢失 bug（ref2 Patch4）在 v0.26.0 已天然修复**：checkpoint
  张量名 `shared_experts.w1/w3`（实证 276 张量）可被 v0.26.0 dspark.py 的 stacked
  映射（`gate_up_proj→w1/w3`）匹配加载；decode 55@1 与 ref2 修复后持平佐证。
- 流式 ITL 测量需按 usage 计数（chunk 聚合多 token）；`bench_session.py` 已修正。

## 追加：prefix caching 命中验证（2026-08-06）

现象：`usage.prompt_tokens_details` 恒为 `null` —— **非缓存失效**：v0.26.0 默认
关闭该统计返回，需 `--enable-prompt-tokens-details`（本仓库已默认开启）。

实测（8K 固定 prompt，随机头破首轮缓存）：
| run | wall | prompt_tokens | cached_tokens |
|---|---|---|---|
| 1（冷） | 9.61s | 8037 | 0 |
| 2 | 6.35s | 8037 | **7936（98.8%）** |
| 3 | 0.33s | 8037 | 7936（全命中，仅算尾部 101） |

结论：prefix caching 行为与统计均正常；短 prompt（<block）首次 cached=0 属预期
（block 级缓存，第二次同 prompt 才命中）。

## 追加：200K 长上下文故障根因与修复（2026-08-06）

现象：200K 请求极慢甚至失败（服务崩溃、连接拒绝）；重启后重复请求"没有命中缓存"。

根因（日志定位）：
1. **推理期 JIT 编译**：200K 长 prefill 触发多个新形状内核的 TileLang/Triton JIT
   编译（`mhc_pre_big_fuse_broadcast_with_norm_tilelang` 等，单次编译 4-90s+），
   期间 worker 卡住 sample_tokens。
2. **RPC 超时**：`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 默认 300s —— JIT 卡顿撞超时
   → `TimeoutError: RPC call to sample_tokens timed out` → EngineCore 被杀 → 连接拒绝。
3. **JIT 缓存不持久**：TileLang/Triton 默认缓存容器内 /root（重启即失），每次重启
   或新形状都要重编译。
4. "没命中缓存"：KV prefix cache 为内存态，崩溃/重启即失；首次请求本就 cold。

修复（已固化到部署脚本/模型 ENV/recipe）：
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`（防 JIT 期间被杀）
- `TRITON_CACHE_DIR=/cache/vllm-cache/triton`、`TILELANG_CACHE_DIR=/cache/vllm-cache/tilelang`
  （JIT 编译缓存持久化到挂载盘，重启不重编译）
- `TILELANG_CLEANUP_TEMP_FILES=1`

实测（200K prompt，同 prompt 连发）：冷 112.4s（HTTP 200，含首次 JIT）→
热 16.2s（cached=199936/200036，99.95% 命中）。

## 追加2：推理期 JIT 编译 OOM 崩溃根因与修复（2026-08-06 二次崩溃）

现象：服务再次崩溃。head (.111) `docker logs` 显示 EngineCore `RuntimeError: cancelled`
（shm_broadcast 等 60s×3），journalctl 确认 **Linux OOM killer 杀掉 worker**：
```
oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=snapd.service,mems_allowed=0,
         global_oom,task=VLLM::Worker_TP,pid=1952195
Out of memory: Killed process 1952195 (VLLM::Worker_TP)
         total-vm:3435729684kB, anon-rss:3158104kB, shmem-rss:3813604kB
```
崩溃请求：`chatcmpl-8004b13e`（prompt 246,639 tokens / max_tokens 32,000，
前缀缓存命中 71.9% → 非连续 block 布局）。

根因链条（GB10 统一内存 121.63GiB、无 swap 环境）：
1. **新形状冷编译**：启动 warmup 只覆盖 `max_num_batched_tokens`(8192) 粒度的混合
   形状；长上下文 prefill 大块内核（`mhc_pre_big_fuse_broadcast_with_norm_tilelang`）
   按请求形状惰性编译。246K 命中前缀后的非连续 block 布局是**从未编译过的新形状**，
   首次触达在推理路径内触发 TileLang JIT（编译期 CPU 内存峰值 + nvrtc）。
2. **内存零余量**：`--gpu-memory-utilization 0.88` 预留 ≈107GiB（权重+KV+cudagraph），
   系统仅剩 ~11-13GiB 余量；JIT 编译峰值叠加后全局物理内存耗尽。
3. **无 swap 兜底**：两节点均无 swap，global_oom 直接杀进程 → worker 死 → EngineCore
   cancelled → API 关闭。

修复（已固化到 scripts/deploy/）：
- **两节点各加 32GiB swap**（`/swapfile`，fstab 持久化）——编译峰值可换出，OOM killer
  不再触发（硬兜底，实测峰值仅用 2-3GiB）。
- **`warmup_longctx.py` 长上下文 JIT 形状预热**（deploy_v0260.sh 默认执行，`WARMUP=0`
  跳过）：对 32K/64K/128K/256K 各发两轮——
  A) 全量随机头（连续 block 布局）；B) 复用前缀+~2K tail（命中缓存 → 非连续 block 布局，
  即真实崩溃场景），编译产物落盘到持久化 TILELANG_CACHE_DIR。
- 保持 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` 与 JIT 缓存持久化（见上）。

实测（2026-08-06 复现验证，均通过）：
- 预热 32K→256K：A 全量 25.1/41.7/70.9/151.3s；B 命中缓存 1.8/1.9/1.8/2.9s
  （同前缀命中提速 14~80 倍，证明非连续布局形状已编译并缓存）。
- 崩溃场景复现：前缀命中 262K+4K tail（266,145 tokens）145.2s 完成；
  **新形状冷编译 246K 全量（246,035 tokens）131.4s 完成**——即上次被杀场景，swap 兜底下通过。
- 服务健康，两节点 0 次 oom-kill，swap 峰值仅用 2-3GiB。

## 追加3：JIT 缓存 Bake 进镜像 + 启动 warmup 全覆盖（2026-08-06，v0.3.0）

### 研究结论（为何不需要"逐长度预热到 1M"）
1. **TileLang 内核 num_tokens 是动态轴**（`T.dynamic("num_tokens")`，grid 运行时
   发射；实测 128/129/262144 的 TIR 脚本相同）→ **单个编译覆盖 1..1M 全部长度**。
2. 真正需要覆盖的是**变体**（n_splits 等标量组合）：n_splits 随 batch token 数变化
   （decode 小 batch≈数十、8192 chunk=1、≤16 走 small-FMA）→ 全集合约 10-15 次编译。
3. **NVIDIA 路径 mHC warmup 原本是 no-op**（根因）：`_find_first_mhc_layer` 要求
   `hc_pre/hc_post` 属性（仅 AMD/XPU 有），NVIDIA `DeepseekV4DecoderLayer` 没有 →
   首层 broadcast 变体（`mhc_pre_big_fuse_broadcast_with_norm_tilelang`，即 246K
   崩溃内核）**从未被启动预热** → 首个长请求必触发推理期 JIT 编译。
4. 缓存可移植性好（同镜像同硬件命中）：tilelang key 含 lib 内容 hash（镜像内固定）；
   triton key 不含 shape；flashinfer autotune 按配置哈希+形状桶；torch_compile 按
   config hash+capture 形状集。

### 实施（v0.3.0）
- **fw-warmup 补丁**（models/deepseek-v4-flash-0731/patches/fw-warmup/，Dockerfile.model
  构建期 AST 校验+幂等应用）：
  - `deepseek_v4_mhc_warmup.py`：`_find_first_mhc_layer` 兼容 NVIDIA 层；新增 NVIDIA
    分支直接调 `mhc_pre_broadcast_tilelang`/`mhc_pre_tilelang`/`mhc_fused_post_pre_tilelang`/
    `hc_head_fused_kernel_tilelang`，token sizes = {1,2,4,...,8192}+cudagraph sizes
    → 启动即编译全部变体（实测 0.75-0.96s，种子命中后秒级）。
  - `flashinfer_sparse_mla_warmup.py`：`_SPARSE_MLA_MIXED_WARMUP_TOKENS` 16 → 8192。
- **warmup_longctx.py 扩展**：lens 默认 32K..1M（含 512K/1M）；max-ctx 截断防 400；
  B 复用 A 前缀子串保证前缀缓存命中。
- **缓存 bake 进镜像**：服务节点跑全量预热（32K..1M A/B 两轮）→ 收集
  vllm-cache（tilelang 7.6M + triton 35M + torch_compile 97M + flashinfer 144K ≈ 139M）
  → `SEED_CACHE_DIR=seed-cache bash scripts/build.sh model` bake 到
  `/opt/fw/vllm-cache-seed/`（非挂载路径）。
- **部署种子灌入**（幂等）：run_vllm_node.sh / recipe command 在 docker run 前用
  `cp -an /opt/fw/vllm-cache-seed/. /cache/vllm-cache/`（只补缺失，宿主已有不覆盖）。

### 实测（2026-08-06，清空缓存模拟全新节点）
- 部署：两节点清空 `/home/spark/.cache/vllm-cache` → 种子自动灌入（139M）→
  服务 225s 就绪。
- **mHC warmup：0.75s**（15 个 token sizes 全部免编译，种子命中；修复前为 no-op，
  修复后未带种子时 0.96s）。
- sparse MLA warmup：mixed tokens=8192（修复前 16）。
- **长上下文零编译停顿验证**（新节点首个请求，均为此前未触达的冷形状）：
  | 场景 | 结果 |
  |---|---|
  | 246K 前缀命中+4K tail（原崩溃场景） | 157.7s（≈1.56K tok/s，无编译停顿） |
  | 512K 冷形状全量 | 374.4s（≈1.4K tok/s） |
  | 1M 冷形状全量（1040038 tokens） | 1160.3s（≈0.9K tok/s） |
  - 本轮部署窗口 **0 次 oom-kill**（journalctl 确认；52 次历史 oom-kill 均为上次
    崩溃时间点 UTC 02:xx 的旧记录），swap 峰值仅用 3GiB 兜底。
  - decode 512@conc1 = 32.7 tok/s（正常量级）。

## 追加4：推理期 TileLang 编译链卡死（EngineCore 28 分钟无心跳）根因与修复（2026-08-06 v0.3.1）

### 现象
服务"崩溃"：请求长时间无响应。head 日志 Engine 000 心跳（每 10s 一条）出现
**18 分钟（06:09→06:27）与 28 分钟（07:31→07:59）的完全空洞**，期间 EngineCore
阻塞（stats 停止、无错误日志、无 OOM）；恢复后一切正常（自愈）。

### 根因（缓存 mtime + 日志关联定位）
1. **fw-warmup 覆盖缺口**：mHC 内核 `n_splits = compute_num_split(64, h,
   cdiv(num_tokens, 64))` 按 **grid = cdiv(num_tokens, 64) 分桶**（每 64 token 一桶）。
   启动 warmup 候选集只含 2 的幂（15 个值 → 仅覆盖 9/128 个桶）。
2. **长上下文 prefill 余数 chunk**：8192 分块后最后一段为任意余数（如 246K =
   30×8192+1567 → grid=25），落在未覆盖桶 → 触发该桶变体的 TileLang 编译。
3. **编译链阻塞**：tilelang 缓存 mtime 实证 06:08-06:27 连续写入 8 个新内核
   （`mhc_pre_big_fuse_broadcast_with_norm_tilelang` / `mhc_pre_big_fuse_with_norm_tilelang`
   等，单次 4-90s）→ 每个变体编译都在推理路径内同步执行 → EngineCore 步进被
   阻塞 → 心跳空洞（18 分钟）；多个新桶变体陆续编译 + 大请求叠加 → 28 分钟空洞。

### 修复（v0.3.1）
- fw-warmup 补丁 `_DEFAULT_TOKEN_SIZE_CANDIDATES` 替换为**每 64 token 一个代表**
  （135 个候选，覆盖 grid 桶 1..128 全部 + small-FMA 小值）→ 启动即编译全部
  n_splits 变体，推理期零 TileLang 编译。
- 重新生成 JIT 缓存 seed 并 bake 进镜像（含全部桶变体）。

### 实测（2026-08-06）
- 0.3.1 启动：warmup token sizes 135 个（1..8192 全覆盖），2.35s 完成（此前 15 个/0.75s）。
- 长上下文请求验证：见下方结果（追加4 验证节）。
