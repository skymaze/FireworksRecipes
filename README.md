# 🎇 FireworksRecipes

为 **Fireworks**（DGX Spark 集群管理工具）提供**每模型专属镜像**与**原生配方**的构建仓库。

> 与常见「单个 vLLM 镜像 + 运行时打补丁」的路线不同：本仓库为**每个模型构建一个专属镜像**，
> 把该模型所需的 vLLM 补丁与调优环境变量在**构建期烘培**进镜像；运行时不打补丁，
> Fireworks 拉取该镜像即可直接发布任务。
> 同时**不拉取任何第三方预编译镜像**——全部组件本地源码编译，脚本在本仓库内自主维护。

**English**: [docs/README.en.md](./docs/README.en.md)

当前模型：

| 模型 | 专属镜像 tag | 说明 |
|---|---|---|
| DeepSeek-V4-Flash-0731 | `fireworks-models/deepseek-v4-flash-0731:0.3.1` | 主流 vLLM v0.26.0 主路径 / Anemll 式 GB10 overlay / InstantTensor + dspark 投机 MTP=5 / 双节点 TP=2（KV=nvfp4_ds_mla · 1M 上下文 · MoE=auto/DeepGEMM）· **fw-warmup 补丁（NVIDIA mHC warmup no-op 修复）+ JIT 缓存 bake（`/opt/fw/vllm-cache-seed`，上线零推理期编译）** |
| DeepSeek-V4-Flash (DSpark) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`（分发镜像） | 双节点 TP=2 DSpark 服务（MiaAI-Lab 参考配方路线，FlashInfer b12x + dspark 投机 · NVFP4 DS-MLA · 1M 上下文）。由 Fireworks 原内置配方迁移而来。**固定 2 节点拓扑** |

> **拓扑固定**：所有配方均声明**确切的节点数**（manifest `nodes`/`tensor_parallel`，
> 如 2 节点 · TP=2），发布时必须恰好匹配——模型参数（KV/上下文/MoE）按该拓扑调优，
> 不再提供「≥2」的模糊空间；不同拓扑请用对应配方。

---

## 配方目录（Fireworks 直读）

本仓库同时是 **Fireworks 的配方源**：Fireworks 同步本仓库 → 读取目录清单 → 用户从
「配方商店」一键安装并发布，无需内置初始配方、无需升级 Fireworks 即可拿最新配方。

约定（新增/维护配方时遵守）：

- `recipes/index.json` —— **目录清单（manifest，等同 recipes.vllm.ai 的 /models.json）**：
  每条列出 `id / provider / model / path / readme / version / params / dtype /
  context_length / modality / topology / image / tags`。Fireworks 只读它，**不做整树扫描**。
- `models/<model>/recipe/fireworks.recipe.json` —— 可运行配方，字段对齐 Fireworks
  `POST /api/recipes/import` schema（`name/description/image/compose_template/variables`），
  并带 `version`（对齐专属镜像 tag）。
- `models/<model>/recipe/README.md` —— 介绍文档（用法 / 变量 / 性能 / 更新记录），
  Fireworks「配方商店」详情里渲染。
- **双向（可选）**：文本字段均可带英文并列字段 `xxx_en`（`name_en / description_en /
  label_en / help_en`）；README 可提供 `README.en.md`。Fireworks 按界面语言选择：
  English 优先取 `_en`，缺省回退主语言（zh）。只写一种语言也可（自动回退）。
- 详细字段规范见 [docs/RECIPE-FORMAT.md](./docs/RECIPE-FORMAT.md)（配方文件字段、
  variables 项、manifest、Fireworks 导入/导出本地化行为）。
- 配方变量模型跟随 Fireworks 当前版本：`MASTER_ADDR`=`cluster/head_roce_ip`（任务级
  head 的 RoCE IP）、`MASTER_PORT` 为 `user` 变量（默认 25000）、`NODE_RANK/HEADLESS/
  VLLM_HOST_IP/NCCL_*` 为 `node` 变量自动填充。

> 仓库大体积构建产物（`.build-ref/`、`dist/`、`logs/`、`seed-cache/`）均在
> `.gitignore`，公开克隆体积保持轻量，仅含配方/源码/文档。

---

## 目录结构

```
FireworksRecipes/
├── versions.conf        # ★ 全局版本锁（所有源码/依赖 pin，构建参数唯一来源）
├── LICENSE / NOTICE.md  # Apache-2.0 许可与第三方来源声明
├── scripts/
│   ├── build.sh         # 本地源码编译驱动（base / model / all）
│   ├── bench/           # 基准 harness：bench_decode.py / bench_prefill.py（同口径对比）
│   └── deploy/          # 真机双节点部署：deploy_v0260.sh（编排）+ run_vllm_node.sh（单节点）
├── docker/
│   └── vllm-b12x.Dockerfile   # 多阶段源码编译：flashinfer / vllm / deepgemm / nccl → runner
├── overlay/
│   └── vllm/            # ★ vLLM 源码层覆写（Anemll 配方定向移植，构建期 rsync 进源码烘培进 wheel）
├── docs/
│   ├── README.en.md           # 英文文档
│   └── BENCHMARK-v0260.md     # 真机双节点基准结果与对比
└── models/
    └── deepseek-v4-flash-0731/
        ├── build.conf              # 模型构建定义（镜像名 / base tag / 模型 id）
        ├── Dockerfile.model        # ★ 专属镜像层：烘培补丁 + 调优 ENV
        ├── patches/
        │   ├── hybrid-draft-loader/   # 自研补丁：目标 instanttensor / 草稿 lazy safetensors
        │   └── fw-warmup/             # 自研补丁：NVIDIA mHC warmup no-op 修复 + sparse MLA 覆盖
        └── recipe/
            └── fireworks.recipe.json   # ★ Fireworks 原生配方（可被仓库加载直读）
```

> 升级 vLLM 版本：改 `versions.conf`（版本锁）后，按需同步 `overlay/vllm/`
> 的源码覆写（版本漂移时 overlay 内锚点会硬失败提示），再 `./scripts/build.sh base && model`。

---

## 架构

### 1. 源码编译体系（`docker/vllm-b12x.Dockerfile`）

多阶段、全本地编译，**不依赖任何第三方镜像/预编译产物**：

| 阶段 | 内容 |
|---|---|
| `base` | CUDA 13 devel + 编译依赖 + PyTorch(cu130) + **NCCL 源码编译**(sm_121 gencode) |
| `flashinfer` | FlashInfer 源码编译（commit 0472b9b≈0.6.15；cubin 默认跳过走运行时 JIT） |
| `vllm` | 主流 vLLM v0.26.0（含 DeepGEMM）Rust 前端源码编译 wheel；**构建期 rsync `overlay/vllm/` 源码覆写烘培进 wheel** |
| `runner` | 运行镜像：安装全部 wheel + `ray / fastsafetensors / instanttensor` + b12x(SparkInfer) 源码安装 + NCCL 顺序修复 |

所有版本都集中在 `versions.conf`，脚本以 `--build-arg` 显式传入。

### 2. 专属模型层（`models/<model>/Dockerfile.model`）

在基础 runner 之上，构建期烘培：

- **补丁**：
  - `hybrid-draft-loader` —— 把「`--load-format instanttensor` 时，同 checkpoint 内嵌的
    dspark/MTP 投机草稿改为 lazy safetensors 加载」写入已安装的 vLLM wheel（AST 校验 + 幂等）。
  - `fw-warmup` —— 修复 NVIDIA 路径 mHC warmup no-op（启动即编译全部变体，消除推理期
    JIT 编译导致的 OOM）；sparse MLA warmup 覆盖提升到 8192。
- **调优 ENV**：GB10 主路径专用环境变量（`PYTORCH_CUDA_ALLOC_CONF` /
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` / `TILELANG_CACHE_DIR` 等，镜像内默认，配方
  environment 层可覆盖）。
- 元数据 `FW_MODEL_ID` + OCI LABEL。

> **权重不打包进镜像**（DeepSeek-V4-Flash-0731 约 167GB）。权重走 Fireworks 模型管理
> （管理网下载 → head → RoCE 同步 → 各节点 HF 缓存），镜像内 `HF_HUB_OFFLINE=1` 离线加载。

### 3. 配方（`recipe/fireworks.recipe.json`）

严格对齐 Fireworks 原生 `POST /api/recipes/import` schema
（`name / description / image / compose_template / variables[]`）。

- `image` 字段 = `build.sh` 产出的专属镜像 tag（脚本会校验一致）。
- `compose_template` 使用 v0.26.0 主路径 `vllm serve` 参数（nvfp4_ds_mla / dspark 投机 /
  MoE auto / instanttensor）+ 多节点分布式参数（`--distributed-executor-backend mp`，
  worker 自动 `--headless`）。

---

## 快速开始

### 前置条件

- **linux/arm64**（aarch64）构建机：DGX Spark 本机或同架构构建机
- Docker ≥ 23（BuildKit，`mount=type=cache`） + compose v2
- 能访问 GitHub / HuggingFace（源码编译需拉取组件仓库）
- 磁盘充足（编译 vLLM/Rust/FlashInfer，建议预留 ≥ 100GB）

> ⚠️ 首次源码编译耗时较长（数小时量级）。日志落盘在 `logs/`，各组件版本全部可参数化。

### 1) 编译基础 runner

```bash
./scripts/build.sh base
```

### 2) 构建模型专属镜像

```bash
# base 已存在时只会构建模型层；--push-registry 推送到你的 registry
./scripts/build.sh model --model deepseek-v4-flash-0731 --push-registry ghcr.io/<org>

# 或导出 tar 归档（Fireworks 镜像管理 docker load 路线）
./scripts/build.sh model --model deepseek-v4-flash-0731 --save dist/

# 可选：bake JIT 缓存种子进镜像（服务节点跑全量 warmup 生成 vllm-cache 后）
#   SEED_CACHE_DIR=seed-cache bash scripts/build.sh model
#   使新节点上线零推理期 JIT 编译（见 docs/BENCHMARK-v0260.md 追加3）。
```

### 3) 在 Fireworks 中运行（WebUI 全流程）

1. **镜像页**：拉取专属镜像或导入 `--save` 归档 → 自动分发到节点。
2. **配方页**：导入 `models/deepseek-v4-flash-0731/recipe/fireworks.recipe.json`。
3. **发布任务**：选该配方 → 选集群（**≥2 节点，head 设为 rank0**，TP=2）→ 发布。
4. Fireworks 自动分发模型 → worker 先起、head 后起 → 健康检查轮询 `:8000/v1/models` 就绪。

### 4) 推理验证

```bash
curl -s http://<head-ip>:8000/v1/models
curl -s http://<head-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"你好"}],"thinking":true}'
```

### 5) 基准与真机部署（scripts/bench + scripts/deploy）

```bash
# 基准（与 docs/BENCHMARK-v0260.md 同口径；prefill 用随机头破前缀缓存）
python3 scripts/bench/bench_decode.py  --base-url http://<head-ip>:8000/v1 --model deepseek-v4-flash-0731 --concurrency 1,2,4 --max-tokens 128
python3 scripts/bench/bench_prefill.py --base-url http://<head-ip>:8000/v1 --model deepseek-v4-flash-0731 --sizes 1024,2048,4096,8192,16384

# 真机双节点部署（TP=2；先起 worker → 后起 head → 健康检查轮询）
# 默认含长上下文 JIT 形状预热（32K..1M，防推理期新形状编译 OOM；WARMUP=0 跳过）。
# 宿主建议预配 32GiB swap（/swapfile + fstab）作为编译峰值兜底。
HEAD_IP=<head-ip> WORKER_IP=<worker-ip> ./scripts/deploy/deploy_v0260.sh [IMAGE]
#   （0.3.0+：docker run 前自动把镜像内 JIT 缓存种子幂等灌入宿主挂载盘，全新节点零推理期编译）

# 单节点启动（供手动/排障）——MASTER_ADDR 需容器内可解析的 head 地址：
./scripts/deploy/run_vllm_node.sh <ip> <rank> <headless:0|1> [IMAGE]
```

> 提示：deploy 脚本的节点地址、各 bench 脚本的 base-url 均可用环境变量覆盖；默认值
> 以 `<head-ip>`/`<worker-ip>` 占位（见各脚本头部注释）。

---

## 实测结果概览（v0.26.0 主路径 · 双节点 DGX Spark TP=2）

完整实测数据、调试历程与各版本对比见 **[docs/BENCHMARK-v0260.md](./docs/BENCHMARK-v0260.md)**。

要点：

| 项 | 结果 |
|---|---|
| 内核栈 | `nvfp4_ds_mla` KV / DeepGemmFP4Experts(MXFP4) MoE / SM120 sparse-MLA decode / dspark MTP=5 + regular CUDA graph |
| decode | 512-token 固定：**41.2 / 94.4 / 151.2 tok/s** @1/2/4；per-session 54.8（数数）/ 34.1（prose） |
| prefill | 2K–16K 1300–2400 tok/s；**128K=1566 / 256K=1289 / 512K=1400 / 1M=900+ tok/s 实测可用** |
| 稳定性 | 长上下文（200K–1M）无崩溃；推理期 JIT 编译已通过 **fw-warmup 补丁 + JIT 缓存 bake 消除**（新节点上线 mHC warmup <1s、0 推理期编译） |

最终配置（对齐 jvr0x 与 tonyd2wild 的 1M/NVFP4 双节点配方）：
`--kv-cache-dtype nvfp4_ds_mla --max-model-len 1048576 --max-num-seqs 6
--max-cudagraph-capture-size 36 --gpu-memory-utilization 0.88` +
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`（regular CUDA graph）+ MoE=auto。

> ⚠️ 真机验证结论：**MoE 用默认 auto（DeepGemmFP4Experts）**——`--moe-backend
> flashinfer_b12x` 的 decode 更优但 prefill 在 v0.26.0 集成下仅 ~88 tok/s（不可用）；
> auto 的 prefill 稳定 2200+ tok/s。

---

## 版本历史

- **0.3.0**（当前）——vLLM v0.26.0 主路径 + fw-warmup 补丁 + JIT 缓存 bake 进镜像。
- **0.2.0**——迁移到主流 vLLM v0.26.0（Anemll 式 GB10 overlay、nvfp4_ds_mla KV、1M 上下文）。
- **0.1.0**——早期自定义 b12x fork 栈（历史存档，已废弃，详见 older git history / 排障表）。

升级路径：`./scripts/build.sh base && ./scripts/build.sh model`（base 0.2.0 + model 0.3.0）。

---

## 新增一个模型

1. 复制目录：`cp -r models/deepseek-v4-flash-0731 models/<new-model>`
2. 改 `build.conf`：`MODEL_NAME / MODEL_ID / IMAGE_REPO / IMAGE_TAG`
3. 改 `Dockerfile.model`：烘培的补丁列表与调优 ENV
4. 改 `recipe/fireworks.recipe.json`：默认 `image`、`DSPARK_MODEL` 默认值、必要参数
5. 有新补丁就放 `models/<new-model>/patches/<name>/` 并在 Dockerfile 中 `COPY + RUN`

---

## 常见排障

| 现象 | 处理 |
|---|---|
| 构建到某阶段失败 | 看 `logs/base-*.log` 定位阶段；改对应版本 pin 或补丁后重跑（ccache/cargo/uv 缓存可复用） |
| `hybrid-draft-loader` / `fw-warmup` 报锚点不匹配 | vLLM 版本漂移；按提示更新对应锚点常量 |
| 运行报 NCCL/RoCE 异常 | 先确认节点已配置 4×100G RoCE（Fireworks 集群页创建/验证通过）；SSH/节点 IP 需可解析 |
| 镜像架构不对 | 必须在 aarch64 构建机编译（脚本已检测并警告） |
| 健康检查超时 | 看 head 节点容器日志；确认 `VLLM_PORT`、head=rank0 |
| 1M 上下文 KV 内存不足 | 权重加载后可用 KV 约 10.75GiB/节点；提高 `GPU_MEMORY_UTILIZATION`（≈0.9）或降 `MAX_MODEL_LEN` |
| 依赖解析把 torch 换掉 | runner 使用 `--override` pin 了 torch/fastapi 版本；如仍被替换检查 `logs` 中 uv 输出 |
| API 不返回缓存命中信息（`prompt_tokens_details: null`） | 非缓存失效：v0.26.0 默认关 `--enable-prompt-tokens-details`（本配方已默认开启） |
| SM120 sparse-MLA 解码崩（`num_tokens > 64` / 页块断言） | flashinfer 必须 0.6.15（0472b9b）+ attention overlay；`--block-size 256` 下 0.6.14 必崩 |
| 空段 reshape 崩溃（`cannot reshape [0, -1]`） | overlay `_forward_prefill` 空 chunk 护栏（`query_end<=query_start: continue`） |
| MoE 用 b12x 后 prefill 极慢（~88 tok/s） | v0.26.0 集成下 b12x prefill 不可用；改 `--moe-backend auto`（DeepGemmFP4Experts，2200+ tok/s） |
| 长上下文请求（≥200K）极慢甚至失败/引擎被杀 | 用 0.3.0+ 镜像（fw-warmup + JIT 缓存 bake 已消除推理期编译）；旧版本见 BENCHMARK 追加2/3 的 timeout/swap/warmup 修复 |
| 构建卡在 triton_kernels fetch | vLLM CMake 全量克隆 triton-lang/triton（需 MLIR 无法换浅克隆）；网络波动重试构建即可 |

---

## 许可与致谢

本项目采用 **Apache-2.0**（见 [`LICENSE`](./LICENSE)）；第三方来源与派生关系见
[`NOTICE.md`](./NOTICE.md)。

参考与致谢：
- [vllm-project/vllm](https://github.com/vllm-project/vllm)（v0.26.0，源代码编译基座）
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)（GB10 性能配方与 attention overlay 移植）
- [lukealonso/b12x](https://github.com/lukealonso/b12x)（MXFP4 MoE 内核，独立安装）
- [jvr0x/dgx-spark-bench](https://github.com/jvr0x/dgx-spark-bench) 与
  [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  （1M/NVFP4 双节点配方参考）
