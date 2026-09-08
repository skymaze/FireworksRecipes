# Qwen3.8-Flash-Next · 单节点 TP=1 · Fireworks 配方（1× DGX Spark）

用 Fireworks 在 **1 台** DGX Spark（GB10，128 GiB 统一内存）上跑起 **Qwen3.8-Flash-Next**
（176.9B）的 vLLM 服务：**TP=1**、PLE 查找表 NVMe 离载、MTP 投机 + reduced-vocabulary
drafting、原生 262,144 / YaRN 524,288 上下文、纯文本 + 图片 + 视频多模态。源自
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
（AGPL-3.0）。

## 模型与机制

- 主模型：`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`（NVFP4 checkpoint，磁盘 ~99 GB）
- **PLE 表离载是本配方成立的关键**：176.9B 中的 n-gram/PLE 查找表（26.82 GiB）从不做
  运算，由 CPU 离载 worker 从 **内存映射** 的打包表服务（`VLLM_PLE_CPU_OFFLOAD=1`）。
  映射用 `MADV_RANDOM` 建议 + `POSIX_FADV` 逐批 prefetch，每 token 磁盘读从 ~1,366 KiB
  降到 **57 KiB（−24×）**，并省出 ~2 GiB 本会浪费在预读上的统一内存
- 权重上 GPU ~71.75 GiB，运行时开销 ~5.6 GiB；KV 由 `GPU_MEMORY_UTILIZATION` 定，
  **宿主侧用 `HOST_RESERVE_GIB=26` 封顶**（对齐上游 **09-06 shipped**：`KV_TARGET_GIB=20`
  愿望 → GMU 0.786 = 95.60 GiB 预算 → KV 16.67 GiB；vLLM profiling 后 ~15.98 GiB fp8 KV =
  ~1,132,586-token 池 ≈4.3× 满 262k 请求，池重启间 ±~10%）
- **BF16 GDN/SSM 常驻状态**（`MAMBA_SSM_CACHE_DTYPE=bfloat16`，上游 09-06 shipped）：
  减半每步状态搬运（~0.23 GB/序列/步）与 mamba 页（注意力块 3,200→1,664 token），
  8 流 decode **+8.5%**、needles 15/15 不变；空 = checkpoint 的 float32
- **V2 model runner 固定**（`VLLM_USE_V2_MODEL_RUNNER=1`）：MTP 草稿副本不在 V2 默认集合，
  回退 V1 会改写共享 `compilation_config`——09-05 动态-K 失败正是 cudagraph_mode 被静默
  改成 PIECEWISE；固定后每次启动都捕获 FULL decode 图
- **镜像**：`registry.cn-shanghai.aliyuncs.com/aixn-public/qwen38-flash-next:v1.2.0`
  （官方基镜像 `vllm/vllm-openai:qwen38-flash-next` + MiaAI **五项补丁**烘焙：PLE layer
  prefetch、PLE offload 宿主握手、MTP draft vocab、ModelOpt MXFP8 BF16 回退、QSA FP8-KV）。
  源仓库在启动时改写容器内文件；**Fireworks 一键部署用本烘焙镜像即可**——换回官方原镜像
  会 PLE 离载死锁、fp8 KV 不可用。09-06 的变更均为运行时 env/参数，无需新镜像
- 服务端口默认 `8888`；served-model-name `qwen3.8-flash-next`；OpenAI 兼容 API
- **上游 09-07/09-08 的 `ABLIT`（gated Keys 消偏检查点，
  `drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only`）不
  采用**：它移除安全拒答、需在 HF 接受 gated 条款并另行下载 ~99 GiB 检查点，属于可选实验，
  非 shipped 默认；本配方固定 stock Mia NVFP4（ABLIT=0）。其余新提交为检查点状态解析、
  文档与 sweep 工具，不改 shipped 配置

## 部署前置（发布前在节点准备好）

1. **模型**：`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` 分发到节点 HF 缓存（磁盘 ~99 GB）。
2. **PLE 打包表**（一次性，~27 GB）：在节点宿主先用源仓库的
   `files/build_ple_packed_table.py` 对已分发的 checkpoint 构建一次，产物落在
   `~/.cache/vllm/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/*.packed_u8`
   （本配方把该目录挂进容器，启动时会检查，缺则退出）。
3. **镜像**：确认跑的是 v1.2.0 烘焙镜像（见上）。
4. **（可选）MTP 精简草稿词表**：用 `files/build_draft_vocab.py` 从模型自身输出语料生成
   一行一个 token id 的文件（256k 档 65,536 行 ≈ 0.16 GiB），填 `MTP_DRAFT_VOCAB` 宿主
   路径即开 reduced-vocabulary drafting（decode +25%）。

## 配方变量速览

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_MODEL_LEN` | 262144 | YARN=0 时上下文（原生上限 262,144，超了会被拒） |
| `YARN` | 0 | 0 = 原生 rope；1 = YaRN → `YARN_MAX_MODEL_LEN`（512k，factor 2.0） |
| `YARN_MAX_MODEL_LEN` | 524288 | YaRN 服务长度；1M 在 BF16 下不通过单机预算，勿抬 |
| `KV_CACHE_DTYPE` | fp8 | fp8 ≈1.85× KV 池（512k 下 2.73× 满请求）；auto = BF16 |
| `MAMBA_SSM_CACHE_DTYPE` | bfloat16 | GDN/SSM 状态精度；bf16 8 流 decode +8.5%（09-06 shipped）；空 = float32 |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | 固定 V2 runner；0 = 关闭（MTP 草稿副本可能回退 V1 改写法图配置） |
| `GPU_MEMORY_UTILIZATION` | 0.786 | 对齐 09-06 shipped（KV_TARGET_GIB 20 → GMU 0.786 = 95.60 GiB 预算）；调高须保宿主余量 |
| `MAX_NUM_SEQS` | 4 | 源已测 1/2/3/4 流；KV 池 ≈4.3×（长上下文是并发上限） |
| `MTP` | 3 | 训练草稿头投机 k=3（0 关，省 ~1.49 GiB）；多模态请求自动回退 |
| `MTP_DRAFT_VOCAB` | 空 | 宿主 token-id 文件路径；非空 = reduced-vocabulary drafting（+25% decode） |
| `MAX_NUM_BATCHED_TOKENS` | 2048 | chunked prefill 单批上限；8192 买 ~11% prefill、~3% KV 池 |
| `CUDAGRAPH_CAPTURE_SIZES` | auto | 每批可构建宽度 (1+K)×S；vLLM 自带列表在 MTP=3 会漏 3/5 序列批 |
| `PORT` | 8888 | 注意 comfy-h3.service 监视该端口（见安全规则） |
| `CUDAGRAPH_MODE` | FULL_DECODE_ONLY | NONE = eager 调试 |
| `VLLM_CACHE` | `${HOME}/.cache/vllm` | 宿主 PLE 打包表目录（挂到容器 `/root/.cache/vllm`） |

## 速度（源仓库实测，sparkDash）

**decode**（散文、空闲）

- **shipped 档（262k、FP8、MTP 3、`CUDAGRAPH_CAPTURE_SIZES=auto`、`MTP_DRAFT_VOCAB`
  65,536 token 全开）——09-05 基线**：

| 并发 | TTFT | 聚合 tok/s | 单流 tok/s | 相对 09-04 |
|---|---:|---:|---:|---:|
| ×1 | 270 ms | 46.3 | 46.3 | +25.5% |
| ×2 | 466 ms | 73.0 | 36.5 | +27.2% |
| ×3 | 355 ms | 91.9 | 31.2 | — |
| ×4 | 346 ms | 108.1 | 27.7 | +25.8% |

- **09-06 sweep（512k YaRN、FP8、MTP 3、bf16 GDN 状态、V2 pinned、每个校验宽度都捕获
  FULL decode 图）+ bf16 前基线对比**：

| 并发 | ms/引擎步 | token/步 | 聚合 tok/s | 单流 tok/s | 升级内容 |
|---|---:|---:|---:|---:|---|
| ×1 | 61.5 | 3.00 | **48.7** | 48.7 | — |
| ×8 | 131.0 | 2.81 | **162.9** | 20.4 | FULL 图全宽度 → 8 流短上下文 162.9（4 流 113.7） |
| ×8 (float32) | 141.4 | 2.80 | 151.6 | — | bf16 状态再 +8.5%（3 次重复同向） |

- 提升主体来自 **reduced-vocabulary drafting**（草稿 lm_head 从 1.18 GiB/步 按 65,536
  词表切片，省 2.61 GiB/步，decode 贴近内存带宽墙）；目标模型逐 token 校验，输出不变
  （MGSM：EN 94.8% vs 93.6%、ZH 86.4% vs 86.4% 持平/略升）
- MTP 平均接受长度 ~2.1/4；**K=3 在各并发都是最优**（09-06 静态扫描，无 crossover 需调度）
- bf16 GDN 状态质量和 fp8 KV 一样经过 15/15 needles @32k

**prefill**（fp8，bf16 状态）：8k **2,200** / 16k **2,304** / 32k **2,314** / 64k 2,257 /
128k 2,146 / 256k 1,944 tok/s（较 09-05 +~1.7%；decode 侧的 +8.5% 来自 bf16 状态，prefill 侧
因每 token 速率相同基本持平）

- 512k YaRN + BF16 KV 档：**KV 池 ~1,431,164–1,502,014 token（2.73–2.86× 满请求）**
- 宿主余量（shipped 档 262k，KV_TARGET 20）：启动后 ≥15.7 GiB、40 分钟空闲 15.5–16.4 GiB、
  5 并发 ~60k prompt 最低 14.26 GiB、`NV_ERR_NO_MEMORY` 0 次（09-06 十次启动全部复现）

## 多模态

- 视觉塔（27 层）已计入「权重上 GPU」，图片/视频**不再额外占 GPU 预算**，默认即开
- 源已验：336×336 三色带 PNG 按序命名；4 s/16 帧短视频按时间序命名四条颜色
- 用标准 OpenAI content-part：`image_url` / `video_url`（`http(s)://` 或 `data:` URI）
- 注意：**MTP 对多模态请求自动降级**（草稿模型吃不了多模态嵌入，vLLM 日志
  `using text-only draft inputs instead`）；视频 token 消耗大，长视频 + 512k 未验

## 推理默认开

`--reasoning-parser qwen3`：思考块单独落在 `reasoning` 字段。要关闭请**每请求**传
`"chat_template_kwargs":{"enable_thinking":false}`（无需重启）。小 `max_tokens` 时
`content` 常为空是正常的（回答还在思考里），budget ~400+ token 或关思考。

## 安全规则（源仓库踩坑经验）

- **宿主侧封顶 GPU 预算**：GMU × MemTotal ≤ MemTotal − `HOST_RESERVE_GIB`(26)。第一版
  方案（KV_TARGET_GIB 22、GMU 0.83）在三台机器上因宿主 MemAvailable 被吸干挂掉
  （2026-09-04）；09-05 起从宿主侧封顶。调高 GMU 必须重读源 `start.sh` 的预算推导，
  保 MemAvailable ≥ ~10 GiB——耗尽统一内存池挂内核（无 OOM、无日志）
- **comfy-h3.service 必须停用**：它轮询 `127.0.0.1:8888`，一有应答就拉起 ComfyUI
  （GPU 共租者）；在用时 8888 会互相抢 GPU。启动方会拒绝占用中的 8888，请
  `sudo systemctl disable --now comfy-h3.service` 或换端口
- **勿设 `PLE_OFFLOAD=false`**（TP=1）：99 GB 权重过 UVM 会挂宿主
- **镜像须含 QSA FP8-KV 补丁**（v1.2.0 已含），否则 `KV_CACHE_DTYPE=fp8` 会静默读出
  垃圾而非报错
- **别把 `YARN_CEILING_MODEL_LEN` 抬过 524288**（BF16）：1M 需要 ~28.8 GiB KV，
  容器 cap 会冲破 105 GiB 硬顶

## 参考上游

- [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  （AGPL-3.0-or-later）：本配方全部参数、内存预算、补丁清单与实测数字的来源；
  其 `start.sh` 的完整内存推导/离载握手/看门狗逻辑见源仓库（09-05 起含护卫双阈值
  `MemAvailable` / `MemFree` watchdog）
- [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) ·
  [Qwen](https://qwen.ai)（模型与 PLE 技术）
- 补丁清单中 FP8-KV 思路致谢 [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)
  （Apache-2.0）

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
