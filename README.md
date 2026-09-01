# 🎇 FireworksRecipes

为 **Fireworks**（DGX Spark 集群管理工具）提供 **DGX Spark（GB10）模型配方**的仓库：
每条配方是可直接导入 Fireworks 的 `fireworks.recipe.json`（含镜像/拓扑/参数/网络自动填充），
配套 `recipes/index.json` 目录清单供「配方商店」读取。

> 本仓库**只含配方与目录清单**，不含镜像构建代码：配方引用的模型镜像由镜像仓库提供
> （Fireworks 拉取后分发到节点），本仓库不做源码编译、不打镜像。

**English**: [docs/README.en.md](./docs/README.en.md)

当前配方：

| 配方 | 镜像 | 说明 |
|---|---|---|
| DeepSeek-V4-Flash (DSpark) | `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix2` | 双节点 TP=2 DSpark 服务 · FlashInfer b12x + dspark 投机 · NVFP4 DS-MLA · 1M 上下文 |
| [DeepSeek-V4-Flash-Vision-Exp (DSpark)](recipes/deepseek-v4-flash-vision-exp-dspark/README.md) | `registry.cn-shanghai.aliyuncs.com/aixn-public/dspark-vllm-gx10-mia:v0.1.1-hotfix4` | **双节点 TP=2**（多模态）· **原生图片输入**（OpenAI image_url ≤8 张 / 仅 user 消息 / 无视频）· FlashInfer b12x + dspark 投机 k=6 · NVFP4 DS-MLA · **1M 上下文**（dev：镜像已烘焙，checkpoint 待实机验证） |
| DeepSeek-V4-Flash (TP=4) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | **四节点 TP=4** DSpark 服务 · FlashInfer b12x + dspark 投机 k=5 · NVFP4 DS-MLA · **1M 上下文** · agentic 工作负载实机验证 |
| DeepSeek-V4-Flash (Spark b12x) | `eugr/spark-vllm-b12x:latest` | **双节点 TP=2** Spark-vLLM 服务 · B12X MLA SPARSE + b12x MoE/线性 · dspark 投机 k=5 · **FP8 KV** · instanttensor + AOT · 1M 上下文 |
| Qwen3.8-27B (SGLang DSPARK) | `lmsysorg/sglang:qwen38-27b` | **单节点** SGLang 服务 · flashinfer + DSPARK 投机（mamba 草稿）· **FP8 KV** · `--mamba-full-memory-ratio 11.01`（疑似笔误，待验证） |
| GLM-5.2 QuantTrio (DCP4) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit` | **四节点 TP=4 + DCP4** · B12X MLA SPARSE + a2a · MTP k=2 · **nvfp4_ds_mla KV** · **315,968** 上下文 · spark-kit 生产 overlay |
| GLM-5.3-Flash (Lane A) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v8` | **四节点 TP=4** · **fp8 KV**（FlashInfer SM12x unlock）· MTP k=4 · **1M 上下文** · ~55 tok/s 结构化解码 · NVFP4 量化（默认 RedHatAI compressed-tensors，可换 uncensored drop-in）· ⚠️ 上游已标 MTP TP4 superseded，新部署用 DFlash2 |
| GLM-5.3-Flash (DFlash2 TP=2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **双节点 TP=2** · fp8 KV + **DFlash2**（incoai drafter）· **262K 上下文** · 单流 46.9 tok/s · C1–C6 零失败（上游 one-to-copy 档）· KV profiler 定池（581K，勿 pin）· 默认 RedHatAI checkpoint |
| GLM-5.3-Flash (DFlash2) | `registry.cn-shanghai.aliyuncs.com/aixn-public/glm53-flash-sm121:v11-dflash2` | **四节点 TP=4**（上游当前默认）· fp8 KV + **DFlash2** k=7 块扩散投机（incoai drafter，KV 池成本 ~0）· **1M 上下文** · **3.9M-token KV 池**（24 GiB/rank，需无条件 flusher）· 单流 54.5 tok/s · 默认 RedHatAI checkpoint |
| GLM-5.3-Flash (EXL3 TP=2) | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | **双节点 TP=2** · **EXL3/TR3 4bpw 权重**（Mia-AiLab 镜像，KLD≈官方 FP8）× fp8 KV + **DFlash2** k=7 · **1M 上下文**（padded slot-share）· Vision 默认开 · 单流 62.9 tok/s（×4 聚合 146.5） |

> **拓扑固定**：每条配方声明**确切的节点数**（如 2 节点 · TP=2 或 4 节点 · TP=4）；
> Fireworks 发布时必须恰好匹配，模型参数按该拓扑调优。不同拓扑请选用对应配方。

---

## 分支模型

| 分支 | 内容 |
|---|---|
| `main` | **经过实机测试**的配方：可发布、供 Fireworks 商店稳定使用 |
| `dev` | **测试中的配方**：新配方 / 参数调整先进这里，实机验证通过后再 merge 到 `main` |

工作流：

- 任何新配方 / 现有配方改动，先进 `dev` 分支（文档与 `recipes/index.json` manifest 随分支一起走）。
- 在 `dev` 上通过实机测试后，`git checkout main && git merge dev` 发布为稳定配方。
- 测试未通过 / 中途放弃的改动，留在或丢弃在 `dev`，**不进 `main`**。

Fireworks 加载配方源时可**自由选择分支**（默认 `main`；添加配方源时可选 `dev` 等），
因此运行中的集群可用 `dev` 分支预览「测试中的配方」，稳定后切回 `main`。

---

## 配方目录（Fireworks 直读）

本仓库同时是 **Fireworks 的配方源**：Fireworks 同步本仓库 → 读取目录清单 → 用户从
「配方商店」一键安装并发布。

- `recipes/index.json` —— **目录清单**：每条列出 `id / provider / model / path / readme /
  readme_en / version / params / context_length / modality / nodes / image / tags`，以及
  `name/description`（含 `*_en`）；Fireworks 只读它，不做整树扫描。
- `recipes/<id>/fireworks.recipe.json` —— 可运行配方，字段对齐 Fireworks
  `POST /api/recipes/import` schema；`image` 为镜像仓库里的现成镜像 tag。
- `recipes/<id>/README.md` —— 介绍文档，Fireworks「配方商店」详情里渲染；可另配 `README.en.md`。
- **双向**：文本字段可带英文并列字段 `xxx_en`（`name_en / description_en / label_en /
  help_en`）；Fireworks 按界面语言选择：English 优先取 `_en`，缺省回退主语言（zh）。
  只写一种语言也可，自动回退。
- 详细字段规范见 [docs/RECIPE-FORMAT.md](./docs/RECIPE-FORMAT.md)。
- 配方变量模型跟随 Fireworks 当前版本：`MASTER_ADDR`=`cluster/head_roce_ip`、
  `MASTER_PORT` 为用户变量默认 25000、`NODE_RANK / HEADLESS / VLLM_HOST_IP / NCCL_*`
  为 `node` 变量自动填充。

---

## 目录结构

```
FireworksRecipes/
├── LICENSE / NOTICE.md / SECURITY.md   # 许可与来源/安全声明
├── .github/workflows/ci.yml            # 轻量校验 CI（validate.py）
├── .gitignore
├── recipes/
│   ├── index.json                  # ★ 目录清单，商店数据源
│   ├── deepseek-v4-flash-dspark/
│   ├── deepseek-v4-flash-vision-exp-dspark/   # 双节点 TP=2 · Vision-Exp 多模态（dev 测试中，待实机验证）
│   │   ├── fireworks.recipe.json   # ★ Fireworks 原生配方
│   │   └── README.md / README.en.md
│   ├── deepseek-v4-flash-0731-tp4-4x/   # 4 节点 TP=4（agentic 实机验证调优）
│   │   ├── fireworks.recipe.json
│   │   └── README.md / README.en.md
│   ├── deepseek-v4-flash-0731-spark-b12x/   # 2 节点 TP=2 · eugr/spark-vllm-b12x（源自 docker run，未实机验证）
│   │   ├── fireworks.recipe.json
│   │   └── README.md / README.en.md
│   ├── qwen38-27b-sglang-dspark/   # 单节点 · SGLang + DSPARK（源自 docker run，未实机验证）
│   │   ├── fireworks.recipe.json
│   │   └── README.md / README.en.md
│   └── glm-5.2-quanttrio-tp4-dcp4-4x/   # 4 节点 TP=4 + DCP4 · spark-kit 生产栈（镜像构建推送至 ACR）
│       ├── fireworks.recipe.json
│       └── README.md / README.en.md
├── scripts/
│   └── validate.py        # 配方/manifest 校验（schema + 一致性 + auto 键）
├── schemas/
│   ├── manifest.schema.json
│   └── recipe.schema.json
└── docs/
    ├── README.en.md       # 英文文档
    └── RECIPE-FORMAT.md   # 配方字段规范
```

---

## 在 Fireworks 中运行（WebUI 全流程）

1. **配方页**：添加配方源（本仓库）→ 从商店导入 `recipes/<id>/fireworks.recipe.json`。
2. **镜像页**：确认配方指向的镜像已在镜像仓库可拉取（Fireworks 分发到节点）。
3. **发布任务**：选该配方 → 选集群（head 设为 rank0，节点数须恰好匹配配方 `nodes`）→ 发布。
4. **推理验证**：

```bash
curl -s http://<head-ip>:8888/v1/models
curl -s http://<head-ip>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"thinking":true}'
```

---

## 新增 / 修改一个配方

1. 复制目录：`cp -r recipes/deepseek-v4-flash-dspark recipes/<new-id>`（去掉非配方文件）。
2. 改 `fireworks.recipe.json`：`name / description / image / 变量默认值 / nodes / version`。
3. 写 `README.md`（可加 `README.en.md`）。
4. 在 `recipes/index.json` 登记一条（`image/version/nodes` 必须与配方一致，否则
   `validate.py` 报错）。
5. 本地校验：`python3 scripts/validate.py`。
6. 提交到 `dev`，实机验证通过后 merge 到 `main`。

---

## 校验

```bash
python3 scripts/validate.py
```

- 校验配方 schema、`recipes/index.json` manifest 一致性、以及 `source=cluster/node`
  变量的 `auto` 键是否已知（与 Fireworks 后端 `AUTO_KEYS` 对齐，见 `docs/RECIPE-FORMAT.md`）。
- CI（`.github/workflows/ci.yml`）在 main/dev push 与 PR 上自动运行该校验，零依赖。

---

## 参考来源与致谢

本项目采用 **Apache-2.0**（见 [`LICENSE`](./LICENSE)）；第三方来源与派生关系见
[`NOTICE.md`](./NOTICE.md)。

参数级参考（配方调参来源）：
- [jvr0x/dgx-spark-bench](https://github.com/jvr0x/dgx-spark-bench)：1M/NVFP4 双节点配方参考
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)：1M/NVFP4 双节点配方参考
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)：DSpark 双节点配方路线参考
