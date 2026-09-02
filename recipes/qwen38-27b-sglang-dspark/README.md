# Qwen3.8-27B · SGLang DSPARK · Fireworks 配方（单节点 DGX Spark）

用 Fireworks 在 **1 台** DGX Spark 上跑起 **RadixArk/Qwen3.8-27B-NVFP4** 的 SGLang 服务
（含 DSPARK 投机解码）。

## 模型

- 主模型：`RadixArk/Qwen3.8-27B-NVFP4`（NVFP4 4-bit 权重）+ `--kv-cache-dtype fp8_e4m3`
- 草稿模型：`RadixArk/Qwen3.8-27B-DSpark`（DSpark mamba 草稿模型，草稿注意力 flashinfer；
  **必须分发到节点缓存**，否则投机解码启动失败）
- 引擎/镜像：SGLang，`lmsysorg/sglang:qwen38-27b`（官方镜像，model 专用 tag）；flashinfer、
  `--chunked-prefill-size 2048`、`--mem-fraction-static 0.85`
- API 端口默认 `30000`；reasoning/tool parser 用 `qwen3` / `qwen3_coder`
- ⚠️ `MAMBA_FULL_MEMORY_RATIO` 默认 `11.01`（应为 `[0,1]`，疑为 `0.1101` 的笔误），
  发布前务必按 SGLang 文档/实测修正

## 速度

未附本地实测（仓库内首个 SGLang 配方，实机验证后再回填）。

## 硬件需求

- **1 台** DGX Spark（固定单节点 · GB10 单 GPU）
- 模型由 Fireworks 分发到节点 HF 缓存后**离线加载**（`HF_HOME=/root/.cache/huggingface`、
  `HF_HUB_OFFLINE=1`）；仅支持 1 节点，勿分配多节点拓扑

## 参考上游

- `lmsysorg/sglang:qwen38-27b`：官方 SGLang 镜像（源 docker run 引用）
- [sgl-project/sglang](https://github.com/sgl-project/sglang)：引擎
- [RadixArk](https://huggingface.co/RadixArk)：模型与 DSpark 草稿模型

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
