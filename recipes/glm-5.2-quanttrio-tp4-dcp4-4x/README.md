# GLM-5.2 QuantTrio · TP=4 · DCP4 · Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上以 **TP=4 + DCP4** 服务
GLM-5.2 QuantTrio（Int4-Int8Mix，unpruned，256 experts，315,968 上下文）。

## 模型

- 主模型：`QuantTrio/GLM-5.2-Int4-Int8Mix`（实测 378G = 124 主权重 + 4 MTP 分片）
- 量化/投机：Int4-Int8Mix · KV `nvfp4_ds_mla` · **MTP k=2** · 注意力 `B12X_MLA_SPARSE`
- 并行：**TP=4 + DCP4**（`--dcp-comm-backend a2a`，`mp` 后端）
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit`
  （AEON `v0.27.1` + `b12x@334a2d75` + spark-kit 生产 overlay）
- API 端口默认 `8210`；默认 gmu 0.90、`MAX_NUM_SEQS=16`

## 速度

实测（2026-08-18，真实 4× DGX Spark GB10，本配方一键部署）：

- 启动 ~**9.5 分钟**到可服务；**TTFT ≈ 0.64 s**（4 路并发 ~0.72 s）
- 单流 **~15–30 tok/s**（含思考）；冷 prefill ~580–618 tok/s
- **前缀缓存 ≈100×**：turn2+ 在 16K/100K 上下文各 ~1.3/1.7 s
- KV 池 683,360 token；v0.27 的 4 节点 idle-stall（vllm #51921）未复现

## 硬件需求

- **4 台** DGX Spark（固定 4 节点 · TP=4 + DCP=4），每机 1 GPU（GB10），双 rail RoCEv2
- gmu 0.90 保 KV 池优先（0.87 会 evict 深会话）；`--max-num-batched-tokens` 保持 4096
  （8192 在 DCP4 下启动失败）；`MAX_MODEL_LEN` 保持 315,968（64 对齐，否则并发下崩溃）
- 权重+编译 ≈ 101.9 GiB / 每机 121.69 GiB 统一内存

## 参考上游

- [joesinvestments/glm52-spark-kit](https://github.com/joesinvestments/glm52-spark-kit)：
  生产 DCP4 平台镜像与 launcher（`platform/Dockerfile`、`launch/launch_gx10.sh`、
  `docs/RECOMMENDATION.md`）
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) · [vllm-project/vllm](https://github.com/vllm-project/vllm)

完整来源与派生关系见仓库根 [`NOTICE.md`](../../NOTICE.md)。
