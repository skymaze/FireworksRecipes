# GLM-5.2 QuantTrio · TP=4 · DCP4 · Fireworks 配方（4× DGX Spark）

在 **4 台** DGX Spark（head + 3 worker，双 rail RoCEv2）上跑起 **GLM-5.2 QuantTrio**
（Int4-Int8Mix, unpruned, 256 experts）的 **TP=4 + DCP4** 服务：

- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit`
  = AEON `v0.27.1`（sm_121a vLLM 重建）+ `b12x@334a2d75` + spark-kit 生产 overlay（烘焙入镜像，
  无运行时挂载）
- 并行：TP=4 + **DCP4** + `--dcp-comm-backend a2a`、`mp` 后端、注意力 `B12X_MLA_SPARSE`
- 投机：**MTP k=2**（probabilistic + `quantization: compressed-tensors` + `draft_tensor_parallel_size 1`）
- KV/形状：`--kv-cache-dtype nvfp4_ds_mla`、**315,968 上下文**、`--max-num-seqs 16`、
  `--max-num-batched-tokens 4096`、gmu 0.90、`FULL` capture 阶梯 [3,6,9,12]
- 端口默认 `8210`；reasoning/tool parser 用 `glm45` / `glm47`

> 对应作者**当前生产栈**：`joesinvestments/glm52-spark-kit` 的 `platform/Dockerfile` +
> `launch/launch_gx10.sh`（RECOMMENDATION 2026-08-17 决定：DCP4 保持生产——相比 DCP1 单流
> 慢 10-25%，但持有 3 个常驻 316K 会话，深会话全落前缀缓存）。本配方镜像即按该生产 V1
> 配置烘焙。

## 快速开始（发布前就绪）

- 集群：4 台节点（head + 3 worker），双 rail RoCEv2 已配置测试。
- 镜像：`registry.cn-shanghai.aliyuncs.com/aixn-public/glm52-dcp4:v0.27.1-spark-kit` 已推送（本配方即由
  spark-kit 生产配置构建所得；Fireworks 拉取后分发到节点）。
- 模型：`QuantTrio/GLM-5.2-Int4-Int8Mix`（HF hub 布局 `models--QuantTrio--GLM-5.2-Int4-Int8Mix`，
  实测 378G = 124 主权重 + 4 MTP 分片）分发到节点缓存；容器内 `HF_HUB_OFFLINE=1` 按 repo id
  离线解析（`HF_HOME=/cache/huggingface`）。
- **NCCL**：默认不预加载——镜像自带 NCCL 已实测全链路可用；仅做 NCCL wedge 研究时才需要随
  缓存分发 `nccl-2.30.4` 并设置 `NCCL_LD_PRELOAD`。

## 主要可调变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM52_IMAGE` | `.../aixn-public/glm52-dcp4:v0.27.1-spark-kit` | 阿里云 ACR 平台镜像 |
| `GLM52_MODEL_PATH` | `QuantTrio/GLM-5.2-Int4-Int8Mix` | 模型（HF hub 离线解析；或填缓存内 snapshot 绝对路径） |
| `SERVED_MODEL_NAME` | `glm-5.2-quanttrio` | 对外服务名 |
| `VLLM_PORT` | `8210` | API 端口 |
| `MAX_MODEL_LEN` | `315968` | 生产值（对齐 64 的 block-table seam；316000/316K 会触发崩溃） |
| `MAX_NUM_SEQS` | `16` | 生产值；32 实测无增益 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | **勿用 8192**（DCP4 下 3/3 启动失败） |
| `GPU_MEMORY_UTILIZATION` | `0.90` | 保 KV 池优先；0.87 会 evict 深会话 |
| `CPU_DIST_TIMEOUT` | `1800` | `--cpu-distributed-timeout-seconds` |
| `NCCL_LD_PRELOAD` | （空） | 留空=镜像自带 NCCL；wedge 研究再填 nccl-2.30.4 路径 |
| `MASTER_PORT` | `29501` | 分布式主端口 |

`NODES_TOTAL`（固定 4）、`MASTER_ADDR`、`NODE_RANK`、`HEADLESS`、`VLLM_HOST_IP`、`NCCL_IB_*`
均由 Fireworks 自动填充。`NCCL_IB_HCA` 自动键按节点返回逗号分隔双 rail（`rocep1s0f0,roceP2p1s0f0`）；
`NCCL_IB_GID_INDEX` 自动解析（源 launcher 也每次启动动态解析，勿硬编码）。

## 实测验证（2026-08-18 · 4× DGX Spark GB10）

真实 4 节点（head + 3 worker）上按本配方一键部署验证：

- 启动约 **9.5 分钟**到可服务（权重加载 + CUDA Graph 预热）；4 容器稳定（restart-count 0），
  镜像经 ACR 直拉到节点。
- API 就绪后均值 **TTFT ≈ 0.64s**（/metrics），4 路并发 TTFT ~0.72s；单流 ~15-30 tok/s（含思考）。
- **v027 的 4 节点 idle-stall（vllm #51921）未复现**：空闲 90s 后请求仍即时送达；本栈
  （b12x + spark-kit overlay）已覆盖 `shm_broadcast` / KV 广播路径。
- KV 池 683,360 token（权重+编译 ≈101.9 GiB / 121.69 GiB，gmu 0.90，符合生产画像）。
- **GLM-5.2 思考默认全开**：reasoning 极啰嗦、常耗尽 `max_tokens`（`finish=length`、content 为空
  属正常）；请求级 `thinking:disabled` 在该构建不生效——生产请给足 `max_tokens` 或关注 thinking 控制。
- 未随缓存分发 nccl-2.30.4、未 LD_PRELOAD，镜像自带 NCCL 全链路（启动/多节点 NCCL/推理）验证 OK。

## 与源 launcher 的差异 / 集成层

- **overlay 烘焙代替运行时挂载**：生产 `launch_gx10.sh` 用 `-v` 把 spark-kit overlay 挂进
  site-packages；本镜像已把同一套（`torch_utils.py`/`kv_cache_interface.py` 用三方合并版）
  拷进镜像，运行时零挂载。
- **`$NODES/$SSH_HOSTS/$WEIGHTS_DIR/$OVERLAY_DIR` 等宿主变量** → Fireworks 自动填充变量
  （head_roce_ip / netdev / hca / gid_index / node_rank / headless）。
- **`--master-port` 从 29501 保留**；`--cpu-distributed-timeout-seconds 1800` 保留。
- 接入 Fireworks 标准集成层（host 网络、`${HEADLESS:+--headless}`、HF 离线加载、
  `VLLM_HOST_IP`、持久化编译缓存目录等）。

## 已知问题与部署注意（源自源仓库实测）

- **不要改**：`--all2all-backend` 保持默认（DeepEP 实测慢 37-85%）；`--enable-dbo`（继承
  DeepEP，已关）；gmu 0.90（0.87 会 evict 深会话）；`--max-num-seqs ≤16`；`--max-num-batched-tokens`
  保持 4096（8192 在 DCP4 启动失败）。
- **cudagraph 阶梯 [3,6,9,12] 是 `1+k` 的倍数**：有 gap 会导致 batch padding、
  非均匀批次触达稀疏 MLA indexer 的坏分支而崩溃——勿改成带空格档的 ladder。
- **上下文长度对齐**：`MAX_MODEL_LEN` 若非 64 整数倍，MTP-overhang seam 会在并发下首请求
  即崩溃；生产 315,968 已实测稳定。
- **前缀缓存 ≈100x**：冷 prefill ~580-618 tok/s，但 turn2+ 在 16K/100K 上下文各为 ~1.3/1.7s；
  深会话请用 streaming 客户端并保持前缀缓存命中（GLM 模板会删前轮 reasoning，需要时传
  `{"chat_template_kwargs":{"clear_thinking":false}}`）。API 返回 `usage.prompt_tokens_details.cached_tokens`
  即为当次请求缓存命中 token 数（已开 `--enable-prompt-tokens-details`），可用同前缀二次请求核对命中率。
- **NCCL wedge**：源仓库结论维持全速配置 + 外部看门狗（CROSS_NIC=1 默认），偏移成本 > 病因；
  真遇到冻结可对比 `NCCL_CROSS_NIC=0`。
- 单一 `mp` 后端、GB10 每机 1 GPU；只支持 4 节点（TP=4 + DCP=4）。

## 参考来源

本项目采用 **Apache-2.0**（见仓库根 [`LICENSE`](./LICENSE)）；第三方组件来源与派生关系见
仓库根 [`NOTICE.md`](../NOTICE.md)。

- [joesinvestments/glm52-spark-kit](https://github.com/joesinvestments/glm52-spark-kit)：生产
  DCP4 平台镜像与 launcher（`platform/Dockerfile`、`launch/launch_gx10.sh`、`docs/RECOMMENDATION.md`）
- [lukealonso/b12x](https://github.com/lukealonso/b12x)：稀疏 MLA / MoE 内核
- [vllm-project/vllm](https://github.com/vllm-project/vllm)：引擎
