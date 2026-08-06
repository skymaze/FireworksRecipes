# overlay/vllm —— vLLM 源码层覆写（构建期烘培进 wheel）

本目录是与当前基座 vLLM **v0.26.0**（`overlay` 覆写的目标版本）同名的 Python
文件模块覆写。构建期由 `docker/vllm-b12x.Dockerfile` 在 `vllm` 阶段以
`rsync -a overlay/vllm/ → 源码 vllm/` 合入，随后编译进 vLLM wheel；运行期不再
打补丁。

## 来源与许可

移植自 **Anemll/dspark-vllm-gx10**（MIT；`overlay/` 内 vLLM 派生文件 Apache-2.0，
SPDX 头保留于各 .py 文件）。Anemll 配方基于 vLLM v0.25.1，此处定向移植到
**v0.26.0**，仅保留对 DGX Spark（GB10 / sm_121a / MXFP4）+ DeepSeek-V4 性能
必需的增量，不做整文件替换（v0.26.0 原生演进部分全部保留）。

## 内容

| 文件 | 相对 stock v0.26.0 的改动 | 作用 |
|---|---|---|
| `config/cache.py` | CacheDType 增加 `nvfp4_ds_mla` | 允许 4bit DS-MLA KV dtype |
| `v1/kv_cache_interface.py` | `nvfp4`/`fp8_ds_mla` 判断纳入 `nvfp4_ds_mla` | 584B sparse-MLA 封包归组 |
| `utils/torch_utils.py` | dtype 映射 + nvfp4 归组 | 运行期 KV buffer dtype |
| `envs.py` | 4 个 `VLLM_USE_B12X_*` → `VLLM_B12X_*` 开关 | b12x MoE 调优开关 |
| `model_executor/layers/fused_moe/oracle/mxfp4.py` | Mxfp4MoeBackend 增 `B12X_MXFP4` + 注册 + 权重路径 + 后加载 | DeepSeek-V4 原生 MXFP4 走 lukealonso/b12x MoE (`--moe-backend flashinfer_b12x`) |
| `model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py` | 新增文件（Anemll 原样） | b12x MXFP4 MoE 专家实现 |
| `models/deepseek_v4/attention.py` | Anemll 增量三方合入 v0.26.0 | DSV4 稀疏 attention 装配 |
| `models/deepseek_v4/sparse_mla.py` | Anemll 增量三方合入 | 稀疏 MLA 层封装 |
| `models/deepseek_v4/nvidia/flashinfer_sparse.py` | Anemll 增量三方合入 | SM120 sparse-MLA decode/prefill 调度（含 584B 封包、TP=2、dispatch 宽填充） |
| `v1/attention/backends/mla/sparse_swa.py` | Anemll 增量三方合入 | 256-token SWA 页 → 64-token 零拷贝子视图（配合 SM120 decode 内核页块=64） |
| `v1/attention/backends/mla/flashmla_sparse.py` | Anemll 增量三方合入 | flashmla sparse 后端修正 |

## 备注
- **flashinfer 需 0.6.15（commit 0472b9b）**：新增 `_packed_kv_page_block_size()` 从
  KV 张量推导打包页块，兼容 `--block-size 256` 存储 + SM120 decode 64-token 子块；
  `v0.6.14`（v0.26.0 默认 pin）真机已验证在 DeepSeek-V4 稀疏解码时崩溃
  （`num_tokens > 64` 断言 / 256 页不满足 decode 实例化表）。
- **MoE 后端实测结论（真机）**：b12x（`--moe-backend flashinfer_b12x`）decode 更优，
  但 prefill 在 v0.26.0 集成下仅 ~88 tok/s（GPU 96% 满载仍慢）；**最终配置默认
  auto（DeepGemmFP4Experts，prefill 2200+ tok/s）**。b12x 保留可用（decode 场景）。
- **KV dtype 最终配置 `nvfp4_ds_mla`（1M 上下文）**：dtype 枚举/归组 + attention
  overlay 数据路径已在真机启用（"padded nvfp4_ds_mla KV cache format"）；配合
  `GPU_MEMORY_UTILIZATION≥0.88` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（regular），
  256K prefill 实测 1289 tok/s，长上下文稳定。`fp8_ds_mla` 仍可用（350K 形状）。
