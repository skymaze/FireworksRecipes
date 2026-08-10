# 配方源格式规范 (Recipe-Source Format)

本规范定义 FireworksRecipes 仓库中的**配方文件格式**，以及 Fireworks 如何**导入/导出**
这一格式。目的是：单个配方文件自包含、可离线、可分享、可提交 PR。

> Fireworks 本地不用双语：导入/导出时按**当前界面语言**本地化，缺省回退配方中存在的
> 其他语言。

---

## 1. 配方文件 `fireworks.recipe.json`

每个模型配方是一个独立 JSON 文件，遵循下表字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | ✅ | 主语言（默认中文）配方名 |
| `name_en` | string | 否 | 英文配方名（缺省回退 `name`） |
| `description` | string | 否 | 主语言描述 |
| `description_en` | string | 否 | 英文描述（缺省回退 `description`） |
| `version` | string | 否 | 配方/镜像版本（如 `0.3.1`；通常对齐专属镜像 tag） |
| `image` | string | 否 | 默认镜像（可为空，由变量覆盖） |
| `nodes` | int | 否 | **固定拓扑**：确切的节点数量（None=不固定）。每个配方按固定设备数调优，发布时须恰好匹配（不用 min/max 比较）。GB10 每机 1 GPU，TP=节点数，不另设 `tensor_parallel`/`topology`；也不设 `dtype`（可能混合架构） |
| `compose_template` | string | ✅ | compose 模板（每节点一份，支持 `${VAR}` 占位符，`.env` 插值） |
| `variables` | array | ✅ | 变量定义（见下表） |

### 变量定义项

| 字段 | 类型 | 说明 |
|---|---|---|
| `key` | string | 变量名（写入 `.env` / 渲染 environment） |
| `label` | string | 主语言标签；`label_en` 为英文（缺省回退） |
| `help` | string | 主语言说明；`help_en` 为英文（缺省回退） |
| `type` | `string/int/float/bool/select` | 变量类型 |
| `source` | `user/cluster/node` | 变量来源 |
| `auto` | string | `source=cluster/node` 时的自动填充键（见 README，如 `head_roce_ip`、`node_rank`） |
| `default` | string | 默认值 |
| `options` | string[] | `select` 的可选值 |
| `required` | bool | 是否必填 |
| `picker` | `""/model/image` | 发布页提供已下载模型/已拉取镜像快速选择 |
| `min`/`max` | int | 数值校验元数据（**节点数不在此表达**，见顶层 `nodes`） |

> 语言约定：字段尽量提供主语言（zh）+ `.en` 并列字段；只写一种语言即可（Fireworks
> 在加载时按界面语言选择，缺省回退存在的语言）。

### 自动填充键（`source=cluster/node` 的 `auto` 取值）

`auto` 取值只能是下表之一（与 Fireworks 后端 `recipe_render` 的自动变量一一对应；
`scripts/validate.py` 会校验未知键，跨仓库新增键需同步下表与后端 renderer）：

| auto 键 | source | 说明 |
|---|---|---|
| `head_roce_ip` | cluster | 当前任务 head 节点的 RoCE IP（填 `MASTER_ADDR`） |
| `nodes_total` | cluster | 任务节点总数 |
| `network_type` | cluster | 集群网络类型（roce/ib/ethernet） |
| `head_ip` | cluster | head 节点管理网 IP |
| `head_hostname` | cluster | head 节点主机名 |
| `node_rank` | node | 任务内节点序号（head=0，worker 递增） |
| `role` | node | 任务内角色（head / worker） |
| `hostname` | node | 节点主机名 |
| `node_ip` | node | 节点管理网 IP |
| `node_roce_ip` | node | 首选 RoCE 口 IP（填 `VLLM_HOST_IP` 等） |
| `hca` | node | 可用 RoCE HCA（逗号分隔，NCCL 多 rail，填 `NCCL_IB_HCA`） |
| `netdev` | node | 首选 RoCE 口网卡名（填 `NCCL_SOCKET_IFNAME`） |
| `gid_index` | node | RoCEv2 GID index（填 `NCCL_IB_GID_INDEX`） |
| `agent_port` | node | 节点 agent 服务端口（9000） |
| `headless` | node | head 为空、worker 为 "1"（worker 不跑 API server） |

> `MASTER_PORT` 等分布式端口属**用户变量**（`source=user`，默认 25000），不作为自动键。

### 示例

```json
{
  "name": "Example-Model（专属镜像）",
  "name_en": "Example-Model (dedicated image)",
  "description": "示例配方。",
  "description_en": "Example recipe.",
  "version": "1.0.0",
  "image": "example/models:1.0.0",
  "nodes": 2,
  "compose_template": "services:\n  app:\n    image: ${IMAGE:-example/models:1.0.0}\n    ...",
  "variables": [
    {
      "key": "MAX_MODEL_LEN",
      "label": "最大上下文长度",
      "label_en": "Max context length",
      "type": "int",
      "source": "user",
      "default": "1048576"
    },
    {
      "key": "MASTER_ADDR",
      "label": "Head 集群地址",
      "source": "cluster",
      "auto": "head_roce_ip",
      "required": true
    }
  ]
}
```

---

## 2. 目录清单 `recipes/index.json` (manifest)

商店目录由 manifest 驱动（等同 recipes.vllm.ai 的 `/models.json`），每条：

| 字段 | 说明 |
|---|---|
| `id` / `provider` / `model` | 标识与厂商/模型 |
| `path` | `fireworks.recipe.json` 仓库内相对路径 |
| `version` | 版本 |
| `readme` / `readme_en` | 默认（中文）/英文 README 相对路径 |
| `params` / `context_length` / `modality` | 卡片元数据（不设 `dtype`：可能混合架构） |
| `nodes` | 固定拓扑（发布时须恰好匹配） |
| `image` / `tags` | 镜像与标签 |
| `description` / `description_en` | 中英文描述 |

---

## 3. 导入 / 导出行为（Fireworks）

- **导出**：本地配方 → `.recipe.json`（上文格式）。名称自动还原为 `base` + `version`
  （剥离本地命名追加的版本后缀）；字段为本地语言（本地只存单语言）。
- **导入（文件 / 粘贴）**：读 `.recipe.json` → 按**当前界面语言**本地化：
  - en：优先取 `_en` 字段；缺省回退主语言；
  - zh/其它：优先主语言；缺省回退 `_en`；
  - 变量 `label/help` 同规则，导入后剥离 `_en` 并列字段。
  - 导入产物为**本地独立配方**（与来源无关联，可任意编辑；每次导入必新建、不覆盖）。
- **从配方源商店导入**：同一本地化规则 + `lang` 由界面语言决定。

---

## 4. 提 PR 建议

1. `recipes/<id>/recipe/fireworks.recipe.json` + `README.md`（+ 可选 `README.en.md`）
2. 在 `recipes/index.json` 登记一条（含 `nodes`、`description(_en)`、`readme(_en)`；`image/version/nodes` 须与配方一致）
3. 保持 `MASTER_ADDR=head_roce_ip` / `MASTER_PORT` 为用户变量等当前变量模型约定
4. 大体积构建产物一律进 `.gitignore`，仓库保持轻量
