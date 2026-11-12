# Season2 公开题集迁移

主线 `tasks/` 使用 Season2 的 60 个算子，来自 dev 仓已合入的 PR 235，源提交为 `d5f55e7318cb72660187d6a5e4ca5ac92c022b58`。原 53 题及其原始 metadata 完整保存在 `archives/season1/tasks/`。

## 公开范围

每题提供 `desc.md`、`proto.yaml`、`golden.py`、`cases.yaml` 和 `cases.csv`，公开 case ID 为 1 到 20，共 1200 个公开用例。隐藏用例、内部筛选报告和内部工具不在本次同步范围内。

| 等级 | 名称 | 算子数 |
|---|---|---:|
| L1 | 基础 | 2 |
| L2 | 中级 | 13 |
| L3 | 高级 | 27 |
| L4 | 复杂 | 13 |
| L5 | 挑战级 | 5 |

算子名单与公开文件摘要见 `tasks/suite_manifest.json`。目录、proto 与 schema 入口保持一致。KV Cache 的公开接口要求显式返回两个输出 Tensor，允许直接返回被原地更新的输入缓冲别名，不要求额外复制；不引入省略返回值的接口扩展。

清单的 `files_sha256_policy` 为 `utf8_text_crlf_to_lf`：对公开 UTF-8 文本仅将 CRLF 换行转换为 LF 后计算 SHA256，不移除空格、BOM 或末尾换行。这样 Windows/Linux 检出不会造成摘要误报，实际文本变化仍会被检测。Season1 归档内容保持不变，其历史回归测试显式读取归档；历史提案与最终题集清单分开校验。

与 dev 题包相比，公开版统一了索引梯度算子的 schema、Golden 与文档入口；修正布尔值域的 YAML/CSV 类型差异。MoE 校验按 softmax 后的 top-k 语义处理并列与 NaN，已结束 token 的专家索引须使用 `num_expert` 哨兵，确定性的 `row_idx` 参与正常精度比较。

## 平台和性能数据

公开题集提供 910B2、910C、950PR 和 950DT 的性能参考数据。评测按实际平台读取对应数据。

910B2 和 910C 各有 58 题、1160 条公开记录。`embedding_hash_table_lookup_or_insert` 和 `dynamic_mx_quant` 在这两个平台标记为 Unsupported / N/A。评测平台应依据 `suite_manifest.json` 中的名单，将这两题从通过率和得分分母中排除。950PR 和 950DT 各有 60 题、1200 条公开记录。

题集 2.1.0 更新了修订用例的性能参考值，基线估算应用对应平台的最小耗时下限。`task_revision_cases` 列出修订用例，`imputed_baseline_cases` 列出按缺失或退化基线规则补齐的记录。估算记录需在目标平台及软件环境复验。

## 使用和验收

现有评测入口默认读取 `tasks/`。复现旧题集时显式指定 `--task-dir archives/season1/tasks`。Season1 与 Season2 题目和用例不同，成绩应分别标记，不直接混排。

公开前验证文件数量、公开 case ID、YAML/CSV 一致性、等级一致性、Golden 语法和入口、metadata 引用以及原题集归档完整性，并对新增的索引校验契约执行 CPU 回归测试。NPU 编译、全量精度和性能复测属于发布验收，不由上述静态检查替代。
