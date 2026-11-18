# 下一版算子名单与名称迁移（2026-09-24）

> **历史提案**：本文描述 2026-09-24 的选题/命名基线，其中“tasks 仍有 53 项”等表述仅对当时提交成立。当前正式 Season2 已发布至 `tasks/`，以[迁移说明](../guide/season2_migration.md)及 `suite_manifest.json` 为准，不能用本提案 JSON 直接替代运行清单。

## 状态与范围

本名单确定大集 60 项（原主线保留 40 项、新增 20 项），micro 子集 16 项。机器可读名单见 [next_suite_20260924.json](next_suite_20260924.json)。该文件是选题清单，不是评测器运行配置。

基线为官方 master 的 `1d31041fa1b6e2cfaf7caa696c4bae238933e78d`。新增交付件均已进入该提交的 bench_lab；tasks 仍有 53 项。本次只统一 20 个入选 bench_lab 算子的名称、同步对应引用和明确名单去重。整个 tasks 目录与基线完全一致，包含原名称、路径、文档和用例。不迁移 bench_lab 至 tasks，不删除原 tasks 中未入选的 13 项，不改变计算、用例参数或难度声明。

## 命名规则

- 仅入选的 bench_lab 项统一小写下划线。其 operator.name、cases 的 operator 字段和文档标题采用同一标识；tasks 保留现有命名，不在本次更名范围内。
- JSON 的 `id` 是目录标识，不保证等于接口名：`selection: retained` 的 tasks 接口名称仍为 `previous_name`；`selection: added` 的 bench_lab 接口名称为 `id`。全局 `rename_scope` 标明仅调整入选 bench_lab。
- 目录及 schema 对应函数去除 ai_infra 来源前缀；Golden、oracle、bench 钩子与测试引用同步。
- `GgufDequantMatmul` 改为 `q4_0_dequant_matmul`。保留 Q4_0 的块大小、打包和解量化语义，不扩大为任意低比特格式，也不读取 GGUF 文件。
- `ChannelwiseGatedDeltaAttention` 改为 `kimi_delta_attention`（展示名称 Kimi Delta Attention / KDA）。仍只评测逐通道门控的状态递推核心，不扩展为完整模型、Decoder 层或分布式通信任务。
- bench_lab 新标识是本次迁移后的接口；未新增旧标识运行时别名。对应旧提交和历史结果保留原名，使用清单的 previous_name / previous_path 追溯；涉及这些 bench_lab 项的旧提交脚本需要更新路径与入口。tasks 提交脚本不需要因本次 PR 改名。

## 新增项及来源

| 原名称 | 新标识 | 规划等级 | 仓库等级 | 来源 PR |
|---|---|---|---|---|
| ImageResizeNormalize | image_resize_normalize | L2 | L2 | [#291](https://gitcode.com/cann/cann-bench/pull/291) |
| EmbeddingHashTableLookupOrInsert | embedding_hash_table_lookup_or_insert | L3 | L3 | [#14](https://gitcode.com/cann/cann-bench/pull/14) |
| AiInfraAggregateHiddenGrad | aggregate_hidden_grad | L3 | L2 | [#171](https://gitcode.com/cann/cann-bench/pull/171) |
| AiInfraScatterBlockUpdate | scatter_block_update | L3 | L2 | [#171](https://gitcode.com/cann/cann-bench/pull/171) |
| CategoricalEmbeddingBag | categorical_embedding_bag | L3 | L2 | [#291](https://gitcode.com/cann/cann-bench/pull/291) |
| KvCacheAppend | kv_cache_append | L3 | L2 | [#291](https://gitcode.com/cann/cann-bench/pull/291) |
| GgufDequantMatmul | q4_0_dequant_matmul | L3 | L3 | [#291](https://gitcode.com/cann/cann-bench/pull/291) |
| AiInfraManifoldConstrainedHyperConnectionPostGrad | mhc_post_grad | L3 | L2 | [#171](https://gitcode.com/cann/cann-bench/pull/171) |
| AiInfraAggregateHidden | aggregate_hidden | L3 | L2 | [#152](https://gitcode.com/cann/cann-bench/pull/152) |
| DynamicMxQuant | dynamic_mx_quant | L4 | L3 | [#133](https://gitcode.com/cann/cann-bench/pull/133) |
| AiInfraFusedCausalConv1d | fused_causal_conv1d | L4 | L3 | [#171](https://gitcode.com/cann/cann-bench/pull/171) |
| LinearCrossEntropy | linear_cross_entropy | L4 | L4 | [#273](https://gitcode.com/cann/cann-bench/pull/273) |
| SelectiveScan | selective_scan | L4 | L4 | [#273](https://gitcode.com/cann/cann-bench/pull/273) |
| FftConv | fft_conv | L4 | L5 | [#273](https://gitcode.com/cann/cann-bench/pull/273) |
| FlashAttentionScoreGradEnhance | flash_attention_score_grad_enhance | L4 | L4 | [#103](https://gitcode.com/cann/cann-bench/pull/103) |
| ChannelwiseGatedDeltaAttention | kimi_delta_attention | L5 | L4 | [#317](https://gitcode.com/cann/cann-bench/pull/317) |
| CompressedSparseAttentionCore | compressed_sparse_attention_core | L5 | L4 | [#317](https://gitcode.com/cann/cann-bench/pull/317) |
| TensorProgramVm | tensor_program_vm | L5 | L5 | [#281](https://gitcode.com/cann/cann-bench/pull/281) |
| LactTttChunk | lact_ttt_chunk | L5 | L5 | [#278](https://gitcode.com/cann/cann-bench/pull/278) |
| SparseLightningIndexerGradKLLossEnhance | sparse_lightning_indexer_grad_kl_loss_enhance | L5 | L4 | [#103](https://gitcode.com/cann/cann-bench/pull/103) |

规划 Level 数量：L1=2、L2=14、L3=27、L4=12、L5=5。规划等级和 proto.yaml 声明分别记录，不代表 verified_level。两个注意力核心的规划 L5 为本轮名单决定，仓库声明仍是 L4。本次不修改评级规则或 difficulty。

## 重复项处理

`FlashAttentionSparse`（PR #104）不再占独立入选名额，由原主线 `sparse_flash_attention` 代表这类稀疏注意力计算。社区交付目录 `bench_lab/cv_agent_fa_bench/flash_attention_sparse` 完整保留，未从仓库删除。

两者不是可以直接替换的 API：社区版输入共享 KV；主线版分别输入 K/V，支持不同特征维度及额外属性。在合法且无重复稀疏索引、K=V、等特征维度和匹配缩放等条件下核心计算重合。越界/重复索引、布局和属性范围存在差别，因此本次不直接合并 cases 或互相 alias。后续补充覆盖需逐项校验契约。

`aggregate_hidden` 与 `fused_causal_conv1d` 等相近项本次均保留。没有把算法家族相同直接判定为重复。

## micro 更新

micro 保留 12 项、退出 4 项、新增 4 项，仍为 16 项。按规划等级分布为 1/6/5/3/1。

- 新增：`image_resize_normalize`、`dynamic_mx_quant`、`linear_cross_entropy`、`kimi_delta_attention`。
- 退出 micro：`foreach_addcdiv_scalar`、`apply_rotary_pos_emb`、`dequant_swiglu_quant`、`mla_prolog`。后三项仍在大集，退出 micro 不代表删除仓库交付件。
- 完整名单以 JSON 中 `micro: true` 为准。历史 micro 策略文档保留旧版本内容。

## 发布前事项

评测平台选题配置与 bench_lab 到正式 tasks 的迁移需在发布步骤中落实；本清单不会自动切换线上题目。执行环境、精度与 NPU 性能回归须单独验证。本次命名调整不意味着公开实现复用风险降低。

MX 量化按当前 PR #315 交付仅支持 FP8，公告与历史表格中 FP4/FP8 的表述以此修正。
