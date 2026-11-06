# npu_benchmark_bench

从内部 `ascendc-kernelgen-data` 的 `npu_benchmark`（L3/L4）按算子种类筛选出的
算子评测任务集合（按用户最终筛选清单对齐）。

- 来源：内部 npu_benchmark L3（21 个）/ L4（104 个）
- 本 bench：54 个算子（L3 6 个、L4 48 个）
- 目录：`bench_lab/npu_benchmark_bench/level3/<op>`、`level4/<op>`
- 每个算子包含 `proto.yaml` / `golden.py` / `desc.md` / `cases.yaml` / `cases.csv`
- 源实现中“内部随机初始化权重”的算子已把权重提升为显式 proto 输入，保证 golden 确定性
- 每个算子 20 个用例，覆盖 dtype、对齐/非对齐、小/中/大 shape 与 attr 变化

## 与 bench_lab 既有套件的去重

初版 45 题中有 12 题与 `bench_lab` 既有套件（主要是 `cv_agent_fa_bench`、
`omni-ops_bench`、`research_bench`）重合，已移除：

- **同名 6 题**：`bigbird_attention`、`crossformer_attention`、`dense_sparse_attention`、
  `longformer_attention`、`multi_head_latent_attention`、`window_attention`
- **同算子不同名 6 题**：`flash_attention_v2`、`fused_infer_attention_score`、
  `paged_attention`、`block_sparse_attn_fwd`、`gqa_sparse_fwd`、`dsa_lightning_indexer`

同时补入 21 题 `bench_lab` 中无对应的任务：

- **视觉注意力模块 16 题**：DAModule、DoubleAttention、GCModule、GlobalFilter、
  HaloAttention、MUSEAttention、MobileViTAttention、OutlookAttention、PSA、
  Parallel/Sequential PolarizedSelfAttention、S2Attention、SKAttention、
  ShuffleAttention、SpatialGroupEnhance、TripletAttention
- **反向与补齐 5 题**：NsaTopkSparseAttentionBwd、BlockSparseAttnBwd、
  FlexAttentionBwd、Mamba2Bwd、FusedRecurrentGatedDeltaRule

## 覆盖类型

- L3：col2im、IoU、CropAndResize、DeformableOffsets、UpsampleNearest3D、SpatialTransformer
- L4 核心注意力：FusionAttention / SelfAttention / MHA / DecoderMhaAttention /
  ViTAttention / HSTUAttention / FlexAttention
- L4 稀疏与长序列：NSA / MSA / SLA / JaggedDenseFA / JaggedSoftmax / JaggedJaggedBmmBwd /
  Linformer / Nystrom / EMSA / Axial
- L4 线性注意力 / 状态空间：OlmoHybridGdn / Gdn2 / Mamba2Bwd /
  FusedRecurrentGatedDeltaRule
- L4 视觉注意力：BAM / CBAM / CoT / CoordAtt / CrissCross / WeightedPermuteMLP /
  DAModule / DoubleAttention / GCModule / GlobalFilter / Halo / MUSE / MobileViT /
  Outlook / PSA / PolarizedSelfAttention / S2 / SK / Shuffle / SGE / Triplet
- L4 反向：SDPA-Bwd / NSA-Bwd / BlockSparse-Bwd / FlexAttention-Bwd

## 运行

```bash
cd cann-bench
PYTHONPATH=src python3 -m kernel_eval.cli list --task-dir bench_lab/npu_benchmark_bench
```

> 本 bench 先放在 `bench_lab/` 孵化区，避免破坏 `tasks/suite_manifest.json` 的
> Season2 60 题不变量；评审通过后可再 PR 晋升到 `tasks/level3|level4/`。
