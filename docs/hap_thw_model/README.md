# HAP `t_hw` 计算逻辑（公开版）

本目录公开 cann-bench 中 **`t_hw`（Hardware-Anchored Performance，硬件锚定理论下限耗时）** 的计算方法，包括共享资源模型库 `hap_lib.py`、硬件常量模板，以及算子的资源计数与复算脚本。

## 模型

```
t_hw = max(t_cube, t_vector, t_hbm_read, t_write)   [微秒]
```

`t_hw` 是 roofline 理论**下界**：所有 engine 满峰值、完美 overlap、无 launch 开销，物理上不可达，用作性能地板。

- **读**：`input_bytes / HBM_BW`（kernel 启动时 L2 为空，输入全部来自 HBM；只计 unique bytes，无 reload 因子）
- **写**：L2-fill-first 分段（先按 L2 总线带宽填满 L2 容量，溢出部分走 HBM；容量判定只看 output bytes）
- **cube**：`FLOPs / cube_peak_for_dtype`（matmul 记 `2*M*N*K`）
- **vector**：`元素操作数 / vector_rate(op_kind, dtype)`；**reduction 按 flat 计**（`N/peak`，**无 log2**——log2 是树深/延迟，不是吞吐代价）；**scan/prefix 才有 log2**（`N*ceil(log2(N_scan))`）
- cast 是真实 vector 分量；FixP 写回旁路（dtype 转换/scale/bias）免费
- cube 与 vector 并行（取 max）；同一 engine 内的分量串行（求和）
- 下表代表脚本的收尾约定：`t_hw_us = max(raw, 1.0)`（1µs clip）；`baseline_cap_us = max(10*t_hw_us, 10.0)` 是对应发布基线的**封顶值**——`baseline_perf_us = min(实测, baseline_cap_us)`（见 `tasks/metadata/910b2.json` 的 `_metadata` 块），仅在无实测处直接取封顶值。代表脚本输出的是封顶值，单凭 `t_hw` 不能复现实测基线；其他测评集的固定基线来源与处理方式见各自模型说明。

## 代表算子

| 算子 | level | 展示的机制 |
|---|---|---|
| swi_glu | level1 | 逐元素 + SFU(sigmoid) + fp16/bf16 cast 规则；输入两半读、输出减半写 |
| apply_adam_w | level2 | 多张量读（4 输入）、fp32 计算 + 5 cast、read-bound |
| arg_max | level2 | flat 归约（无 log2）、int64 索引小输出写 |
| softmax | level2 | fp32 计算链（max-reduce/sub/exp/sum-reduce/div）+ 双 cast，vector-bound |
| strided_slice | level3 | 纯数据搬运：fetched-bytes 规则、L2-fill-first 写、无计算算子的瓶颈标注 |
| conv_2d | level3 | cube FLOPs 计数、fp32 卷积的 HF32/TF32 路径（`cube_hf32`）、FixP 免费 bias |
| mha | level4 | 双 matmul cube + fp32 softmax vector、因果三角精确计数 |
| mla | level4 | 分裂 QK 头维的双 matmul、GQA/共享 latent KV 的 unique-bytes 读、显式 cast |

每个 `compute_<op>.py` 的 docstring 给出该算子的完整分量分解（读/写字节数、cube FLOPs、vector 逐操作计数及依据）。

## 分解的依据与校验

上表代表算子脚本内置**开源用例复算**：对仓库开源 1–20 用例计算 `t_hw`，与 `tasks/metadata/910b2.json` 已发布值对比（门限：每条 ≤2% 相对误差或 ≤0.05µs 绝对误差），直接运行即可复现。其他脚本的校验对象与门限以各自说明为准。

分解中的计数分两类，脚本里明确区分：**由算子语义直接推导**的（绝大多数：字节数、FLOPs、逐操作 vector 计数），开源复算对它们是独立复核；以及**少数按开源用例标定的经验约定**（如 mha 的 softmax 不计 cast 而 mla 计 cast），这类在脚本中以 "empirical convention validated on the open cases" 字样标出，开源复算对它们是一致性检查而非独立验证。

## 硬件常量与公开范围

`hap_platform_constants.public.json` 是运行时使用的公开常量入口，包含有白皮书出处的 Cube、HBM 和互联峰值子集，来源与适用范围见对应模型说明。各计算脚本通过 `hap_lib.py` 读取共享常量，不单独维护同名平台数值。

`hap_platform_constants.template.json` 仅用于创建外部完整配置，不作为运行时数据读取。完整模型需要的 L2、Vector、cast 等常量，以及其他平台常量，仍由外部 `hap_platform_constants.json` 提供。模板中的未知项保持 `null`，不以其他芯片数据代填。加载时检查完整性；外部配置若与同名公开系列配置冲突会明确报错，避免同一平台静默使用两套硬件规格。硬件规格请查阅官方文档：

> https://www.hiascend.com/document

## 运行方式（三种状态）

1. **不提供完整常量文件**（默认）：仅依赖公开常量子集的计算与复算可运行；需要 L2/Vector 等未知常量的计算明确报错。各专项模型应说明省略分量及适用限制，仅使用部分分量的结果不代表完整性能预测。
2. **提供完整常量文件**：填写原有 `910b2`、`950pr`、`910c` 三个平台块（`vec_basic` 必须含 `float32` 回退键）。`950dt` 是可选完整平台：未取得完整数据时删除该块，取得数据后补齐。`compute_both()` 默认计算原三平台，并在 DT 完整配置存在时纳入 DT；也可显式传入 `platforms=("950dt",)`。DT 的 L2 驻留写瓶颈标为 `Unconfirmed`。各代表脚本的历史复算对象仍以各脚本说明为准。
3. **闭源用例**：`inner/tasks/` 下的闭源 21–100 用例不随仓库公开；代表脚本在开源复算后检测到其缺失会打印说明并正常退出。

各脚本的复算与校验范围以对应说明为准，**不代表已统一复算 `tasks/metadata` 全量数据，也不补造未覆盖算子的基线**。完整模型数据更新需要相应常量、算子计数和测量依据。

## 目录

单一平铺目录，无子目录：

```
hap_thw_model/
├── README.md
├── hap_lib.py                             # 共享分量公式 + 汇总/收尾
├── hap_platform_constants.public.json     # 有公开出处的运行时常量
├── hap_platform_constants.template.json   # 外部完整配置模板
├── compute_mc2_a5.py                      # MC2 资源计数与复算
├── mc2_a5.md                              # MC2 模型说明
└── compute_<op>.py                        # 代表算子的分量分解（见上表）
```
