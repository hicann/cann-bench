# MC2 A5 用例与三方精度验证

## 测评范围

任务位于 `bench_lab/kernel_bench/level4`，共 **15 个 A5 子集、12 个任务目录、300 条用例**：
非量化 7 个子集、140 条，量化 8 个子集、160 条。三个任务各含非量化与量化子集，
因此注册日志显示 12 个算子接口，不是遗漏注册。

| 任务目录 | A5 非量化 case ID | A5 量化 case ID | 保留的历史 case ID | A5 平台 |
| --- | --- | --- | --- | --- |
| matmul_all_reduce | 41–60 | 21–40，INT8 | 1–20 | 950PR / 950DT |
| all_gather_matmul | 21–40 | — | 1–20 | 950PR / 950DT |
| all_gather_matmul_v2 | — | 21–40，FP8 per-tensor | 1–20 | 950PR / 950DT |
| allto_all_matmul | 1–20 | 21–40，MX FP8 | — | 950PR / 950DT |
| matmul_allto_all | 1–20 | 21–40，MX FP8 | — | 950PR / 950DT |
| matmul_reduce_scatter_v2 | 21–40 | — | 1–20 | 950PR / 950DT |
| grouped_mat_mul_allto_allv | 21–40 | — | 1–20 | 950DT |
| allto_allv_grouped_mat_mul | 1–20 | — | — | 950DT |
| quant_all_reduce | — | 1–20，INT8 / group-128 scale | — | 950DT |
| quant_reduce_scatter | — | 1–20，INT8 / group-128 scale | — | 950DT |
| quant_grouped_mat_mul_allto_allv | — | 1–20，MX FP8 | — | 950DT |
| allto_allv_quant_grouped_mat_mul | — | 1–20，MX FP8 | — | 950DT |

这些目录共保留 100 条历史用例，总计 400 条；历史规格与 A5 用例不能混为同一覆盖范围。
MatmulAllReduce、AllGatherMatmul 的历史 1–20 保留原输入生成方式和 910b2 基线，
A5 非量化用例及其性能值分别迁移至 41–60、21–40。

本批 A5 用例均为两卡，覆盖接口、数值精度及该拓扑下的性能，不代表多卡扩展性能。
GroupedMatMulAlltoAllv、QuantGroupedMatMulAlltoAllv 的 A5 子集定位为小规模功能和精度覆盖；
其 `T_HW` 均落在 1μs 下限，不能据此评价部署规模的算力利用率或通信重叠能力。
更大规模和更多卡数的性能用例需使用新增 ID 及独立实测基线。

## 三方数据流

所有 A5 用例显式启用 `attrs.mc2_accuracy_mode: three_way`，未启用的历史用例保持原流程。

1. 候选执行前将各 rank 输入复制到 CPU。`mc2_cpu_golden` 以 FP32 计算独立参考，
   并用 Gloo 汇总实际 rank 数据。INT8 矩阵乘先以 INT32 累加，再以 FP32 反量化。
2. NPU 小算子级联通过 Matmul / QuantMatmul 和 HCCL 生成设备参考，保留输出舍入。
3. 候选和设备参考分别与 CPU 参考比较，再比较两组误差；所有 rank 的所有输出均需通过。

QuantAllReduce / QuantReduceScatter 没有矩阵乘：各 rank 先按每 128 个 H 元素的 scale
进行 FP32 反量化，再求和并转换输出类型。不同 rank 的 scale 可以不同，不能改为先合并 INT8
再乘一个 scale。ReduceScatter 输出展平为二维。

MX FP8 路径按每 32 个 K 元素的 E8M0 scale 解码，包含实际 scale 填充和可选输出；
AlltoAllQuantMatmul、QuantMatmulAlltoAll 共用非量化任务目录，由用例量化属性选择调用路径。

缺少 CPU hook、Gloo 不可用或参考失效不会降级为两方比较。
CPU 输入快照、参考计算和精度比较不计入候选性能耗时。

### 量化归约通信资源

QuantAllReduce / QuantReduceScatter 使用专用候选 HCCL 域，该域不执行普通集合通信、
计时 barrier 或 profiler 同步。默认 HCCL 域用于级联及同步，Gloo 域用于 CPU 参考。
候选应使用传入的 `hcomm_info` 建立对应 context，并将实际通信 buffer 大小传入 kernel，
不得在 wrapper 中忽略该参数、转而取得默认通信组。

当前接口可参考 `ops-transformer/mc2/common/torch_extension/comm_context.py` 的
`CommContextManager(hcomm_info, world_size, backend={"Ascend950": "channel"}, opName=...,
customCclBufferSize=...)`。保守 buffer 容量为
`2 * (world_size * (align32(x_bytes) + align32(scales_bytes)) + 1048576)` 字节，
其中 `x_bytes=T*H`，`scales_bytes=T*(H/128)*4`。context 与窗口在重复调用期间保持有效，
容量变化或释放前同步设备。

## 精度标准

设 C 为候选对 CPU 的误差，B 为级联对 CPU 的误差。原倍率判定保留：

| 指标 | 通过条件 |
| --- | --- |
| diff_big_max | C ≤ 10 × B |
| diff_big_avg / diff_big_sum / diff_rmse | C ≤ 2 × B |
| err_small_num | 小值总数 ≥ 100000 时，C ≤ 2 × B |

为避免 B=0 退化为逐位相同要求，增加有限值逐元素容差分支：
`abs(candidate - golden) <= atol + rtol * abs(golden)`。
FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，
`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。原倍率规则或全体有限元素容差满足即可，
但结构、非有限值和小值错误数门禁始终有效，不以 epsilon 放宽错误计数。

有效比较 dtype 决定容差：AlltoAllMatmul / MatmulAlltoAll 显式按 FP32 比较，
FP8 gathered 辅助输出也按 FP32 比较，不能套用 FP8 的宽容差。
小值界限仍为 FP16/BF16 的 `1e-3` 和 FP32 的 `1e-6`；大值按 ≥、小值按 < 计数，
小值错误掩码保留包含边界的行为。

级联参考先接受候选无关的独立质量检查。未提供任务级 CPU 分阶段舍入参考时，
按真实输出 dtype 使用仓库固定精度标准与 CPU FP32 参考比较；FP8 辅助输出按 FP32 检查。
两卡低精度 MC2 GEMM 提供独立 CPU 分阶段舍入参考 H 时，参考健康检查使用
与三方比较相同的 MC2 指标，要求 NPU 级联 B 相对数学 FP32 参考 G 的误差
不超过 H 对 G 误差的相应倍率；同时要求 H 对 G、B 对 H 的归一化 RMSE
均不超过输出 dtype 的相对容差，以拦截系统性错误的 CPU 分阶段实现或 NPU 级联。
不使用逐点相对误差均值约束 H：带符号归约在接近零的位置会发生正常相消，
该约束会重新引入参考误判。分阶段 hook 的语义还须通过独立单元测试和真机回归确认；
全局 RMSE 检查并不能单独证明每个位置的参考值正确。
结构、shape 不一致及数学 CPU FP32 参考 G 的非有限值始终拒绝。
CPU 分阶段参考 H 和 NPU 级联 B 的异常值位置、符号必须分别满足输出 dtype 的范围契约：
允许可预期的低精度溢出 Inf，仍拒绝 NaN、意外或缺失的 Inf、位置或符号错误。
含预期溢出时，只在 G/H/B 的共同有限位置计算健康检查误差，避免 Inf−Inf 和溢出值稀释有限误差；
健康检查不读取候选输出、不接受候选自定义阈值，也不更改候选对 B/G 的评分倍率。
该分阶段策略避免通用逐点小值/相消绝对计数在百万级带符号 GEMM 上误判少量
正常舍入差异；报告记录 H/B 各自的误差、倍率上限与未通过指标。
FP32 输出的 H 与 G 逐位相同时，普通逐元素容差内的 B 不会仅因误差同向而被拒绝。
超出该容差、需要额外 FP32 参考精度额度时，继续检查偏置比例；归一化 RMSE、最大误差、
非有限值和小值错误数等原有门禁保持有效。该额度只用于参考健康，不放宽候选比较。
参考失效记为 `reference_invalid`，此次评分不可用并返回非零，不计为候选编译/运行失败。
正常的 FP16/BF16 输出舍入不得被误判为参考失效。

两卡非量化 MatmulAllReduce、AlltoAllMatmul、MatmulAlltoAll、MatmulReduceScatterV2
通过任务内 `mc2_cpu_references` 返回数学 FP32 G 及独立 CPU 分阶段舍入 H。
H 按各算子实际的 matmul、bias、归约或通信边界舍入；其中 AlltoAll 的纯搬运
不引入数值舍入。H 不读取候选或设备级联输出，也不参与候选误差倍率计算。
只有参考健康检查使用 H，报告标记 `reference_check.reference=cpu_staged_cascade`。
候选与设备级联的三方误差、倍率及评分仍基于原数学 FP32 golden，不将其替换为舍入后的结果。
该 hook 不参与性能计时；未提供 H 的任务及其他卡数保留原有固定精度参考自检方式。

两卡量化 MatmulAllReduce 的 INT8→FP16/BF16 路径也提供 H：INT32 GEMM 与 bias
之后按 channel 再 token 进行 FP32 反量化，转换到输出 dtype；可选 x3 加法后再次转换，
两卡 SUM 后再转换。数学 G 保留原 token→channel、FP32 x3 和 FP32 SUM 的计算顺序。
这修复了量化分支缺失 H 而以通用逐点标准误判正常阶段舍入的问题，不放宽候选精度。
不能把双卡模型直接推广到多卡归约树；未建模的卡数或 dtype 继续使用原检查。

输出数量、shape、dtype、None 契约及 NaN/±Inf 掩码必须匹配；非有限误差指标拒绝通过。
报告保留逐 rank 的 candidate_metrics、cascade_metrics、原倍率 limits、失败指标及容差通过依据。

## 选取与执行

执行前按实际硬件过滤 `supported_devices` 不匹配的用例，过滤项不进入总用例数和评分分母；
运行期仍进行防御检查。离线目录浏览不按假定硬件过滤。`platform` 是说明文本，不能代替
`supported_devices`；共用目录仍应按上表选择 ID 区间，以区分历史、非量化和量化测评。

在 Linux 的 cann-bench 根目录执行，例如 A5 非量化 MatmulAllReduce：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPU_ARCH=ascend950
OPS_TRANSFORMER="$(cd ../ops-transformer && pwd)"

./scripts/run_evaluation.sh \
  --source-dir "$OPS_TRANSFORMER" \
  --source-build-args "--pkg --soc=ascend950 --vendor_name=custom_mainline_matmul_all_reduce --ops=matmul_all_reduce -j16" \
  --task-dir bench_lab/kernel_bench/level4/matmul_all_reduce \
  --operator MatmulAllReduce --case-id-range 41-60 \
  --processes-per-card 1 --timeout-per-process 3600 --build-timeout 7200 \
  --reports-dir reports/a5/mainline/matmul_all_reduce
```

该源码目录必须具备 cann-bench 构建及加载适配；脚本会构建、安装交付件并加载 custom vendor
环境，候选由 `cann_bench.<函数>` 或 `torch.ops.cann_bench.<函数>` 提供。
不要指定单卡 `--device-id`；MC2 独占父进程可见的整组卡。每个算子、子集使用独立报告目录。

默认运行超时为 300 秒、构建超时为 600 秒。构建超时支持 shell 和 Python 两个入口，
优先级为 `--build-timeout` > `CANN_BENCH_BUILD_TIMEOUT_SEC` > 默认值；只接受正整数。
`--source-build-args` 仅在指定时向提交目录 `build.sh` 透传参数。

外部调度器需要隔离并行任务时，可设置 `MC2_OPP_INSTALL_ROOT` 为本任务的绝对安装目录；
它只改变候选 `.run` 包的安装目标，不替换编译依赖的系统 `ASCEND_OPP_PATH`。
还需分别隔离源码构建目录、Python 环境与报告目录；框架不内置模型候选清单或模型分支。
`BENCH_DEVICE_VISIBILITY` 保存分配给本任务的原始物理卡集合，MC2 子进程启动前恢复该集合，
避免设备初始化把物理卡号重写为逻辑卡号后，不同任务误用同一组卡。
可见卡数须不少于用例的 `world_size`，多余可见卡不增加该用例的 rank 数。

分阶段评测中，`--no-perf` 仅验证精度。启用性能时，精度通过但性能未完成会保留精度结果并返回非零；
报告明确标出性能未完成原因。退出码表示请求的流程是否完成，不仅表示精度是否通过。

## 性能基线与验证

本批 MC2 A5 的固定基线按平台区分来源：

| 平台 | A5 固定基线 | 来源与用途 |
| --- | ---: | --- |
| 950DT | 300 条 | ops-transformer 主线在 Ascend950DT_9582、两卡环境的实测报告值 |
| 950PR | 180 条 | 950PR 实测值；保留全部实测数值 |

950DT 的 `_metadata.baseline_source` 记录报告、设备及已知采集信息。历史报告缺失的
源码提交号、warmup、repeat、aggregation 保持 `null`，不以当前默认值回填；
原始报告未随代码发布，现有信息不足以精确重放历史采集。
950PR 的 `_metadata.mc2_baseline_source.kind=measured` 单独记录实测来源、
9 个子集的用例范围和数值指纹；同文件的既有非 MC2 实测条目保留原来源。
本次未提供可逐条核对的 PR 原始报告；设备子型号、历史卡数、实现及 CANN 版本、
日期和采样参数等未知信息保留 `null`，不沿用非 MC2 数据或当前 runner 的默认配置。
无 NPU 的只读检查验证声明、用例覆盖与数据完整性，不重新采集或认证测量真实性。

当前 MC2 runner 的耗时口径为重复批次 wall time / repeat，再取各 rank 最大值，包含调度与同步，
不是 profiler 单 kernel Task Duration 中位数；该说明不补全历史报告缺失的采样配置。
`T_HW` 与固定性能基线分别维护，理论模型及实测来源字段见
[MC2 A5 模型说明](../hap_thw_model/mc2_a5.md)。

新报告的每个用例通过 `baseline_source` 分别记录 `baseline_perf_us` 和 `t_hw_us`
的请求平台、实际来源平台、metadata 文件和 `fallback_used`。两项数值的来源可能不同。
沿用框架原有的默认平台回退及评分规则；跨平台取值会在 Markdown/HTML 中标为
“跨平台复用”，不会恢复逐用例的终端提示。来源文件不等于实测声明，实测声明仍以
对应 metadata 为准。MC2 在固定基线缺失时使用本轮参考路径计时，标为
`source=runtime_reference`，采样口径见同一用例的 `accuracy.metadata.measurement`。
旧报告未记录来源时不补造声明；缺失数值仍保留原框架的评分兜底行为。

```bash
python -m pytest tests/ut -q
# 自检理论 T_HW 和来源指纹，不修改 metadata
python docs/hap_thw_model/compute_mc2_a5.py
# 只读自检 180 条 950PR 实测声明、用例覆盖和数值指纹
python docs/hap_thw_model/compute_mc2_pr_baselines.py
```

`compute_mc2_pr_baselines.py` 不再支持 `--write`，不会从 DT 或理论模型生成 PR 实测值。
`compute_mc2_a5.py --write` 仅更新理论 `t_hw_us` 及模型信息，不改写实测基线。
静态检查和 CPU UT 不替代 A5 环境下的编译、HCCL、精度及性能回归。
