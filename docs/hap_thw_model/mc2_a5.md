# MC2 A5 T_HW 与性能评分

`compute_mc2_a5.py` 为 level4 中全部显式标记 A5 的 MC2 case 生成
正数 `t_hw_us`，保留原 metadata 中的其他算子。按非量化、量化子集分别计数，
当前覆盖 15 个测评子集、12 个目录、300 个 A5 case（950PR 180 条，950DT 300 条）：

| 算子 | A5 case ID | 数量 | 精度模式 |
| --- | --- | --- | --- |
| matmul_all_reduce | 21–60 | 40 | 21–40 INT8；41–60 非量化三方 |
| all_gather_matmul | 21–40 | 20 | 非量化三方 |
| all_gather_matmul_v2 | 21–40 | 20 | FP8 三方 |
| allto_all_matmul | 1–40 | 40 | 1–20 非量化；21–40 MX FP8 三方 |
| matmul_allto_all | 1–40 | 40 | 1–20 非量化；21–40 MX FP8 三方 |
| matmul_reduce_scatter_v2 | 21–40 | 20 | FP16/BF16 三方 |
| grouped_mat_mul_allto_allv | 21–40 | 20 | 950DT，FP16/BF16 三方 |
| allto_allv_grouped_mat_mul | 1–20 | 20 | 950DT，FP16/BF16 三方 |
| quant_all_reduce | 1–20 | 20 | 950DT，INT8/FP32 scale 三方 |
| quant_reduce_scatter | 1–20 | 20 | 950DT，INT8/FP32 scale 三方 |
| quant_grouped_mat_mul_allto_allv | 1–20 | 20 | 950DT，MX FP8 三方 |
| allto_allv_quant_grouped_mat_mul | 1–20 | 20 | 950DT，MX FP8 三方 |

## 硬件数据来源与适用范围

来源：华为技术有限公司《昇腾950 NPU架构白皮书》（2026），
第 3 章表 3-1（PDF 第 13–14 页）与互联描述（PDF 第 11–13 页）。
可通过[昇腾官方文档中心](https://www.hiascend.com/document)查阅。
记录白皮书中的 Cube 独立峰值，不使用 Cube+Vector 总算力代替 GEMM 峰值。

数值的唯一代码来源是 `hap_platform_constants.public.json` 中公开的 A5 子集；
`hap_lib.py` 统一加载并计算 Cube、HBM 读及通信分量，MC2 脚本负责算子资源计数。
同名外部完整配置若与公开系列常量冲突会报错，不静默切换规格。
完整 L2/Vector 模型需要的其他常量仍需外部提供，不能由本表推断。

| metadata 平台 | FP16/BF16 Cube TFLOPS | INT8/FP8 Cube TOPS/TFLOPS | HBM TB/s | 互联峰值 TB/s |
| --- | ---: | ---: | ---: | ---: |
| 950pr | 432 | 865 | 1.6 | 2 |
| 950dt | 486 | 973 | 4 | 2 |

950PR/950DT 通用配置取各系列公开最高规格，因此对裁剪 SKU 是偏乐观的理论下界。
所有 950DT 子型号统一使用 `950dt.json`，按通用系列口径计算 T_HW 与 HAP，
不再维护子型号专用理论值。该口径不代表各子型号硬件规格或实测性能等价；
对算力较低的 SKU，通用峰值给出的 T_HW 是更乐观的理论下界。
PR 与 DT 不共用 HBM 常量，也不映射到 A2。

2 TB/s 是芯片聚合互联峰值，不代表两卡间的实际单向链路带宽。
本模型将它作为远端发送字节数的乐观带宽上界；未计协议、拓扑瓶颈、重传、
启动延迟和收发双向竞争，因此生成的是可审计的资源理论下界，**不是实测性能基线**。

## 公式与计数

沿用 HAP 的资源完美重叠、unique input bytes 和 1 微秒下限约定：

```text
T_HW = max(1 us, Cube_ops / Cube_peak,
           local_input_bytes / HBM_peak,
           remote_send_bytes / link_peak)
Cube_ops = 2 * GEMM_M * GEMM_N * GEMM_K
```

各除法换算为微秒。数值向下截断至 6 位小数，避免舍入抬高下界。
这里只包含能够从公开规格支持的 Cube、HBM 读、通信分量；不伪造缺失的 L2
写带宽或 vector/cast 常量。略去这些分量会放宽下界，而不是假设这些操作不耗时。
HBM 采用现有 HAP 的冷输入口径；热缓存测量可能低于这个建模分量，应结合测量口径解读。

M 是 x1 前导维度乘积，K 是 x1 最后维度，N 按 x2 转置标志还原。
所有数值是每 rank 的资源计数，不把 world_size 再乘到单卡算力分母。

| 算子 | 每 rank GEMM 形状（M,N,K） | 两卡时每 rank 远端发送字节 |
| --- | --- | --- |
| MatmulAllReduce | M,N,K | M*N*output_element_bytes |
| AllGatherMatmul | 2M,N,K | local_x1_bytes |
| AllGatherMatmulV2 | 2M,N,K | local_x1_bytes |
| AlltoAllMatmul | M/2,N,2K | local_x1_bytes/2 |
| AlltoAllQuantMatmul（MX FP8） | M/2,N,2K | (local_x1_bytes + local_x1_scale_bytes)/2 |
| MatmulAlltoAll | M,N,K | M*N*output_element_bytes/2 |
| QuantMatmulAlltoAll（MX FP8） | M,N,K | M*N*output_element_bytes/2 |
| MatmulReduceScatterV2 | M,N,K | M*N*output_element_bytes/2 |
| AlltoAllvGroupedMatMul | 每专家接收行数,N,K；加可选共享 GEMM | 远端 token 行数*K*element_bytes |
| GroupedMatMulAlltoAllv | 每专家本地行数,N,K；加可选共享 GEMM | 远端 token 行数*N*element_bytes |
| QuantAllReduce | 无 Cube 矩阵乘 | local_x_bytes + local_scale_bytes |
| QuantReduceScatter | 无 Cube 矩阵乘 | (local_x_bytes + local_scale_bytes)/2 |
| QuantGroupedMatMulAlltoAllv | 每专家本地行数,N,K；加可选共享 GEMM，FP8 Cube | 远端 token 行数*N*output_element_bytes |
| AlltoAllvQuantGroupedMatMul | 每专家接收行数,N,K；加可选共享 GEMM，FP8 Cube | 远端 token 行数*(K+2*ceil(K/64)) |

量化 scale、bias、x3 等非空输入均计入本 rank unique 读字节。
GMM 用例通过 C[source][destination][expert] 明确每条通信边的行数，
远端字节不包含本 rank 自发自收的数据；按各 rank 最大资源量建模，
不假设非均匀分布必定发送一半 token。每个用例的所有 rank 总行数保持一致。
GMM 的 host counts 列表不计入设备输入读字节，有 token 的专家权重、gmm_x、非空共享矩阵均计入；空专家权重不计入必需读取量；
共享矩阵乘的 FLOPs 与专家分组矩阵乘相加。排列与调度开销沿用本模型省略未知 Vector 分量的口径。
QuantAllReduce / QuantReduceScatter 传输 INT8 x 和 FP32 分组 scale，
不是级联参考中已经反量化的 FP32 张量；两卡 QAR 发送一份完整本地输入，
QRS 发送远端一半。它们的 Cube 分量为零，Vector 反量化/归约与输出转换
沿用公开模型省略未知分量的口径，并不表示这些操作不耗时。
MX GMM 的输入/权重按 FP8 的1字节计，E8M0 scale 按1字节计，空专家的权重及权重scale不计必需读取；
QuantGroupedMatMulAlltoAllv 通信 FP16/BF16 输出，按2字节计；
AlltoAllvQuantGroupedMatMul 通信 FP8 输入及 E8M0 scale，K 尾部按 scale 的实际填充形状计数。
共享矩阵乘也使用 FP8 Cube 峰值。

当前模型严格要求 world_size=2；修改用例卡数或新增算子必须补充资源模型后重生成。

GroupedMatMulAlltoAllv、QuantGroupedMatMulAlltoAllv 的本批 A5 用例用于小规模功能与精度覆盖，
其理论值均落在 1μs 下限，不能区分部署规模的资源利用率。两卡测量不代表多卡扩展性能。

## 与实测和评分的关系

MC2 runner 对候选融合算子和 NPU 级联基线使用相同 wall-clock 测量方法，
跨 rank 分别取最大耗时。其数值是分布式端到端耗时（包含 Python 调度、同步），
不是 profiler 单 kernel Task Duration。rank0 profiler 数据另行归档。

复算仅为用例 `supported_devices` 声明的平台生成条目，不向其他芯片推断适用性。

MC2 runner 优先读取 metadata 中正数 `baseline_perf_us`；缺失时取同次测量的
NPU 级联耗时。最终报告合并保留它以及 `t_hw_us`，理论下界与参考耗时分别维护。

本批固定 `baseline_perf_us` 均为对应平台的实测值，与理论 `t_hw_us` 分开维护：

- **950DT：300 条实测值。** 来源是 ops-transformer 主线在 Ascend950DT_9582、
  两卡环境的历史报告。260 条来自 JSON 的 `operators[].cases[].elapsed_us`，
  40 条来自 Markdown 耗时列；不读取报告内的 `baseline_perf_us` 作为新实测值。
  Markdown 报告的设备子型号及卡数由报告提供者确认，耗时保留原报告两位小数。
- **950PR：180 条实测值。** 覆盖同时支持 950PR/950DT 的 9 个 A5 子集；
  六个仅支持 DT 的子集不填 PR 数据。本次修订
  更正此前错误的推导代理标注，保留全部既有 `baseline_perf_us` 和 `t_hw_us` 数值。
  950PR 基线不由 DT 耗时、带宽比例或资源模型拟合生成。

950DT 的 `_metadata.baseline_source` 记录设备、报告时间、取值方法和仓外报告相对路径。
JSON 报告记录 CANN 9.2.0，Markdown 报告未记录 CANN 版本。
历史报告未记录的 `source_revision`、`warmup`、`repeat`、`aggregation` 保持 `null`，
不以当前 runner 的配置回填；报告时间不等于采样起止时间。原始报告未随代码发布，
现有来源信息可追溯交付数值，但不足以精确重放原采集过程。

950PR 的 `_metadata.mc2_baseline_source.kind=measured` 单独记录实测声明，
`provenance_evidence` 记录实测说明；`case_ids`、`case_sha256` 和
`baseline_sha256` 记录适用用例与交付数值的完整性。本次未提供可逐条核对的原始报告，
设备子型号、历史卡数、实现版本、CANN 版本、采样日期、方法、warmup、repeat、
aggregation 及报告路径均保留 `null`。用例配置中的 `world_size=2` 不补作历史采样证据。
同文件的 `_metadata.baseline_source` 和 `generated_at` 仅描述既有非 MC2 实测项，
其中的 warmup、trials 及统计口径不适用于这 180 条 MC2 实测值。
来源声明和数据指纹检查不等于原始报告核对，也不构成重新实测或精确重放。

两平台的 `_metadata.mc2_hap` 记录理论模型、公开硬件常量、用例指纹与数量。
`T_HW` 仍是理论资源下界，不能作为实测耗时或用来生成、覆盖、舍入修正固定基线。
在仓库根目录执行以下命令，可在无 NPU 环境分别检查理论模型与实测数据完整性：

```bash
# 复算理论 T_HW 及其模型指纹，不修改 metadata
python docs/hap_thw_model/compute_mc2_a5.py
# 只读检查 950PR 实测声明、用例覆盖及数值指纹，不复算或认证实测来源
python docs/hap_thw_model/compute_mc2_pr_baselines.py
```

`compute_mc2_pr_baselines.py` 保留历史文件名以兼容现有检查命令，但已移除代理拟合、
基线生成及 `--write` 入口，也不读取 950DT 实测值。实测基线发生变化时，应依据对应
950PR 的实际测量记录更新来源、数值及指纹；检查不通过时不能用资源模型自动修复。
经确认需要更新理论模型时，可以显式生成理论数据：

```bash
python docs/hap_thw_model/compute_mc2_a5.py --write
```

该命令仅更新 A5 理论 `t_hw_us` 及 `mc2_hap`，不改写两平台的实测 `baseline_perf_us`。

精度成功不等于性能采集完成。最终 Markdown 的相应用例会显示
“精度 / 性能未完成”；JSON 的 `perf_recheck` 保留失败原因。
修复后的单 case 超时同时覆盖执行与退出回收，外层保留清理预算。
A5 rank 初始化时默认 `HCCL_NPU_SOCKET_PORT_RANGE=auto`，尊重用户显式设置；A2 不自动设置。
该取值及 Ascend 950PR/950DT 适用范围见[官方 HCCL 环境变量文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md)。
