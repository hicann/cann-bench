# CANN-Bench算子测试报告

## 1. 概述
本报告覆盖CANN-Bench算子测试Level 1和Level 2两个测试级别的测试结果。本次测试验证了基础算子和中级算子的功能正确性，测试范围包括激活函数、归一化算子、数学运算算子、图像处理算子、优化器算子等多种类型的算子测试用例。

## 2. 版本测试信息

**硬件和版本要求**

- 产品型号：910B3 (NPU) / CPU设备（默认）
- 操作系统：Linux 5.10.0-182.0.0.95.r2673_211.hce2.aarch64
- CANN版本：9.0.0
- 驱动版本：25.5.0
- Python版本：3.12.9
- PyTorch版本：2.7.1+cpu
- torch_npu版本：2.7.1.post2.dev20251226
- 测试设备：CPU
- 测试Repo源：cann-bench

## 3. 测试结论

本版本测试，共计执行480个测试用例，发现0个问题。整体质量良好，满足出口质量标准，建议发布。

- Level 1测试：160个用例，成功率100.00%
- Level 2测试：320个用例，成功率100.00%

## 4. 特性质量评估

|序号|特性|测试结论|功能|精度|性能|可靠性|兼容性|
|---|---|---|---|---|---|---|---|
|1|Level 1基础算子测试|通过|Pass|Pass|Pass|Pass|Pass|
|2|Level 2中级算子测试|通过|Pass|Pass|Pass|Pass|Pass|

### 4.1 Level 1算子测试详情

Level 1测试涵盖8类算子，每类20个用例，共计160个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|MaskedScale|20|20|100%|
|ForeachNorm|20|20|100%|
|Mish|20|20|100%|
|Exp|20|20|100%|
|Sigmoid|20|20|100%|
|SwiGlu|20|20|100%|
|Gelu|20|20|100%|
|ForeachAddcdivScalar|20|20|100%|

执行耗时：56.98s，平均每用例：0.356s

### 4.2 Level 2算子测试详情

Level 2测试涵盖16类算子，每类20个用例，共计320个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|ResizeBilinear|20|20|100%|
|Maximum|20|20|100%|
|ArgMax|20|20|100%|
|Cummin|20|20|100%|
|Gcd|20|20|100%|
|Softmax|20|20|100%|
|GridSampler3D|20|20|100%|
|ApplyAdamW|20|20|100%|
|ApplyRotaryPosEmb|20|20|100%|
|Gather|20|20|100%|
|DynamicQuant|20|20|100%|
|UnsortedSegmentSum|20|20|100%|
|RmsNorm|20|20|100%|
|CrossEntropyLoss|20|20|100%|
|Scatter|20|20|100%|
|GroupNorm|20|20|100%|

执行耗时：173.39s，平均每用例：0.542s

## 5. DFX专项质量评估

### 5.1 安全测试
本次测试为功能验证测试，未涉及安全测试专项。

### 5.2 可靠性测试
|序号|可靠性特性|测试结论|遗留风险|
|---|---|---|---|
|1|算子功能稳定性|Pass|暂无|
|2|算子精度稳定性|Pass|暂无|

### 5.3 性能测试

|场景|算子类型|特性|性能指标|测试环境|测试结果|遗留风险|
|---|---|---|---|---|---|---|
|Level 1|MaskedScale|激活函数|平均6.03ms|CPU|Pass||
|Level 1|ForeachNorm|归一化|平均8.51ms|CPU|Pass||
|Level 1|Mish|激活函数|平均22.33ms|CPU|Pass||
|Level 1|Exp|数学运算|平均10.55ms|CPU|Pass||
|Level 1|Sigmoid|激活函数|平均5.58ms|CPU|Pass||
|Level 1|SwiGlu|激活函数|平均9.65ms|CPU|Pass||
|Level 1|Gelu|激活函数|平均11.94ms|CPU|Pass||
|Level 1|ForeachAddcdivScalar|数学运算|平均10.31ms|CPU|Pass||
|Level 2|ResizeBilinear|图像处理|平均5.76ms|CPU|Pass||
|Level 2|Maximum|数学运算|平均3.42ms|CPU|Pass||
|Level 2|ArgMax|数学运算|平均20.52ms|CPU|Pass||
|Level 2|Cummin|数学运算|平均711.22ms|CPU|Pass||
|Level 2|Gcd|数学运算|平均16.45ms|CPU|Pass||
|Level 2|Softmax|归一化|平均9.76ms|CPU|Pass||
|Level 2|GridSampler3D|图像处理|平均822.80ms|CPU|Pass||
|Level 2|ApplyAdamW|优化器|平均19.95ms|CPU|Pass||
|Level 2|ApplyRotaryPosEmb|位置编码|平均35.72ms|CPU|Pass||
|Level 2|Gather|数据操作|平均26.45ms|CPU|Pass||
|Level 2|DynamicQuant|量化|平均15.71ms|CPU|Pass||
|Level 2|UnsortedSegmentSum|数据操作|平均676.02ms|CPU|Pass||
|Level 2|RmsNorm|归一化|平均8.70ms|CPU|Pass||
|Level 2|CrossEntropyLoss|损失函数|平均7.90ms|CPU|Pass||
|Level 2|Scatter|数据操作|平均5.76ms|CPU|Pass||
|Level 2|GroupNorm|归一化|平均1.90ms|CPU|Pass||

### 5.4 兼容性测试
兼容性评估：通过

|序号|兼容性场景|验证结果|遗留风险|
|---|---|---|---|
|1|CPU设备兼容|Pass||
|2|算子接口兼容|Pass||

## 6. 测试执行评估

### 6.1 测试覆盖

|测试活动|测试结论|用例数|用例覆盖率|用例通过率|
|---|---|---|---|---|
|Level 1算子测试|Pass|160|100%|100%|
|Level 2算子测试|Pass|320|100%|100%|
|特性测试|Pass|480|100%|100%|
|继承特性测试|Pass|480|100%|100%|

## 7. 遗留问题和关键风险
本次测试未发现遗留问题。

### 7.1 遗留问题统计

||问题总数|严重|主要|次要|不重要|已取消|
|---|---|---|---|---|---|---|
|数目|0|0|0|0|0|0|
|百分比|0%|0%|0%|0%|0%|0%|

### 7.2 遗留问题列表

|问题单(issue链接)|问题描述|问题级别|问题影响和规避措施|当前状态|
|---|---|---|---|---|
|无|无|无|无|无|

## 8. 附件

测试结果已保存到：reports/test_results.json