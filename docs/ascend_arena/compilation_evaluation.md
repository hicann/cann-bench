# CANN Bench 编译与评测调用链

本文依据当前仓库代码，描述提交工程如何接入编译和评测。远端骨架仓的其他分支、版本不在本次核对范围。

## 1. 提交接口

评测器接受含 `build.sh` 的源码工程。`PackageManager.build_packages()` 调用一次
`bash build.sh`，把完整输出写入源码目录的 `_compile.log`。编译失败记录为提交失败，
不会自动修改提交源码来补救。已生成的 wheel 也可进入包扫描与安装流程。

源码：[PackageManager](../../src/kernel_eval/data/package_manager.py)。

## 2. ACLNN 示例的编译设施

[build.sh](../../examples/aclnn_launch_example/build.sh) 识别 SoC，依次调用
`scripts/build_run.sh` 和 `scripts/build_wheel.sh`。

- `build_run.sh` 使用 CMake 配置工程，执行构建、`binary` 和 `package` 目标，将 `.run` 放入 `dist/`。
- wheel 构建由 `setup.py` 调用 CMake 编译绑定扩展，形成 Python 可加载的 `cann_bench` 包。
- `.run` 承载自定义算子安装产物，wheel 提供 Python/PyTorch 调用接口。

这是 ACLNN 示例的工程契约；Direct Launch、Triton 等示例有各自的构建实现，不能将它们统称为同一种骨架。

## 3. 安装、正确性与性能

[staged_eval.py](../../src/kernel_eval/staged_eval.py) 的执行顺序：

1. 编译提交工程并确认 wheel 产物，检查编译与运行的 CANN 版本是否匹配。
2. 安装包：存在 `.run` 时先安装 `.run`，再安装 wheel；扫描 `cann_bench` / `torch.ops.cann_bench` 的算子接口。
3. 关闭 Profiler，执行正确性评测，保存正确性结果。
4. 从正确性通过的用例构建允许列表，开启 Profiler 采集性能。
5. 合并正确性和性能结果，生成最终报告。性能阶段还会检查精度，不能用性能结果覆盖正确性失败。

安装阶段使用 API guard 检查；普通 CANN 评测入口要求 `cann_bench_utils`。

## 4. Ascend Arena 的接入方式

Arena 的 `kernel_eval.optimize` 使用主仓固定提交的 `ops-skeleton` submodule，
复制到每次运行的 sandbox，组装原始/候选算子并生成自定义 OPP 和绑定库。
原始与候选评测在独立 worker 进程中执行，以隔离 CANN 的进程级算子注册状态。

直接评测默认采集两份正确性和性能，报告逐 case 加速比；Agent 逐轮评测通过
`--baseline-dir` 复用已采集基线。它使用独立的 smoke、complete 和 profile 用例集。

编译依赖的分发与评测接口各有职责。接入 CANN Bench 的原有流程时，
需要把算子工程适配为其 `build.sh`、安装产物和 `cann_bench` 调用接口，并映射用例、
精度规则及结果字段。
