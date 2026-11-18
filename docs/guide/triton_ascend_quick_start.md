# Triton-Ascend 快速验证

这份文档面向不熟悉算子开发、但需要确认 CANN Bench 能否运行 Triton 算子的开发者。

## 它是怎么工作的

Triton 算子仍然提交为普通的 `cann_bench` Python wheel：

```text
cann_bench.exp(x)
  -> Python wrapper 分配输出 Tensor
  -> _exp_kernel[grid](...) 首次特化时按需 JIT，后续复用缓存
  -> kernel 在 Ascend NPU 上执行
  -> CANN Bench 对比 Golden 并采集 Profiler 数据
```

CANN Bench 不需要另一套 Triton runner。评测镜像负责提供与 CANN 匹配的
Triton-Ascend 编译器和运行时。完整调用链和源码见
[CANN Bench Triton-Ascend 集成实现与运行原理](../design/triton_ascend_integration_implementation.md)。

## 1. 构建环境

下列 CANN 9.0.0 + Triton-Ascend 3.2.1 命令保留为历史独立 JIT smoke 的环境记录，不代表当前完整评测流程的最低配置。当前 `docker/dev` 默认 CANN 9.1.0；运行 CANN Bench 的 ACLNN 样例和支持 kernel 前，应按[当前环境指南](quick_start.md)及 [Docker 配套说明](../../docker/dev/README.md)重新确认软件组合，并完成目标硬件验证。不能把独立 Triton smoke 成功当作完整 harness 验收。

```bash
cd docker/dev
docker build --network=host \
  --build-arg CANN_VERSION=9.0.0 \
  --build-arg DEVICE=950 \
  --build-arg TRITON_ASCEND_VERSION=3.2.1 \
  -t cann-bench:cann9.0.0-950-triton3.2.1 .
```

910B 环境把 `DEVICE=950` 改成 `DEVICE=910b`。

## 2. 运行环境 Smoke

```bash
IMAGE=cann-bench:cann9.0.0-950-triton3.2.1 bash run.sh smoke
```

这个 smoke 不只检查 import。它会真实 JIT 编译并运行一个 vector add。成功时应看到：

```text
[OK]   [5] triton-ascend 3.2.1, target=..., vector add passed
ALL CHECKS PASSED
```

## 3. 运行示例 Exp

从仓库根目录先跑一个精度 case：

```bash
./scripts/run_evaluation.sh \
  --bench-name cann \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --case-id 1 \
  --device-id 0 \
  --processes-per-card 1 \
  --no-perf
```

成功标志：

```text
扫描到的 cann_bench 接口:
  1. exp(...)

level1/exp_1: ... float16[y]: ... MERE=0.000000, MARE=0.000000
```

然后启用性能采集：

```bash
./scripts/run_evaluation.sh \
  --bench-name cann \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --case-id 1 \
  --device-id 0 \
  --processes-per-card 1 \
  --warmup 3 \
  --repeat 5
```

成功标志是精度通过且耗时不是 `N/A`。Profiler 归档中的 `kernel_details.csv`
应包含 `_exp_kernel`。

## 4. 本地 Python 环境

不用 Docker 时，可以在独立虚拟环境中安装可选依赖：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-triton.txt
```

确认实际加载的是 NPU backend：

```bash
.venv/bin/python -c \
  'from triton.runtime import driver; print(driver.active.get_current_target())'
```

输出中的 `backend` 必须为 `npu`。

## 5. 常见失败

### 只 import 成功，但 kernel 编译失败

Triton-Ascend wheel 包含 NPU-IR 编译组件。CANN 和 Triton-Ascend 不匹配时，
可能在 `bishengir-compile` 或 `bisheng` 阶段失败。先确认使用的是 CANN 9.0.0
正式版和 Triton-Ascend 3.2.1，再检查镜像 smoke，不要把 import 成功当作环境通过。

### 扫描出不属于提交的旧算子

检查日志中的安装位置，应该指向当前 Python 环境中的新 wheel。CANN Bench 使用
`--force-reinstall --no-deps` 安装 submission，避免同为 `cann_bench==1.0.0` 时
被旧包遮蔽。

### 精度通过，但耗时是 `N/A`

`--no-perf` 模式本来就不会采集耗时。去掉该参数后再运行，并检查报告目录下的
`prof_data/.../kernel_details.csv` 是否存在候选 Triton kernel。
