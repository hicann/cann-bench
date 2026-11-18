# `cann-bench` 镜像

三个镜像,一条继承链。绝大多数人只需要 **`eval/`**。

```
base/  (cann-toolkit-base)          环境底座: CANN toolkit + torch/torch_npu, 0 ops, 自己不评测
  └── eval/  (cann-bench-eval)      评测器: 烘入 kernel_eval + tasks/, docker run 即跑评测
dev/   (cann-bench:cann9.1.0-*)     AscendHub 全量 CANN 的交互/CI 调试镜像 (独立血统)
```

**base 就是 common。** 它到 toolkit 为止不装 ops,唯一的变量是 CPU 架构(`ARCH`,写进
`ENV CANN_ARCH` 供下游继承)。**芯片(SoC)的分叉正好从 `ops.run` 开始**,所以整条分叉都在
`eval/` 里,而且只是三个 ARG 值(`NPU_ARCH` / `OPS_PKG` / `OPS_MODE`)—— a2 与 a5 的镜像**零
结构差异**,不需要两份 Dockerfile。

| 目标 | 建在哪 | `ARCH` | `NPU_ARCH` | `OPS_MODE` |
|---|---|---|---|---|
| A2 (910B2) | aarch64 host | `aarch64` | `ascend910b` | `none` |
| A5 (950PR) | x86_64 host | `x86_64` | `ascend950` | **`refonly`**(`none` 在 950 上跑不了,见 eval README) |

原生构建,没有交叉编译 —— base 和 eval 必须在目标架构的机器上建,不符会在 build 期直接失败。

| | 干什么 | 什么时候用 |
|---|---|---|
| [`eval/`](eval/) | **`docker run <image> [源码目录] [选项]` 直接产出评测报告** | 评一个提交;CI 打分;任何要求"这个分数出自哪个 benchmark 版本"可回答的场景 |
| [`base/`](base/) | toolkit-only 底座,给 `eval` 继承;也可单独当直调开发环境 | 自己搭环境、调 kernel、给 `eval` 重建底座 |
| [`dev/`](dev/) | AscendHub 完整 CANN(含 ops/nnal)+ tmux/gh/clangd 等 | 需要内置算子的交互调试、Triton-Ascend smoke、老 CI |

## 为什么有 `eval/`

在此之前,跑一次评测需要一棵"活"的 cann-bench 工作树加一套手工装好的 CANN/torch_npu 环境:
`scripts/run_evaluation.sh` 从自身位置推 `PROJECT_ROOT`,报告写回仓库 `reports/`。于是评测依赖
被机器之间搬来搬去的环境和二进制,而且"这个分数出自哪个 benchmark 版本"没有答案。

`eval/` 把 harness(`src/kernel_eval` + `tasks/` + `cann_bench_utils`)冻结进镜像,镜像 tag 当前包含框架 VERSION、SoC、CPU 架构与 ops 模式；题集另有
`metadata/VERSION`，复现还需记录镜像 digest / 构建提交及报告评分口径;外部只挂两样东西:提交源码(`/submission`,只读)和报告目录(`/reports`)。

```bash
bash docker/eval/build.sh
bash docker/eval/run.sh /path/to/ai_ops --operator Exp
```

细节见 [`eval/README.md`](eval/README.md)。
