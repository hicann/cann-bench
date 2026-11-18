# `cann-bench` docker/eval -- 评测镜像:`docker run` 即评测

harness 冻结在镜像里(`src/kernel_eval` + `tasks/` + `cann_bench_utils` 全部烘入
`/opt/cann-bench`),外部只挂两样东西:

| 容器路径 | 内容 | 挂法 |
|---|---|---|
| `/submission` | AI 生成的算子源码目录 | 只读即可(entrypoint 会先复制) |
| `/reports` | 评测报告 + `prof_data/` + `build/`(编译日志、wheel) | 读写 |

**镜像 tag 当前使用框架 VERSION + 目标配置**:
`cann-bench-eval:<VERSION>-<NPU_ARCH>-<ARCH>-ops<OPS_MODE>`,
例如 `cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone`。题集版本独立位于 `tasks/metadata/VERSION`；相同 tag 可能被重建覆盖，不能单独证明题集/评测器相同。应同时归档镜像 digest、构建提交、实际题集版本和报告中的评分档/公式。

## 分工:哪一层管什么

`docker/base` 就是 common —— 它到 toolkit 为止,**不装 ops**,唯一的变量是 CPU 架构。
**SoC 分叉正好从 `ops.run` 开始**,所以整条分叉都在本层,而且只有三个值:

| 轴 | 归属 | 怎么定 |
|---|---|---|
| CPU 架构(aarch64 / x86_64) | **base** | base 的 `ARCH` build-arg;建完写进 `ENV CANN_ARCH`,本层**继承**它,不再声明自己的 —— 否则两边可以不一致,悄悄往 arm 镜像装 x86 的 ops |
| SoC(910b / 910_93 / 950) | **eval** | `NPU_ARCH`(给 `cann_bench_utils` 编 kernel)+ `OPS_PKG`(ops 包名)+ `OPS_MODE` |

因此单 Dockerfile 就够:a2 与 a5 镜像的差别只有 4 个 ARG 值,**零结构差异**。

## Build

底座是 [`docker/base`](../base/) 的 `cann-toolkit-base`,先有它(**同一台机器、同一架构**,
这是原生构建,不是交叉编译):

```bash
cd docker/base && docker build --build-arg ARCH=$(uname -m) -t cann-toolkit-base:9.1.0-py3.13 .
```

然后(**build context 必须是仓库根**,`build.sh` 已经处理好):

```bash
bash docker/eval/build.sh                       # 本机架构 + ascend910b + OPS_MODE=none
NPU_ARCH=ascend950 bash docker/eval/build.sh    # 950PR —— OPS_MODE 自动取 refonly,见下
NPU_ARCH=ascend910_93 bash docker/eval/build.sh # A3
MIRROR=cn bash docker/eval/build.sh             # 受限网络: 一把切到在区镜像源
TRITON_ASCEND_VERSION=3.2.1 bash docker/eval/build.sh
```

`build.sh` 由 `NPU_ARCH` + `uname -m` 推导其余一切,正常情况下你只需要说芯片:

| `NPU_ARCH` | → `OPS_PKG` | → 默认 `OPS_MODE` |
|---|---|---|
| `ascend910b` | `910b` | `none` |
| `ascend910_93` | `910_93` | `none` |
| `ascend950` | `950` | **`refonly`**(`none` 在 950 上不可用,见下) |

| build-arg | 默认 | 说明 |
|---|---|---|
| `BASE_IMAGE` | `cann-toolkit-base:<CANN_VERSION>-py3.13` | 底座;其 `CANN_ARCH` 必须与本机架构一致,`build.sh` 会校验 |
| `NPU_ARCH` | `ascend910b` | `cann_bench_utils` 的 kernel 是 SoC 相关的,故本镜像 per-SoC |
| `OPS_PKG` | `910b` | ops `.run` 的 SoC 拼写(与编译器 flag 不同:`ascend950` ↔ `950`) |
| `OPS_MODE` | 见上表 | `none` / `refonly` / `full`,见下 |
| `CANN_OPS_URL` | 空(由 `CANN_VERSION` + `CANN_ARCH` + `OPS_PKG` 推导) | 只在需要钉死某个 `.run` 时设 |
| `TRITON_ASCEND_VERSION` | 空 | 非空则装 Triton-Ascend(体积大),语义同 `docker/dev` |
| `PYPI_MIRROR` / `TORCH_MIRROR` / `UV_PYTHON_INSTALL_MIRROR` | 空(官方源) | 同 `docker/base`;`MIRROR=cn` 是这三个的快捷方式 |

**没有 `ARCH` / `CANN_VERSION` build-arg** —— 这两样都由 base 经 `ENV` 单向下传:架构决定装哪个
架构的 ops,版本决定装哪个 release 的 ops,本层各自重新声明就可能和底座不一致。base 缺
`CANN_ARCH` / `CANN_VERSION`(即早于跨架构改动)、或 base 架构与本机不符,build 期三道断言都会
直接失败。换 CANN 版本只需重建 base,见 `docker/base/README.md`。

python 依赖由 `docker/eval/{pyproject.toml,uv.lock}` 锁定,是 `docker/base` 依赖集的**严格超集**
(`uv sync` 会把环境对齐到 lock —— 非超集会把底座已装的 torch 卸掉)。改任一边都要同步另一边。

## Run

```bash
bash docker/eval/run.sh self-test                                # 镜像自检
bash docker/eval/run.sh /path/to/ai_ops                          # 全量评测一个提交
bash docker/eval/run.sh /path/to/ai_ops --operator Exp --no-perf # 单算子, 仅精度
bash docker/eval/run.sh -- --task-dir tasks/level1 --no-perf     # 无提交(golden 自评)
bash docker/eval/run.sh shell                                    # 交互 shell, NPU 已绑入

REPORTS=/data/out ASCEND_RT_VISIBLE_DEVICES=0 bash docker/eval/run.sh /path/to/ai_ops
```

`run.sh` 只是把 NPU device flags 和两个挂载拼好;裸 `docker run` 完全等价:

```bash
docker run --rm --privileged --ipc=host \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64 \
  -v /path/to/ai_ops:/submission:ro -v "$PWD/reports:/reports" \
  cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone --operator Exp
```

### entrypoint 契约

```
docker run <image> [源码目录] [选项]
```

- `[源码目录]` 是**容器内**路径;省略则用 `/submission`。第一个参数不以 `-` 开头才算源码目录
  —— 这和 `scripts/run_evaluation.sh` 自己的约定一致(否则 `--operator Exp` 的 `Exp` 会被误判)。
- `[选项]` 原样透传给 `run_evaluation.sh`(`--operator` / `--case-id` / `--task-dir` /
  `--no-perf` / `--device-id` / `--warmup` / `--repeat` / `--eval-seed` / ...),
  完整表:`docker run <image> --help`。
- `--reports-dir` 由 entrypoint 强制注入 `/reports`。不注入的话 `run_evaluation.sh` 会写
  `${PROJECT_ROOT}/reports`,也就是镜像里那份冻结的 harness。
- **源码目录默认先复制到容器内 `/work/src` 再编译**。因为 `PackageManager` 是就地构建的:
  它 `rmtree` `<src>/{build,dist,*.egg-info}`、写 `<src>/_compile.log`,`build.sh` 的 cwd 就是
  源码目录。复制之后 `/submission` 可以 `:ro` 挂载,host 上那棵树也不会被反复跑脏。
  `--in-place` 关掉复制。收尾时 `_compile.log` 和 `dist/` 会被收到 `/reports/build/`(失败也收,
  编译错误的 bisheng/g++ 诊断只在那里面)。
- 显式给了 `--source-dir` 就原样尊重,不复制。
- 逃生口(作为第一个参数):`bash` / `sh` / `python3` / `python` / `uv` / `pip` 直接执行;
  `--self-test` 跑自检;`--help` 打用法。

### 开发时覆盖冻结的 harness

改 `kernel_eval` 时不必重 build:

```bash
docker run ... -v "$PWD:/opt/cann-bench" cann-bench-eval:... /submission --operator Exp
```

## ops 模式(反作弊形态)

提交"作弊"的方式是**调用内置算子**(`aclnn<Op>` / `torch_npu` op),它们下发到
`opp/built-in/op_impl/ai_core/tbe/kernel` 这棵多 GB 的二进制树(910B 4.2G,950 5.0G)。旁边约
151M 的 `tbe/impl` 是 AscendC **源码**,那不是作弊,是合法参考。

| `OPS_MODE` | 做什么 | 后果 |
|---|---|---|
| `none` | 不装 ops(底座本来就没有) | 内置算子起不来 —— 挡住它的是 **`libopapi` 缺席**(aclnn 的入口库在 ops 包里),而不是"kernel 树为空":toolkit 自带一棵 ~11M 的 `tbe/kernel`,但那不足以下发内置算子。镜像最小。`opp/` 仍在,`.run` 形态的自定义算子提交照常装进 `opp/vendors` |
| `refonly` | 装 ops,**同层**删掉 `tbe/kernel` 二进制 | 保留 `tbe/impl` AscendC 源码作参考;`libopapi` 在,但内置算子下发失败。比 `none` 多出 opp 的全套机制 |
| `full` | 装 ops 不删 | 可被蹭内置算子;用于重采 aclnn baseline |

自检的 `[6]` 项会直说当前镜像里内置算子能不能下发,报错码还能区分是哪种形态:
`none` → `500001 LazyInitAclops`;`refonly` → `561103 Parse dynamic kernel config fail`。

### 选哪个:910B 用 `none`,950 **必须** `refonly`

| | 910B2 / aarch64 / `none` | 950PR / x86_64 / `refonly` |
|---|---|---|
| 裸 `.npu()` H2D 拷贝 | PASS | PASS |
| `cann_bench_warmup`(10240²) | ok 4.5 ms | ok 6.4 ms |
| `cann_bench_cache_clean`(96×1024²) | ok 0.7 ms | ok 1.1 ms |
| 内置 `matmul` | BLOCKED(500001) | BLOCKED(561103) |

**`none` 在 950 上不可用**:实测那里连一次 `torch.arange(8).npu()` 都会死 —— H2D 拷贝在 950 上
走 `aclnnInplaceCopy`(`ERR01007 OPS feature not supported`),而 910B 走的是普通 `aclrtMemcpy`,
什么都不需要。`refonly` 修好这条且**仍然挡住内置算子**,所以 950 的反作弊形态是完整的,只是不能用
`none`。Dockerfile 里有 build 期断言直接拒绝 `ascend950 + none` 这个组合,不会让一个"第一个张量
就崩"的镜像出厂。

**910B 上默认 `none` 够跑全量三阶段(编译/精度/性能)** —— 实测 `direct_launch_example` 的 Sqrt
4/4 精度通过,profiler 正常产出 `prof_data/` 与 device kernel 耗时(`sqrt_kernel` 6.9us),综合得分
73.00。`LazyInitAclops` 在 0-ops 下确实会失败(自检 `[6]` 就是它),但性能采集不经过这条路 ——
升频/清 cache 由镜像里烘好的 `cann_bench_utils` 直调 kernel 提供。

## 已知取舍

- `tasks/` 烘进镜像，但 tag 不单独编码题集版本；开发期用 `-v` 覆盖工作树后更应记录实际提交与题集版本。
- `cann_bench_utils` 在 build 期就编好装好(只要 bisheng,不需要 NPU),容器起来直接开跑,
  `ensure_cann_bench_utils()` 短路返回。它含 SoC 相关 kernel,所以本镜像 per-SoC。
- **原生构建,没有交叉编译**:base 和 eval 必须在目标架构的机器上建。`uname -m` 与 base 的
  `CANN_ARCH` 不符时 build 期直接失败,而不是产出一个跑不起来的镜像。
- 在区网络下 `MIRROR=cn` 基本是必需项,不是可选项:不换源时 `uv sync` 会卡在 PyPI 上几十分钟
  (纯网络阻塞,看着像挂死);a5 上 apt 和 docker.io 同样需要换源。
