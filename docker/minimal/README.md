# `cann-bench` docker/minimal -- 最小 toolkit-only 直调基础镜像

与同目录 `docker/`(dev/CI 参考镜像,基于 AscendHub 完整 CANN,含 ops/nnal)**互补**。本镜像只装
CANN **toolkit**(bisheng + AscendC 头 + acl runtime)+ torch(cpu)/torch_npu 胶水,**不装 ops**
(`libopapi` + kernel blob)/nnal,面向 **直调(direct-launch)风格提交** -- 提交自带 bisheng 编译的
AscendC/CCE kernel,经 `KNAME<<<grid, nullptr, stream>>>` 直接下发,不走 aclnn,故 ops 是死重。

| | `docker/`(dev / CI) | `docker/minimal` |
|------|------|------|
| base | AscendHub 完整 CANN(`cann:<ver>-<device>-...`,per-device) | `debian:12-slim`,从 `.run` 自装 toolkit |
| ops / nnal | 有 | **无(0 ops)** |
| chip | per-`DEVICE` tag | **chip-agnostic**(chip 只进 mounted driver + bisheng `--soc`) |
| py / env | ubuntu22.04 + py3.12 | uv 管理的 py3.13 standalone(`uv.lock` 锁定) |
| 适用 | 全量评测(含 aclnn baseline + perf 开箱) | 直调提交:精度独立可跑;perf 见下 |

## 为什么(直调 / 反作弊)

直调 kernel 自带实现,不蹭内置优化算子 -- 这正是反作弊(anti-cheat)想要的形态。本镜像根本不含
内置 kernel 树,提交无从蹭起;相比在完整镜像上运行时搬走 4.2G kernel 树的做法(见
`scripts/anti_cheat/`),这里天然如此,且镜像更小、chip-agnostic、自包含(不依赖体积大且受限的
AscendHub per-device 镜像)。

## Build

```bash
cd docker/minimal/
docker build -t cann-toolkit-base:9.0.1-py3.13 .
```

镜像 **aarch64-only**(Ascend host = Kunpeng/ARM;x86_64 若需另开专门测过的变体)。python 依赖由
`pyproject.toml` + `uv.lock` 锁定(hash 校验),`uv sync --frozen` 装入 `/opt/venv`。

### 镜像源(每个都默认走官方/全球源;受限网络用 `--build-arg` 换在区镜像)

| build-arg | 默认(官方) | 换镜像示例(CN) |
|------|------|------|
| `BASE_OS` | `debian:12-slim`(docker.io) | `docker.m.daocloud.io/library/debian:12-slim` |
| `UV_IMAGE` | `ghcr.io/astral-sh/uv:0.11.29` | `ghcr.m.daocloud.io/astral-sh/uv:0.11.29` |
| `APT_MIRROR` | (空 = `deb.debian.org`) | `mirrors.huaweicloud.com` |
| `UV_PYTHON_INSTALL_MIRROR` | (空 = github releases) | `https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone` |
| `PYPI_MIRROR` | (空 = `files.pythonhosted.org`) | `https://mirrors.huaweicloud.com/repository/pypi` |
| `TORCH_MIRROR` | (空 = `download.pytorch.org`) | `https://mirror.nju.edu.cn/pytorch/whl/cpu` |
| `CANN_VERSION` | `9.0.1` | toolkit `.run` 版本(从 OBS 拉取) |

`PYPI_MIRROR`/`TORCH_MIRROR` 就地改写 `uv.lock` 里的 canonical wheel URL(**同 hash**,`--frozen` 仍校验),
换源不破坏可复现性。CN 全量示例:

```bash
docker build -t cann-toolkit-base:9.0.1-py3.13 \
  --build-arg BASE_OS=docker.m.daocloud.io/library/debian:12-slim \
  --build-arg APT_MIRROR=mirrors.huaweicloud.com \
  --build-arg UV_IMAGE=ghcr.m.daocloud.io/astral-sh/uv:0.11.29 \
  --build-arg UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  --build-arg PYPI_MIRROR=https://mirrors.huaweicloud.com/repository/pypi \
  --build-arg TORCH_MIRROR=https://mirror.nju.edu.cn/pytorch/whl/cpu .
```

## Run(NPU host)

`run.sh` 负责 device 挂载 + 把 host driver runtime libs 放上 `LD_LIBRARY_PATH`(dev 镜像从
AscendHub base 继承,本镜像需显式设):

```bash
bash run.sh smoke     # torch_npu device_count / name
bash run.sh shell     # 交互 shell,NPU 已绑入
bash run.sh dev       # 后台 sleep infinity,供 docker exec 调试
```

## 评测直调提交

挂载本仓库(`src/` + `tasks/`)+ 提交源码进容器,跑 `scripts/run_evaluation.sh`:

- **精度**(`--no-perf`):**0 ops 即可,独立可跑**。golden 走 CPU,提交 kernel 走 NPU,CPU 对比。
- **性能**:另需两项(均 lean-side,非 ops)——
  - 框架 warmup 算子 `cann_bench_utils`:无 ops 镜像上内置 `torch.matmul` / `torch.max`
    (升频 / 清 cache)因 `LazyInitAclops` 不可用,需自定义**直调** warmup 顶替(参见 PR #207)。
  - `libsqlite3`:**已烘入本镜像** -- msprof 导出 `kernel_details.csv`(perf stage 经
    `torch_npu.profiler` 触发)`import sqlite3` 需要它,`debian-slim` 默认不带。

## 不适用

- **aclnn baseline** 或任何需内置算子的评测:内置 torch_npu 算子在本镜像上 `LazyInitAclops` 失败 ->
  走 `docker/`(dev)镜像(需 ops)。本镜像专供直调 kernel。
