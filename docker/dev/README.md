# `cann-bench` docker/dev — 多平台交互/CI 调试镜像

> 跑评测请用 [`docker/eval`](../eval/)(`docker run` 即评测,harness 冻结在镜像里)。
> 本镜像基于 AscendHub 完整 CANN(含 ops/nnal),定位是**需要内置算子的交互调试**、
> Triton-Ascend smoke,以及历史 CI。它不自带 harness,仍需挂一棵 cann-bench 工作树进来。

CANN-BENCH 参考执行镜像。torch 2.10.0 + torch\_npu 2.10.0.post4 + 相关科学计算栈。
通过 `--build-arg` 参数化 CANN 版本和硬件型号，同一 Dockerfile 适配不同设备。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CANN_VERSION` | `9.1.0` | CANN toolkit 版本 |
| `DEVICE` | `910b` | 硬件型号（910 / 910b / 950 / a3 / 310p，对应 AscendHub 镜像 tag 的硬件段） |
| `TRITON_ASCEND_VERSION` | 空 | 可选 Triton-Ascend 版本；当前验证版本为 `3.2.1` |
| `VARIANT` | `devel` | AscendHub tag 的末段；`devel` 才带编译工具链，评测要编提交 |
| `TORCH_VERSION` | `2.10.0` | 与 `CANN_VERSION` 配套,见下 |
| `TORCH_NPU_VERSION` | `2.10.0.post4` | 同上 |
| `NODE_MAJOR` | `22` | Codex 与 Claude Code 使用的 Node.js 主版本 |
| `CODEX_VERSION` | `0.150.1` | 镜像内 Codex CLI 版本 |
| `CLAUDE_CODE_VERSION` | `2.1.220` | 镜像内 Claude Code 版本 |

torch 栈**不**取基础镜像自带的串号,由 Dockerfile 显式安装并钉死。改 `CANN_VERSION` 时这两个
要一起按[官方配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)改
(9.1.0 这一行是 torch `2.10.0` <-> torch_npu `2.10.0.post4`)。两个值会落到镜像的
`CANN_BENCH_TORCH_VERSION` / `CANN_BENCH_TORCH_NPU_VERSION`,`test_env.py` 的 [1] 会回读
实际装到的版本比对 —— 钉错或被上游覆盖,`run.sh smoke` 就红。

默认 tag `cann-bench:cann9.1.0-910b-latest`,基础镜像是
`ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`(arm64 + amd64 双架构)。

> 上游 tag 会下架:`9.0.0-910b-ubuntu22.04-py3.12`(本文件此前的默认)在 2026-08-17
> 核查时已不存在,即那个默认值已经建不出镜像。换版本前先 `docker manifest inspect`
> 确认 tag 还在。

- `Dockerfile` — 镜像定义（ARG 参数化）
- `entrypoint.sh` — 容器入口 (source CANN env, 转交 CMD)
- `run.sh` — host 端 launcher (smoke / shell / dev 三种模式)
- `test_env.py` — smoke 验证脚本 (版本 / torch\_npu device / npu-smi / CANN)

## 1. Build image

在 NPU HOST 上 build image。**build context 必须是本目录 `docker/dev/`** —— Dockerfile 里
`COPY test_env.py` / `COPY entrypoint.sh` 取的是同目录文件，在上层 `docker/` 下构建会找不到。
下面每条 `docker build ... .` 都以先 `cd` 到本目录为前提：

```bash
cd /path/to/repo/docker/dev/
```

### 910B（默认）

```bash
docker build --network=host -t cann-bench:cann9.1.0-910b-latest .
```

### 950PR

```bash
docker build --network=host \
    --build-arg CANN_VERSION=9.1.0 --build-arg DEVICE=950 \
    -t cann-bench:cann9.1.0-950-latest .
```

### Triton-Ascend（910B / 950PR）

```bash
docker build --network=host \
    --build-arg CANN_VERSION=9.1.0 \
    --build-arg DEVICE=950 \
    --build-arg TRITON_ASCEND_VERSION=3.2.1 \
    -t cann-bench:cann9.1.0-950-triton3.2.1 .
```

设置该参数后，镜像 smoke 会实际 JIT 编译并运行 Triton vector add，而不只是检查 import。

也可配置代理:
```bash
docker build --network=host \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
    -t cann-bench:cann9.1.0-910b-latest .
```

也可配置 pypi 镜像源: `--build-arg PYPI_INDEX_URL`。

## 2. Smoke

验证 python / torch / torch\_npu / npu-smi / CANN 全 OK:

```bash
bash run.sh smoke
# 或指定 950PR 镜像:
IMAGE=cann-bench:cann9.1.0-950-latest bash run.sh smoke
# Triton-Ascend image:
IMAGE=cann-bench:cann9.1.0-950-triton3.2.1 bash run.sh smoke
```

期望 `ALL CHECKS PASSED`。

镜像同时包含 `codex` 与 `claude`，供仓库根目录的 `scripts/batch_optimize.py --docker`
按算子启动临时容器。凭据、API 地址和模型通过运行时环境变量传入：

```bash
docker run --rm cann-bench:cann9.1.0-910b-latest codex --version
docker run --rm cann-bench:cann9.1.0-910b-latest claude --version
```

## 3. 启动临时容器

退出即删:

```bash
bash run.sh shell
```

## 4. 启动常驻容器

后台 `sleep infinity`, 多次 `docker exec` 进入; `docker/workspace/` 绑到容器内 `/workspace`:

```bash
bash run.sh dev                          # 起 'cann-bench'
docker exec -it cann-bench bash
docker rm -f cann-bench                  # 收尾
```

Override: `CONTAINER=<name> WORKSPACE=<host-path> bash run.sh dev`。

## Env

| 变量        | 默认                              |
|-------------|-----------------------------------|
| `IMAGE`     | `cann-bench:cann9.1.0-910b-latest` |
| `CONTAINER` | `cann-bench` (仅 dev)             |
| `WORKSPACE` | `$(pwd)/workspace` (仅 dev)       |
