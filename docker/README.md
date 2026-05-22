# `cann-bench:cann9.0.0-*`

CANN-BENCH 参考执行镜像。torch 2.9.0 + torch\_npu 2.9.0 + 相关科学计算栈。
默认 tag `cann-bench:cann9.0.0-latest`。

- `Dockerfile` — 镜像定义
- `entrypoint.sh` — 容器入口 (source CANN env, 转交 CMD)
- `run.sh` — host 端 launcher (smoke / shell / dev 三种模式)
- `test_env.py` — smoke 验证脚本 (版本 / torch\_npu device / npu-smi / CANN)

## 1. Build image

在 NPU HOST 上 build image:

```bash
cd /path/to/repo/docker/
docker build --network=host -t cann-bench:cann9.0.0-$(date +%y%m%d) .
docker tag cann-bench:cann9.0.0-$(date +%y%m%d) cann-bench:cann9.0.0-latest
```

也可配置代理
```bash
# export HTTP_PROXY=http://<proxy_host>:<proxy_port>
# export HTTPS_PROXY=http://<proxy_host>:<proxy_port>
docker build --network=host \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    -t cann-bench:cann9.0.0-latest .
```

也可配置 pypi 镜像源: `--build-arg PYPI_INDEX_URL`。

## 2. Smoke

验证 python / torch / torch\_npu / npu-smi / CANN 全 OK:

```bash
bash run.sh smoke
```

期望 `ALL CHECKS PASSED`。

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
| `IMAGE`     | `cann-bench:cann9.0.0-latest`     |
| `CONTAINER` | `cann-bench` (仅 dev)             |
| `WORKSPACE` | `$(pwd)/workspace` (仅 dev)       |
