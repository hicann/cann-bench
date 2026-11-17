# cann-bench — CI execution image.
#
# 多平台支持：通过 --build-arg 切换 CANN 版本和硬件型号。
#
# 示例：
#   docker build --network=host -t cann-bench:cann9.1.0-910b-latest \
#       --build-arg CANN_VERSION=9.1.0 --build-arg DEVICE=910b .
#
#   docker build --network=host -t cann-bench:cann9.1.0-950-latest \
#       --build-arg CANN_VERSION=9.1.0 --build-arg DEVICE=950 .
#
# 完整镜像 tag 格式：cann:<CANN_VERSION>-<DEVICE>-ubuntu22.04-py3.12-<VARIANT>
# 2026-08-17 在 AscendHub 上核实过的取值：
#   CANN_VERSION  9.0.1 / 9.1.0（9.0.0 的 tag 已下架 —— 本文件此前的默认值就指向它，
#                 也就是说这个镜像已经建不出来了；9.2.0 尚未发布）
#   DEVICE        910 / 910b / 950 / a3 / 310p（910c 不存在，旧注释里写错了）
#   VARIANT       devel（含编译工具链，评测要编提交，选它）/ runtime 系列
# 9.1.0-910b 与 9.1.0-950 的 -devel 均为 arm64 + amd64 双架构。
#
# 版本配套:CANN 9.1.0 这一行是 torch 2.10.0 <-> torch_npu 2.10.0.post4(另有 2.11 / 2.12
# 两条线),见 https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md
# torch/torch_npu 由下面显式安装并钉版本,不依赖基础镜像自带的串号;换 CANN_VERSION 时
# 必须按配套表同步改 TORCH_VERSION / TORCH_NPU_VERSION,否则 ABI 不配套。
# 两者落到 ENV,由 test_env.py 回读实际装到的版本校验,文本改错会在 smoke 阶段红。

ARG CANN_VERSION=9.1.0
ARG DEVICE=910b
# devel 才带编译工具链；AscendHub 从某个版本起把它拆成了单独的 tag 后缀。
ARG VARIANT=devel
ARG TRITON_ASCEND_VERSION=
ARG TRITON_ASCEND_INDEX_URL=https://triton-ascend.osinfra.cn/pypi/simple
ARG NODE_MAJOR=22
ARG CODEX_VERSION=0.150.1
ARG CLAUDE_CODE_VERSION=2.1.220

FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VERSION}-${DEVICE}-ubuntu22.04-py3.12-${VARIANT}

ARG TRITON_ASCEND_VERSION
ARG TRITON_ASCEND_INDEX_URL
ARG NODE_MAJOR
ARG CODEX_VERSION
ARG CLAUDE_CODE_VERSION

# 防作弊：永久删除 CANN 内置 kernel 树，阻断 aclnn 内置算子的 NPU 内核访问。
# kernel/ascend* 删预编译的 .o + config/ascend* 删 JSON 配置；
# impl/ops_nn/ascendc 删 AscendC 源码防止 JIT 编译回退。
# overlay whiteout 确保删除在镜像中持久化，不受运行时 mount 操作影响。
RUN rm -rf /usr/local/Ascend/cann-*/opp/built-in/op_impl/ai_core/tbe/kernel/ascend* \
           /usr/local/Ascend/cann-*/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend* \
           /usr/local/Ascend/cann-*/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc \
    && echo "Anti-cheat: built-in kernel tree removed"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential cmake \
    clang lld ninja-build \
    tmux clangd gh rsync && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-interactive optimization agents used by scripts/batch_optimize.py.
# Credentials and API endpoints remain runtime environment variables.
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" -o /tmp/nodesource.sh \
    && bash /tmp/nodesource.sh \
    && rm -f /tmp/nodesource.sh \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g \
       "@openai/codex@${CODEX_VERSION}" \
       "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv  /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx

ARG PYPI_INDEX_URL=
ENV UV_INDEX_URL=${PYPI_INDEX_URL}

# 与 CANN_VERSION 配套的 torch 栈,见文件头的配套表链接。
ARG TORCH_VERSION=2.10.0
ARG TORCH_NPU_VERSION=2.10.0.post4
ENV CANN_BENCH_TORCH_VERSION=${TORCH_VERSION}
ENV CANN_BENCH_TORCH_NPU_VERSION=${TORCH_NPU_VERSION}

# torch_npu install needs these at build time to locate driver libs.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:${ASCEND_HOME_PATH}/aarch64-linux/lib64

# 使用华为云源（清华源持续 403）；PYPI_INDEX_URL build-arg 可覆盖此默认源。
RUN uv pip install --system --index-url ${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/} \
    filelock typing-extensions sympy networkx jinja2 fsspec mpmath markupsafe \
    numpy pyyaml decorator scipy attrs cloudpickle \
    ml-dtypes psutil tornado pandas expecttest protobuf \
    pytest pytest-timeout 'ruamel.yaml>=0.18' build setuptools wheel pybind11
RUN uv pip install --system --no-deps --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION}
RUN uv pip install --system --no-deps --index-url ${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/} torch_npu==${TORCH_NPU_VERSION}

# Triton-Ascend is optional because its compiler runtime is large. The wheel
# owns the top-level `triton` package, so install it and its pinned dependencies
# together rather than mixing it with a separately installed Triton package.
RUN if [ -n "${TRITON_ASCEND_VERSION}" ]; then \
      uv pip install --system \
        --index-url "${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/}" \
        --extra-index-url "${TRITON_ASCEND_INDEX_URL}" \
        "triton-ascend==${TRITON_ASCEND_VERSION}" && \
      uv pip check; \
    fi

ENV TRITON_ASCEND_VERSION=${TRITON_ASCEND_VERSION}
ENV TRITON_CACHE_DIR=/tmp/cann-bench-triton-cache
RUN mkdir -p ${TRITON_CACHE_DIR} && chmod 1777 ${TRITON_CACHE_DIR}

# Pre-populate uid 1001 in /etc/passwd so getpass/getuser resolves in the
# sandbox runtime (which executes processes as uid 1001); not a privilege drop.
RUN echo "benchuser:x:1001:1001:bench user:/tmp:/bin/bash" >> /etc/passwd

COPY test_env.py    /test_env.py
COPY entrypoint.sh  /entrypoint.sh
RUN chmod +x /entrypoint.sh /test_env.py

# Drop the mirror ENV so runtime stays clean (build-time only).
ENV UV_INDEX_URL=

ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /workspace
