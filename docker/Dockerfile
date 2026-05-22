# cann-bench:cann9.0.0-* — CI execution image.
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential cmake \
    clang lld ninja-build \
    tmux clangd gh rsync && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -LSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv  /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx

ARG PYPI_INDEX_URL=
ENV UV_INDEX_URL=${PYPI_INDEX_URL}

# torch_npu install needs these at build time to locate driver libs.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:${ASCEND_HOME_PATH}/aarch64-linux/lib64

RUN uv pip install --system torch==2.9.0
RUN uv pip install --system numpy pyyaml decorator scipy attrs cloudpickle \
    ml-dtypes psutil tornado pandas expecttest protobuf
RUN uv pip install --system torch_npu==2.9.0
RUN uv pip install --system pytest pytest-timeout 'ruamel.yaml>=0.18'

COPY test_env.py    /test_env.py
COPY entrypoint.sh  /entrypoint.sh
RUN chmod +x /entrypoint.sh /test_env.py

# Drop the mirror ENV so runtime stays clean (build-time only).
ENV UV_INDEX_URL=

ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /workspace
