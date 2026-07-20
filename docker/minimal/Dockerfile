# cann-toolkit-base: minimal CANN toolkit + torch_npu direct-launch base. Single self-contained Dockerfile.
#
# WHY toolkit-only (no ops/nnal): direct-launch (bisheng-compiled AscendC/CCE kernels launched
# via `<<<>>>`) never calls aclnn, so ops (libopapi + kernel binaries) and nnal (ATB/SiP) are dead
# weight. Verified on 910b: torch_npu imports + npu memcpy works with libopapi absent.
# Pairing (Ascend/pytorch compat table): torch_npu 2.10.0.post2 <-> torch 2.10 <-> CANN 9.0.x, py3.13 cp313.

# ----------------------------------------------------------------------------------------------------------
# Mirror / version knobs -- every one DEFAULTS to the official/global source; pass --build-arg to swap to an
# in-region mirror on network-restricted hosts. Full CN-mirror recipe:
#   --build-arg BASE_OS=docker.m.daocloud.io/library/debian:12-slim  (base image; docker.io blocked)
#   --build-arg APT_MIRROR=mirrors.huaweicloud.com                   (apt; deb.debian.org throttled)
#   --build-arg UV_IMAGE=ghcr.m.daocloud.io/astral-sh/uv:0.11.29      (uv binary; ghcr slow)
#   --build-arg UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone
#                                                                    (uv's cpython; github flaky. USTC keeps
#                                                                     LatestRelease only -> no pinned tag)
#   --build-arg PYPI_MIRROR=https://mirrors.huaweicloud.com/repository/pypi   (general wheels)
#   --build-arg TORCH_MIRROR=https://mirror.nju.edu.cn/pytorch/whl/cpu    (torch +cpu wheel)
# PYPI_MIRROR/TORCH_MIRROR rewrite the hash-pinned lock's canonical URLs IN PLACE (same hashes, so `uv sync
# --frozen` still verifies) -> mirroring keeps reproducibility. debian:12-slim proven with CANN by triton-
# ascend. UV_IMAGE needs a global ARG + `FROM ... AS uvbin` (COPY --from can't expand a build-arg image ref).
# ----------------------------------------------------------------------------------------------------------
ARG BASE_OS=debian:12-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29
ARG APT_MIRROR=
ARG UV_PYTHON_INSTALL_MIRROR=
ARG PYPI_MIRROR=
ARG TORCH_MIRROR=
ARG CANN_VERSION=9.0.1

FROM ${UV_IMAGE} AS uvbin
FROM ${BASE_OS}

# Docker resets ARG scope at each FROM -- re-declare the ones the build stage below uses.
ARG APT_MIRROR
ARG UV_PYTHON_INSTALL_MIRROR
ARG PYPI_MIRROR
ARG TORCH_MIRROR
ARG CANN_VERSION

# WHY aarch64 is hardcoded (not an ARG): the image is aarch64-only by construction -- torch/torch_npu
# wheels, --platform, and every tested path are aarch64 (Ascend hosts are Kunpeng/ARM). A parametric
# ARCH knob we never build on x86_64 is a false promise (silent FHS-path breakage in the JSON ENTRYPOINT
# that can't expand build-args); an x86_64 image, if ever needed, is a separate deliberately-tested variant.

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# plugin compiles with g++, kernel with bisheng. libsqlite3-0: msprof's profiling analysis
# (the kernel_details.csv export the kernel_eval perf stage triggers via torch_npu.profiler) does
# `import sqlite3`, whose _sqlite3 ext dynlinks libsqlite3.so.0 -- debian-slim omits it, so without
# it perf yields no kernel timing (accuracy is unaffected). APT_MIRROR (top): sed-swap deb.debian.org in
# the deb822 sources (bookworm /etc/apt/sources.list.d/debian.sources) to an in-region mirror.
RUN if [ -n "${APT_MIRROR}" ]; then sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    build-essential g++ cmake ninja-build patchelf pkg-config xz-utils \
    libsqlite3-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv-managed standalone py3.13 -- its libpython3.13.so satisfies CANN; system python isn't 3.13.
COPY --from=uvbin /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Lockfile-driven env: `uv sync --frozen` installs the exact hash-pinned graph (pyproject + uv.lock)
# into /opt/venv. WHY pip/setuptools/wheel are in the lock: the Toolkit ge-compiler sub-installer runs
# `pip3 install`, and the submitted extension's setup.py / `python -m build` needs setuptools.
# UV_PYTHON_INSTALL_MIRROR redirects uv's cpython download off github; PYPI_MIRROR/TORCH_MIRROR rewrite the
# lock's canonical wheel URLs to a mirror IN PLACE (same hashes -> --frozen still verifies). All "" = official.
COPY pyproject.toml uv.lock /opt/base-env/
RUN L=/opt/base-env/uv.lock; \
    if [ -n "${PYPI_MIRROR}" ];  then sed -i "s|https://files.pythonhosted.org/packages/|${PYPI_MIRROR%/}/packages/|g" "$L"; fi; \
    if [ -n "${TORCH_MIRROR}" ]; then sed -i "s|https://download-r2.pytorch.org/whl/cpu/|${TORCH_MIRROR%/}/|g" "$L"; fi; \
    UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" uv sync --frozen --project /opt/base-env
ENV PATH=/opt/venv/bin:${PATH}

# Toolkit ONLY -- the ops/nnal .run the upstream cann-container-image also fetches are omitted.
RUN cd /tmp \
    && URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run" \
    && wget --quiet --header="Referer: https://www.hiascend.com/" -O toolkit.run "${URL}" \
    && chmod +x toolkit.run && ./toolkit.run --quiet --install --install-for-all && rm -f toolkit.run

# WHY stub: torch_npu _C.so has an unconditional DT_NEEDED on libhccl.so but imports 0 symbols from it
# -- an empty SONAME-only .so satisfies the loader without the real HCCL lib (which ships in ops).
RUN printf 'int __hccl_stub;\n' | gcc -shared -fPIC -x c -Wl,-soname,libhccl.so \
    -o /usr/local/Ascend/cann-${CANN_VERSION}/aarch64-linux/lib64/libhccl.so -

# Env = set_env.sh + aarch64-linux/lib64 (WHY the extra path: set_env omits it, but libhccl + its
# siblings live there). ENTRYPOINT applies it for `docker run img <cmd>`; bash.bashrc for `docker exec`.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV PATH=/opt/venv/bin:${ASCEND_HOME_PATH}/aarch64-linux/ccec_compiler/bin:${PATH}
RUN printf '%s\n' \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true' \
    'export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:${LD_LIBRARY_PATH}' \
    >> /etc/bash.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:${LD_LIBRARY_PATH}; exec \"$@\"", "bash"]
WORKDIR /workspace
CMD ["bash"]
