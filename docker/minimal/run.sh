#!/usr/bin/env bash
# Host-side launcher for the minimal (toolkit-only) cann-bench image. Run on an NPU host.
#   bash run.sh smoke    # torch_npu device smoke
#   bash run.sh shell    # one-shot interactive shell with NPU bound in
#   bash run.sh dev      # detached `sleep infinity` for `docker exec` debugging
#
# Unlike the dev image (which inherits driver libs from its AscendHub base), this image is
# toolkit-only, so the host driver runtime libs (libascend_hal etc.) must be put on
# LD_LIBRARY_PATH here; the toolkit lib64 is already added by the image ENTRYPOINT.
#   IMAGE=<tag> ASCEND_RT_VISIBLE_DEVICES=0 bash run.sh shell
#
# Build-time mirror knobs (restricted networks) are a separate concern -- see README.md's build-arg
# table (BASE_OS / APT_MIRROR / UV_IMAGE / UV_PYTHON_INSTALL_MIRROR / PYPI_MIRROR / TORCH_MIRROR).

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-cann-toolkit-base:9.0.1-py3.13}"
MODE="${1:-smoke}"

DRV=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64

NPU_FLAGS=(
    --privileged
    --ipc=host
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -e LD_LIBRARY_PATH="${DRV}"
    -e ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
)

case "$MODE" in
    smoke)
        docker run --rm "${NPU_FLAGS[@]}" "$IMAGE" \
            python3 -c 'import torch, torch_npu; print("device_count:", torch.npu.device_count(), "name:", torch.npu.get_device_name(0))'
        ;;
    shell)
        docker run --rm -it "${NPU_FLAGS[@]}" "$IMAGE" bash
        ;;
    dev)
        CONTAINER="${CONTAINER:-cann-toolkit-base}"
        WORKSPACE="${WORKSPACE:-$(pwd)/workspace}"
        mkdir -p "$WORKSPACE"
        docker run -d --name "$CONTAINER" "${NPU_FLAGS[@]}" \
            -v "$WORKSPACE:/workspace" "$IMAGE" sleep infinity
        echo "==> Started '$CONTAINER' (image: $IMAGE); attach: docker exec -it $CONTAINER bash"
        ;;
    *)
        echo "Usage: $0 [smoke|shell|dev]   Env: IMAGE=<tag> ASCEND_RT_VISIBLE_DEVICES=<id>"
        exit 1
        ;;
esac
