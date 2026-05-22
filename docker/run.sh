#!/usr/bin/env bash
# Debug / smoke helper for cann-bench:cann9.0.0-* image. Run on $NPU_HOST.
#   bash run.sh smoke    # post-build smoke (/test_env.py)
#   bash run.sh shell    # one-shot interactive shell with NPU bound in
#   bash run.sh dev      # detached sleep infinity for docker exec debugging

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-cann-bench:cann9.0.0-latest}"
MODE="${1:-smoke}"

NPU_FLAGS=(
    --privileged
    --ipc=host
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64:ro
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
)

case "$MODE" in
    smoke)
        docker run --rm "${NPU_FLAGS[@]}" \
            "$IMAGE" python3 /test_env.py
        ;;
    shell)
        docker run --rm -it "${NPU_FLAGS[@]}" \
            "$IMAGE" bash
        ;;
    dev)
        CONTAINER="${CONTAINER:-cann-bench}"
        WORKSPACE="${WORKSPACE:-$(pwd)/workspace}"
        mkdir -p "$WORKSPACE"
        docker run -d --name "$CONTAINER" \
            "${NPU_FLAGS[@]}" \
            -v "$WORKSPACE:/workspace" \
            "$IMAGE" sleep infinity
        echo "==> Started container '$CONTAINER' (image: $IMAGE)"
        echo "==> Workspace: $WORKSPACE -> /workspace (host bind)"
        echo "==> Attach:    docker exec -it $CONTAINER bash"
        echo "==> Cleanup:   docker rm -f $CONTAINER"
        ;;
    *)
        echo "Usage: $0 [smoke|shell|dev]"
        echo "Env:   IMAGE=<tag>  CONTAINER=<name for dev>"
        exit 1
        ;;
esac
