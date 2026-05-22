#!/bin/bash
set -euo pipefail

# Dockerfile ENV covers PATH/LD_LIBRARY_PATH; sourcing here is for
# callers that need the full set_env.sh side effects (ASCEND_OPP_PATH etc).
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

if [[ $# -eq 0 ]]; then
    cat <<EOF
Usage: docker run --rm cann-bench:cann9.0.0-* <command> [args...]
  python3 /test_env.py            # post-build smoke
  bash                            # interactive shell
EOF
    exit 1
fi

exec "$@"
