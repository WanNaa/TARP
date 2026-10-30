#!/usr/bin/env bash

if [[ "${TARP_RUNTIME_LAYOUT_SOURCED:-0}" == "1" ]]; then
    return 0
fi
TARP_RUNTIME_LAYOUT_SOURCED=1

# Keep the root short because Ray's AF_UNIX socket paths are length-limited.
export TARP_RUNTIME_ROOT="${TARP_RUNTIME_ROOT:-/path/to/runtime}"
export TARP_RUNTIME_TMP_ROOT="${TARP_RUNTIME_TMP_ROOT:-${TARP_RUNTIME_ROOT}/tmp}"
export TARP_RUNTIME_RAY_ROOT="${TARP_RUNTIME_RAY_ROOT:-${TARP_RUNTIME_ROOT}/ray}"
export TARP_RUNTIME_TORCHINDUCTOR_ROOT="${TARP_RUNTIME_TORCHINDUCTOR_ROOT:-${TARP_RUNTIME_ROOT}/torchinductor}"
export TARP_RUNTIME_EVAL_ROOT="${TARP_RUNTIME_EVAL_ROOT:-${TARP_RUNTIME_ROOT}/eval}"
export TARP_RUNTIME_LOG_ROOT="${TARP_RUNTIME_LOG_ROOT:-${TARP_RUNTIME_ROOT}/logs}"
export TARP_RUNTIME_HYDRA_ROOT="${TARP_RUNTIME_HYDRA_ROOT:-${TARP_RUNTIME_ROOT}/hydra}"

tarp_runtime_tmp_dir() {
    printf '%s/%s' "$TARP_RUNTIME_TMP_ROOT" "$1"
}

tarp_runtime_ray_dir() {
    printf '%s/%s' "$TARP_RUNTIME_RAY_ROOT" "$1"
}

tarp_runtime_torchinductor_dir() {
    printf '%s/%s' "$TARP_RUNTIME_TORCHINDUCTOR_ROOT" "$1"
}

tarp_runtime_eval_dir() {
    printf '%s/%s' "$TARP_RUNTIME_EVAL_ROOT" "$1"
}

tarp_runtime_log_dir() {
    printf '%s/%s' "$TARP_RUNTIME_LOG_ROOT" "$1"
}

tarp_runtime_hydra_dir() {
    printf '%s/%s' "$TARP_RUNTIME_HYDRA_ROOT" "$1"
}

tarp_prepare_runtime_dirs() {
    mkdir -p \
        "$TARP_RUNTIME_TMP_ROOT" \
        "$TARP_RUNTIME_RAY_ROOT" \
        "$TARP_RUNTIME_TORCHINDUCTOR_ROOT" \
        "$TARP_RUNTIME_EVAL_ROOT" \
        "$TARP_RUNTIME_LOG_ROOT" \
        "$TARP_RUNTIME_HYDRA_ROOT" \
        "$TMPDIR" \
        "$RAY_TEMP_ROOT" \
        "$RAY_TMPDIR" \
        "$TORCHINDUCTOR_CACHE_DIR" \
        "$TARP_RUNTIME_LOG_DIR"
}

export TMPDIR="${TMPDIR:-$TARP_RUNTIME_TMP_ROOT}"
export RAY_TEMP_ROOT="${RAY_TEMP_ROOT:-$TARP_RUNTIME_RAY_ROOT}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TEMP_ROOT}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TARP_RUNTIME_TORCHINDUCTOR_ROOT}}"
export TARP_RUNTIME_LOG_DIR="${TARP_RUNTIME_LOG_DIR:-${TARP_RUNTIME_LOG_ROOT}}"
tarp_prepare_runtime_dirs
