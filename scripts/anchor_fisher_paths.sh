#!/usr/bin/env bash

anchor_compact_value() {
    local value="${1:-}"
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    value="${value//./p}"
    value="${value//+/_}"
    value="${value//-/_}"
    value="${value#_}"
    value="${value%_}"
    printf '%s' "$value"
}

anchor_detect_model_size() {
    local model_name
    model_name="$(basename "$1")"
    if [[ "$model_name" =~ ([0-9]+(\.[0-9]+)?B) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        printf '%s' "$model_name"
    fi
}

anchor_join_stage_expr() {
    local first=1
    local item
    for item in "$@"; do
        if (( first )); then
            printf '%s' "$item"
            first=0
        else
            printf '+%s' "$item"
        fi
    done
}

anchor_build_suite_name() {
    local model_size="$1"
    local cap_value="$2"
    local lam_value="$3"
    shift 3 || true
    local stage_expr
    stage_expr="$(anchor_join_stage_expr "$@")"
    printf '%s-%s-cap%s-lam%s' \
        "$model_size" \
        "$stage_expr" \
        "$(anchor_compact_value "$cap_value")" \
        "$(anchor_compact_value "$lam_value")"
}

anchor_build_suite_dir() {
    local root="$1"
    local model_size="$2"
    local cap_value="$3"
    local lam_value="$4"
    shift 4 || true
    printf '%s/%s\n' "$root" "$(anchor_build_suite_name "$model_size" "$cap_value" "$lam_value" "$@")"
}

anchor_build_suite_dir_from_model() {
    local root="$1"
    local model_path="$2"
    local cap_value="$3"
    local lam_value="$4"
    shift 4 || true
    local model_size
    model_size="$(anchor_detect_model_size "$model_path")"
    anchor_build_suite_dir "$root" "$model_size" "$cap_value" "$lam_value" "$@"
}
