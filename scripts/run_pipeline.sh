#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/anchor_fisher_paths.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime_layout.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh [PRESET] [extra args...]

Presets:
  qwen_3b_baseline
  qwen_3b_tarp
  llama_3b_baseline
  llama_3b_tarp
  qwen_7b_baseline
  qwen_7b_tarp

Examples:
  bash scripts/run_pipeline.sh qwen_3b_baseline
  bash scripts/run_pipeline.sh llama_3b_tarp
  PIPELINE_PRESET=qwen_7b_tarp bash scripts/run_pipeline.sh
EOF
}

normalize_preset() {
    local raw="${1,,}"
    case "$raw" in
        qwen_3b_baseline|3b_baseline|pipeline_3b_baseline|run_qwen_3b_baseline.sh|run_pipeline_3b_baseline.sh)
            printf '%s' "qwen_3b_baseline"
            ;;
        qwen_3b_tarp|3b_tarp|pipeline_3b_tarp|run_qwen_3b_tarp.sh|run_pipeline_3b_tarp.sh)
            printf '%s' "qwen_3b_tarp"
            ;;
        llama_3b_baseline|llama3_3b_baseline|l32_3b_baseline|run_llama_3b_baseline.sh|run_pipeline_llama_3b_baseline.sh)
            printf '%s' "llama_3b_baseline"
            ;;
        llama_3b_tarp|llama3_3b_tarp|l32_3b_tarp|run_llama_3b_tarp.sh|run_pipeline_llama_3b_tarp.sh)
            printf '%s' "llama_3b_tarp"
            ;;
        qwen_7b_baseline|7b_baseline|pipeline_7b_baseline|run_qwen_7b_baseline.sh|run_pipeline_7b_baseline.sh)
            printf '%s' "qwen_7b_baseline"
            ;;
        qwen_7b_tarp|7b_tarp|pipeline_7b_tarp|run_qwen_7b_tarp.sh|run_pipeline_7b_tarp.sh)
            printf '%s' "qwen_7b_tarp"
            ;;
        *)
            return 1
            ;;
    esac
}

set_preset_defaults() {
    DEFAULT_MODEL_ENV="qwen_ins"
    DEFAULT_MODEL_TAG_PREFIX=""
    DEFAULT_PIPELINE_NAME_TAG=""
    case "$PIPELINE_PRESET" in
        qwen_3b_baseline)
            PIPELINE_MODE="baseline"
            DEFAULT_INITIAL_MODEL="/path/to/models/Qwen2.5-3B-Instruct"
            DEFAULT_STAGE_ORDER="CD KK R1"
            DEFAULT_EXPERIMENT_PREFIX="EXP1-3B"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="False"
            ;;
        qwen_3b_tarp)
            PIPELINE_MODE="tarp"
            DEFAULT_INITIAL_MODEL="/path/to/models/Qwen2.5-3B-Instruct"
            DEFAULT_STAGE_ORDER="CD KK R1"
            DEFAULT_EXPERIMENT_PREFIX="EXP1-3B"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="True"
            ;;
        llama_3b_baseline)
            PIPELINE_MODE="baseline"
            DEFAULT_INITIAL_MODEL="/path/to/models/Llama-3.2-3B-Instruct"
            DEFAULT_STAGE_ORDER="CD KK R1"
            DEFAULT_EXPERIMENT_PREFIX="EXP1-L32-3B"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="False"
            DEFAULT_MODEL_ENV="llama_ins"
            DEFAULT_MODEL_TAG_PREFIX="L32-3B"
            DEFAULT_PIPELINE_NAME_TAG="L32"
            ;;
        llama_3b_tarp)
            PIPELINE_MODE="tarp"
            DEFAULT_INITIAL_MODEL="/path/to/models/Llama-3.2-3B-Instruct"
            DEFAULT_STAGE_ORDER="CD KK R1"
            DEFAULT_EXPERIMENT_PREFIX="EXP1-L32-3B"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="True"
            DEFAULT_MODEL_ENV="llama_ins"
            DEFAULT_MODEL_TAG_PREFIX="L32-3B"
            DEFAULT_PIPELINE_NAME_TAG="L32"
            ;;
        qwen_7b_baseline)
            PIPELINE_MODE="baseline"
            DEFAULT_INITIAL_MODEL="/path/to/models/Qwen2.5-7B-Instruct"
            DEFAULT_STAGE_ORDER="R1 KK CD"
            DEFAULT_EXPERIMENT_PREFIX="EXP1"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="False"
            ;;
        qwen_7b_tarp)
            PIPELINE_MODE="tarp"
            DEFAULT_INITIAL_MODEL="/path/to/models/Qwen2.5-7B-Instruct"
            DEFAULT_STAGE_ORDER="R1 KK CD"
            DEFAULT_EXPERIMENT_PREFIX="EXP1"
            DEFAULT_FORCE_ANSWER_STOP_TRAIN="False"
            ;;
        *)
            echo "Unexpected normalized PIPELINE_PRESET: ${PIPELINE_PRESET}" >&2
            exit 1
            ;;
    esac
}

join_by() {
    local delimiter="$1"
    shift || true
    local result=""
    local first=1
    local item
    for item in "$@"; do
        if (( first )); then
            result="$item"
            first=0
        else
            result+="${delimiter}${item}"
        fi
    done
    printf '%s' "$result"
}

sanitize_name() {
    printf '%s' "$1" | tr ' /:,' '_' | tr -s '_'
}

compact_tag_value() {
    local value="${1:-}"
    value="${value#\'}"
    value="${value%\'}"
    value="${value#\"}"
    value="${value%\"}"
    value="$(sanitize_name "$value")"
    value="${value,,}"
    value="${value//./p}"
    value="${value//+/_}"
    value="${value//-/_}"
    value="${value#_}"
    value="${value%_}"
    printf '%s' "$value"
}

build_stage_tarp_online_tag() {
    local lam_value="$1"
    shift || true
    local stage_prefix
    stage_prefix="$(join_by - "$@")"
    printf '%s_TARP_cap%s_lam%s' \
        "$stage_prefix" \
        "$(compact_tag_value "$TARP_TO_PG_RATIO_CAP_TAG_VALUE")" \
        "$(compact_tag_value "$lam_value")"
}

append_raw_tag_part() {
    local -n parts_ref="$1"
    local token="${2:-}"
    [[ -n "$token" ]] || return 0
    parts_ref+=("$token")
}

append_tag_part() {
    local -n parts_ref="$1"
    local key="$2"
    local value="${3:-}"
    [[ -n "$value" ]] || return 0
    parts_ref+=("${key}$(compact_tag_value "$value")")
}

has_extra_override() {
    local key="$1"
    local arg normalized
    for arg in "${PIPELINE_EXTRA_ARGS[@]}"; do
        normalized="${arg#+}"
        if [[ "$normalized" == "${key}="* ]]; then
            return 0
        fi
    done
    return 1
}

maybe_append_pipeline_override() {
    local key="$1"
    local value="$2"
    if ! has_extra_override "$key"; then
        PIPELINE_EXTRA_ARGS+=("${key}=${value}")
    fi
}

get_extra_override_value() {
    local key="$1"
    local arg normalized
    for arg in "${PIPELINE_EXTRA_ARGS[@]}"; do
        normalized="${arg#+}"
        if [[ "$normalized" == "${key}="* ]]; then
            printf '%s' "${normalized#*=}"
            return 0
        fi
    done
    return 1
}

append_tarp_reasoning_cache_args() {
    local -n args_ref="$1"

    if ! has_extra_override "actor_rollout_ref.actor.tarp_reasoning_cache_enabled"; then
        if [[ -n "${TARP_REASONING_CACHE_ENABLED:-}" ]] && [[ "${TARP_REASONING_CACHE_ENABLED,,}" != "auto" ]]; then
            args_ref+=("actor_rollout_ref.actor.tarp_reasoning_cache_enabled=${TARP_REASONING_CACHE_ENABLED}")
        fi
    fi
    if ! has_extra_override "actor_rollout_ref.actor.tarp_reasoning_cache_path"; then
        if [[ -n "${TARP_REASONING_CACHE_PATH:-}" ]]; then
            args_ref+=("actor_rollout_ref.actor.tarp_reasoning_cache_path=${TARP_REASONING_CACHE_PATH}")
        fi
    fi
    if ! has_extra_override "actor_rollout_ref.actor.tarp_reasoning_cache_key_fields"; then
        if [[ -n "${TARP_REASONING_CACHE_KEY_FIELDS:-}" ]]; then
            args_ref+=("actor_rollout_ref.actor.tarp_reasoning_cache_key_fields=${TARP_REASONING_CACHE_KEY_FIELDS}")
        fi
    fi
    if ! has_extra_override "actor_rollout_ref.actor.tarp_reasoning_cache_version"; then
        if [[ -n "${TARP_REASONING_CACHE_VERSION:-}" ]]; then
            args_ref+=("actor_rollout_ref.actor.tarp_reasoning_cache_version=${TARP_REASONING_CACHE_VERSION}")
        fi
    fi
}

read_yaml_scalar_default() {
    local key="$1"
    local yaml_path="$2"
    local line value

    [[ -f "$yaml_path" ]] || return 1

    if command -v rg >/dev/null 2>&1; then
        line="$(rg -N "^[[:space:]]*${key}:[[:space:]]*" "$yaml_path" | head -n 1 || true)"
    else
        line="$(grep -E "^[[:space:]]*${key}:[[:space:]]*" "$yaml_path" | head -n 1 || true)"
    fi
    [[ -n "$line" ]] || return 1

    value="$(printf '%s' "$line" | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//")"
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    [[ -n "$value" ]] || return 1

    printf '%s' "$value"
}

detect_model_tag() {
    local model_name
    model_name="$(basename "$1")"
    if [[ "$model_name" =~ ([0-9]+(\.[0-9]+)?B) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        sanitize_name "$model_name"
    fi
}

build_anchor_dir_for_sequence() {
    local lam_value="$1"
    shift || true
    anchor_build_suite_dir \
        "$TARP_ANCHOR_ROOT" \
        "$MODEL_TAG_PREFIX" \
        "$TARP_TO_PG_RATIO_CAP_TAG_VALUE" \
        "$lam_value" \
        "$@"
}

is_true() {
    case "${1,,}" in
        1|true|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

activate_conda() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        return
    fi

    # shellcheck disable=SC1091
    source activate "$CONDA_ENV_NAME"
}

validate_stage() {
    local stage="$1"
    if [[ -z "${STAGE_STEPS[$stage]+x}" ]]; then
        echo "Unsupported stage: $stage" >&2
        exit 1
    fi
}

run_math_and_code() {
    local output_log="$1"
    shift || true
    local base_model_path="${RUN_MATH_AND_CODE_BASE_MODEL_OVERRIDE:-$current_base_model}"

    (
        export MODEL="$MODEL"
        export DATA_NAME="$experiment_name"
        export EXPERIMENT_NAME="$experiment_name"
        export BASE_MODEL="$base_model_path"
        export DIR="$stage_dir"
        export TRAIN_DATA="${STAGE_TRAIN_DATA[$stage]}"
        export TEST_DATA="${STAGE_TEST_DATA[$stage]}"
        export N_GPUS="$N_GPUS"
        export VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND"
        export LOG_FILE="$output_log"
        export PROJECT_NAME="$PROJECT_NAME"
        export PPO_MICRO_BATCH_SIZE="$PPO_MICRO_BATCH_SIZE"
        export PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE"
        export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
        export EPOCHS="$steps"
        export MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
        export MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH"
        export ROLLOUT_N="$ROLLOUT_N"
        export LOGGER="$LOGGER"
        export VAL_ONLY="$VAL_ONLY"
        export LR="${STAGE_LR[$stage]}"
        export SAVE_FREQ="${STAGE_SAVE_FREQ[$stage]}"
        export TEST_FREQ="${STAGE_TEST_FREQ[$stage]}"

        bash "$ROOT_DIR/examples/grpo_trainer/math_and_code.sh" "$@"
    ) 2>&1 | tee "$output_log"
}

find_online_prior_rank0_path() {
    local target_tag="${1:-${stage_tarp_tag:-}}"
    local target_dir="${2:-${stage_tarp_dir:-}}"
    [[ -n "$target_tag" ]] || return 1
    [[ "$target_tag" != "null" ]] || return 1
    [[ -n "$target_dir" ]] || return 1
    [[ "$target_dir" != "null" ]] || return 1
    find "$target_dir" -maxdepth 1 -type f -name "${target_tag}__merged__rank-0-of-*.pt" | sort | head -n 1
}

online_prior_contains_experiment() {
    local target_experiment_name="$1"
    local target_tag="${2:-${stage_tarp_tag:-}}"
    local target_dir="${3:-${stage_tarp_dir:-}}"
    local prior_path
    prior_path="$(find_online_prior_rank0_path "$target_tag" "$target_dir")"
    if [[ -z "$prior_path" ]]; then
        return 1
    fi

    python - "$prior_path" "$target_experiment_name" <<'PY' >/dev/null
import sys
import torch

path, target = sys.argv[1], sys.argv[2]
state = torch.load(path, map_location="cpu")
meta = state.get("_meta", {}) if isinstance(state, dict) else {}
history = list(meta.get("online_history", [])) if isinstance(meta, dict) else []
if not history and isinstance(meta, dict) and meta.get("experiment_name"):
    history = [meta["experiment_name"]]
raise SystemExit(0 if target in history else 1)
PY
}

maybe_bootstrap_online_tarp_from_merged_model() {
    local source_merged_model_path="${1:-$merged_model_path}"

    if [[ "$PIPELINE_MODE" != "tarp" ]]; then
        return 0
    fi
    if (( ! TARP_USE_ONLINE_TARP )); then
        return 0
    fi
    if [[ "${stage_tarp_train_mode,,}" == "baseline" ]]; then
        return 0
    fi
    if online_prior_contains_experiment "$experiment_name"; then
        echo "Online TARP prior already includes $experiment_name. Skipping TARP export-only bootstrap."
        return 0
    fi
    if (( current_continual_stage_index > 1 )); then
        echo "Existing outputs for $experiment_name are stale under online TARP." >&2
        echo "Reason: this is continual stage #${current_continual_stage_index}, so its objective depends on an online TARP prior from earlier stages." >&2
        echo "The current online prior does not include $experiment_name yet, which means this stage has not been retrained with the TARP penalty you now want." >&2
        echo "Use a fresh EXPERIMENT_PREFIX/MODEL_ROOT for KK and later stages, or remove the old outputs under $stage_dir and rerun." >&2
        exit 1
    fi

    if ! merged_model_is_ready "$source_merged_model_path"; then
        echo "Cannot bootstrap online TARP prior from merged weights; merged model missing or incomplete: ${source_merged_model_path}" >&2
        exit 1
    fi

    local export_log="${LOG_ROOT}/$(sanitize_name "${model_tag}__${experiment_name}__tarp_export_only").log"
    local export_hydra_run_dir=""
    if (( MANAGE_HYDRA_RUN_DIR )); then
        export_hydra_run_dir="${HYDRA_RUN_BASE_DIR}/${PIPELINE_NAME}/${LOG_RUN_TAG}/$(sanitize_name "${experiment_name}__tarp_export_only")"
    fi
    echo "Online TARP prior does not yet include $experiment_name. Exporting Fisher from merged model ${source_merged_model_path} without PPO training."

    if is_true "$STOP_RAY_BEFORE_STAGE"; then
        ray stop --force >/dev/null 2>&1 || true
        sleep 1
    fi

    local export_args=(
        "trainer.resume_mode=disable"
        "trainer.total_training_steps=$steps"
        "trainer.tarp_export_only=True"
        "actor_rollout_ref.actor.tarp_export_only=True"
        "actor_rollout_ref.rollout.force_answer_stop=${FORCE_ANSWER_STOP_TRAIN}"
        "actor_rollout_ref.actor.tarp_root=${TARP_ANCHOR_ROOT}"
        "actor_rollout_ref.actor.tarp_train_mode=${stage_tarp_train_mode}"
        "actor_rollout_ref.actor.tarp_load_dataset_key=${tarp_load_dataset_key}"
        "actor_rollout_ref.actor.tarp_load_dir=${stage_tarp_load_dir}"
        "actor_rollout_ref.actor.tarp_load_tag=${stage_tarp_load_tag}"
        "actor_rollout_ref.actor.tarp_dir=${stage_tarp_dir}"
        "actor_rollout_ref.actor.tarp_tag=${stage_tarp_tag}"
        "actor_rollout_ref.actor.online_tarp=${stage_online_tarp}"
        "actor_rollout_ref.actor.tarp_online_decay=${stage_tarp_online_decay}"
        "actor_rollout_ref.actor.tarp_mode=${STAGE_TARP_MODE[$stage]}"
        "actor_rollout_ref.actor.tarp_lambda=${STAGE_TARP_LAMBDA[$stage]}"
    )
    append_tarp_reasoning_cache_args export_args
    (
        if [[ -n "$export_hydra_run_dir" ]]; then
            export HYDRA_RUN_DIR="$export_hydra_run_dir"
        fi
        RUN_MATH_AND_CODE_BASE_MODEL_OVERRIDE="$source_merged_model_path"
        run_math_and_code "$export_log" "${export_args[@]}" "${PIPELINE_EXTRA_ARGS[@]}"
    )
}

find_reusable_merged_model_path() {
    local primary_path="$1"
    local legacy_primary_path="${2:-}"
    local candidate

    for candidate in "$primary_path" "$legacy_primary_path"; do
        [[ -n "$candidate" ]] || continue
        if merged_model_is_ready "$candidate"; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    if [[ "$PIPELINE_MODE" != "tarp" ]]; then
        return 1
    fi
    if (( ! TARP_USE_ONLINE_TARP )); then
        return 1
    fi
    if (( current_continual_stage_index != 1 )); then
        return 1
    fi
    if ! is_true "${TARP_BOOTSTRAP_FIRST_CONTINUAL_STAGE:-1}"; then
        return 1
    fi
    if [[ -z "${TARP_BOOTSTRAP_MERGE_ROOT:-}" ]]; then
        return 1
    fi

    local bootstrap_candidates=(
        "${TARP_BOOTSTRAP_MERGE_ROOT}/$(decorate_model_tag "$target_model_tag")"
        "${TARP_BOOTSTRAP_MERGE_ROOT}/${target_model_tag}"
    )
    for candidate in "${bootstrap_candidates[@]}"; do
        if [[ "$candidate" == "$primary_path" ]] || [[ -n "$legacy_primary_path" && "$candidate" == "$legacy_primary_path" ]]; then
            continue
        fi
        if merged_model_is_ready "$candidate"; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

build_stage_model_tag() {
    local base_tag="$1"
    local stage_name="$2"
    local step="$3"
    printf '%s-%s%s' "$base_tag" "$stage_name" "$step"
}

build_pipeline_step_signature() {
    local parts=()
    local stage
    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        parts+=("${stage}${STAGE_STEPS[$stage]}")
    done
    join_by - "${parts[@]}"
}

build_stagewise_assoc_tag() {
    local label="$1"
    local assoc_name="$2"
    shift 2
    local -n assoc_ref="$assoc_name"
    local stages=("$@")
    local first_value=""
    local has_value=0
    local all_same=1
    local stage value
    local parts=()

    for stage in "${stages[@]}"; do
        [[ -n "${assoc_ref[$stage]+x}" ]] || continue
        value="${assoc_ref[$stage]}"
        if (( ! has_value )); then
            first_value="$value"
            has_value=1
        elif [[ "$value" != "$first_value" ]]; then
            all_same=0
        fi
    done

    if (( ! has_value )); then
        return 0
    fi

    if (( all_same )); then
        printf '%s%s' "$label" "$(compact_tag_value "$first_value")"
        return 0
    fi

    for stage in "${stages[@]}"; do
        value="${assoc_ref[$stage]}"
        parts+=("${stage,,}${label}$(compact_tag_value "$value")")
    done
    join_by - "${parts[@]}"
}

select_primary_tarp_stage_for_tag() {
    if [[ "$PIPELINE_MODE" != "tarp" ]]; then
        return 1
    fi

    local stage stage_train_mode
    local continual_stage_index=0

    if (( TARP_USE_ONLINE_TARP )); then
        for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
            stage_train_mode="${STAGE_TARP_TRAIN_MODE[$stage]}"
            if [[ "${stage_train_mode,,}" == "baseline" ]]; then
                continue
            fi
            continual_stage_index=$((continual_stage_index + 1))
            if (( continual_stage_index > 1 )); then
                printf '%s' "$stage"
                return 0
            fi
        done
    fi

    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        stage_train_mode="${STAGE_TARP_TRAIN_MODE[$stage]}"
        if [[ "${stage_train_mode,,}" != "baseline" ]]; then
            printf '%s' "$stage"
            return 0
        fi
    done

    return 1
}

build_tarp_tuning_signature() {
    if [[ "$PIPELINE_MODE" != "tarp" ]]; then
        return 0
    fi

    local parts=()
    local tarp_stages=()
    local stage stage_train_mode

    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        stage_train_mode="${STAGE_TARP_TRAIN_MODE[$stage]}"
        if [[ "${stage_train_mode,,}" != "baseline" ]]; then
            tarp_stages+=("$stage")
        fi
    done

    if [[ -n "${TARP_LAMBDA_TAG_VALUE:-}" ]]; then
        append_tag_part parts "lam" "$TARP_LAMBDA_TAG_VALUE"
    else
        append_raw_tag_part parts "$(build_stagewise_assoc_tag "lam" STAGE_TARP_LAMBDA "${tarp_stages[@]}")"
    fi
    if (( TARP_USE_ONLINE_TARP )); then
        append_tag_part parts "dec" "${TARP_ONLINE_DECAY_TAG_VALUE:-$TARP_ONLINE_DECAY}"
    fi
    append_tag_part parts "tvs" "${TARP_TARGET_VALID_SAMPLES_TAG_VALUE:-20}"
    append_tag_part parts "traj" "${TARP_N_TRAJ_TAG_VALUE:-4}"
    if [[ -n "${TARP_MAX_NEW_TOKENS_TAG_VALUE:-}" ]]; then
        append_tag_part parts "new" "$TARP_MAX_NEW_TOKENS_TAG_VALUE"
    fi
    if [[ -n "${TARP_MAX_FISHER_SEQ_LEN_TAG_VALUE:-}" ]]; then
        append_tag_part parts "fseq" "$TARP_MAX_FISHER_SEQ_LEN_TAG_VALUE"
    fi

    join_by - "${parts[@]}"
}

decorate_model_tag() {
    local base_tag="$1"
    if [[ -z "${MODEL_TAG_TUNING_SIGNATURE:-}" ]]; then
        printf '%s' "$base_tag"
    else
        printf '%s-%s' "$base_tag" "$MODEL_TAG_TUNING_SIGNATURE"
    fi
}

build_default_tarp_online_tag() {
    local model_variant_tag
    local step_signature
    local tuning_signature
    model_variant_tag="$(sanitize_name "${MODEL_TAG_PREFIX}_${MODEL_VARIANT_DIR}")"
    step_signature="$(build_pipeline_step_signature)"
    tuning_signature="${MODEL_TAG_TUNING_SIGNATURE:-}"
    if [[ -n "$tuning_signature" ]]; then
        sanitize_name "$(join_by __ "$model_variant_tag" "$ONLINE_EXPERIMENT_PREFIX" "$step_signature" "$tuning_signature" "online_tarp")"
    else
        sanitize_name "$(join_by __ "$model_variant_tag" "$ONLINE_EXPERIMENT_PREFIX" "$step_signature" "online_tarp")"
    fi
}

build_default_log_run_tag() {
    local model_variant_tag
    local step_signature
    local tuning_signature
    model_variant_tag="$(sanitize_name "${MODEL_TAG_PREFIX}_${MODEL_VARIANT_DIR}")"
    step_signature="$(build_pipeline_step_signature)"
    tuning_signature="${MODEL_TAG_TUNING_SIGNATURE:-}"
    if [[ -n "$tuning_signature" ]]; then
        sanitize_name "$(join_by __ "$model_variant_tag" "$LOG_EXPERIMENT_PREFIX" "$step_signature" "$tuning_signature")"
    else
        sanitize_name "$(join_by __ "$model_variant_tag" "$LOG_EXPERIMENT_PREFIX" "$step_signature")"
    fi
}

build_default_runtime_bucket() {
    local order_tag="$1"
    printf 'ppo_%s' "$(compact_tag_value "$order_tag")"
}

merged_model_is_ready() {
    local merged_dir="$1"
    [[ -d "$merged_dir" ]] || return 1
    [[ -f "${merged_dir}/config.json" ]] || return 1

    if compgen -G "${merged_dir}/*.safetensors" >/dev/null; then
        return 0
    fi
    if compgen -G "${merged_dir}/*.bin" >/dev/null; then
        return 0
    fi
    [[ -f "${merged_dir}/model.safetensors.index.json" ]] && return 0
    [[ -f "${merged_dir}/pytorch_model.bin.index.json" ]] && return 0
    return 1
}

run_stage_merge() {
    local checkpoint_actor_dir="$1"
    local merged_path="$2"
    local base_model_path="$3"

    local merge_args=()
    if (( PIPELINE_USE_LORA )); then
        merge_args=(
            --base-model "$base_model_path"
            --checkpoint-actor-dir "$checkpoint_actor_dir"
            --merge-path "$merged_path"
            --device "$LORA_MERGE_DEVICE"
            --dtype "$LORA_MERGE_DTYPE"
        )
        if is_true "$LORA_MERGE_TRUST_REMOTE_CODE"; then
            merge_args+=(--trust-remote-code)
        fi
        if is_true "$OVERWRITE_MERGED_MODEL"; then
            merge_args+=(--overwrite)
        fi
        python3 "$ROOT_DIR/scripts/merge_lora_checkpoint.py" "${merge_args[@]}"
    else
        merge_args=(
            --local-dir "$checkpoint_actor_dir"
            --merge-path "$merged_path"
        )
        if is_true "$OVERWRITE_MERGED_MODEL"; then
            merge_args+=(--overwrite)
        fi
        python3 "$ROOT_DIR/merge_model.py" "${merge_args[@]}"
    fi
}

read_latest_checkpointed_step() {
    local tracker_path="${stage_dir}/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker_path" ]]; then
        return 0
    fi

    local tracker_step
    tracker_step="$(<"$tracker_path")"
    if [[ "$tracker_step" =~ ^[0-9]+$ ]]; then
        printf '%s' "$tracker_step"
    fi
}

merge_and_delete_checkpoint_step() {
    local checkpoint_step="$1"
    local checkpoint_dir="${stage_dir}/global_step_${checkpoint_step}"
    local checkpoint_actor_dir="${checkpoint_dir}/actor"
    if [[ ! -d "$checkpoint_actor_dir" ]]; then
        return 0
    fi

    local merged_tag
    merged_tag="$(decorate_model_tag "$(build_stage_model_tag "$stage_model_tag_base" "$stage" "$checkpoint_step")")"
    local merged_path="${MERGE_ROOT}/${merged_tag}"

    echo "Merging intermediate checkpoint: ${checkpoint_actor_dir} -> ${merged_path}"
    if [[ -d "$merged_path" ]] && ! is_true "$OVERWRITE_MERGED_MODEL"; then
        if ! merged_model_is_ready "$merged_path"; then
            echo "Existing intermediate merged model looks incomplete: $merged_path" >&2
            return 1
        fi
        echo "Reusing existing intermediate merged model: $merged_path"
    else
        run_stage_merge "$checkpoint_actor_dir" "$merged_path" "$current_base_model"
        if ! merged_model_is_ready "$merged_path"; then
            echo "Merged model validation failed: $merged_path" >&2
            return 1
        fi
    fi

    echo "Deleting raw checkpoint directory after successful intermediate merge: $checkpoint_dir"
    rm -rf "$checkpoint_dir"
}

process_stale_stage_checkpoints() {
    local latest_step="$1"
    if [[ ! "$latest_step" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    local checkpoint_dir
    local checkpoint_step
    local checkpoint_steps=()

    shopt -s nullglob
    for checkpoint_dir in "${stage_dir}"/global_step_*; do
        [[ -d "$checkpoint_dir" ]] || continue
        checkpoint_step="$(basename "$checkpoint_dir")"
        checkpoint_step="${checkpoint_step#global_step_}"
        if [[ "$checkpoint_step" =~ ^[0-9]+$ ]] && (( checkpoint_step < latest_step )); then
            checkpoint_steps+=("$checkpoint_step")
        fi
    done
    shopt -u nullglob

    if (( ${#checkpoint_steps[@]} == 0 )); then
        return 0
    fi

    mapfile -t checkpoint_steps < <(printf '%s\n' "${checkpoint_steps[@]}" | sort -n)
    for checkpoint_step in "${checkpoint_steps[@]}"; do
        merge_and_delete_checkpoint_step "$checkpoint_step"
    done
}

watch_stage_checkpoints() {
    local training_pid="$1"
    local last_seen_step=""
    local latest_step=""

    while kill -0 "$training_pid" 2>/dev/null; do
        latest_step="$(read_latest_checkpointed_step || true)"
        if [[ -n "$latest_step" ]] && [[ "$latest_step" != "$last_seen_step" ]]; then
            process_stale_stage_checkpoints "$latest_step"
            last_seen_step="$latest_step"
        fi
        sleep 10
    done

    latest_step="$(read_latest_checkpointed_step || true)"
    if [[ -n "$latest_step" ]] && [[ "$latest_step" != "$last_seen_step" ]]; then
        process_stale_stage_checkpoints "$latest_step"
    fi
}

if (( $# > 0 )); then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
fi

PIPELINE_PRESET="${PIPELINE_PRESET:-}"
if [[ -z "$PIPELINE_PRESET" ]] && (( $# > 0 )); then
        if normalized="$(normalize_preset "$1" 2>/dev/null)"; then
            PIPELINE_PRESET="$normalized"
            shift
        fi
fi
PIPELINE_EXTRA_ARGS=("$@")
PIPELINE_PRESET="${PIPELINE_PRESET:-qwen_7b_baseline}"

raw_pipeline_preset="$PIPELINE_PRESET"
if ! normalized_preset="$(normalize_preset "$PIPELINE_PRESET" 2>/dev/null)"; then
    echo "Unsupported PIPELINE_PRESET: ${raw_pipeline_preset}" >&2
    echo "Run with --help to see supported presets." >&2
    exit 1
fi
PIPELINE_PRESET="$normalized_preset"
set_preset_defaults
MANAGE_HYDRA_RUN_DIR=1
if has_extra_override "hydra.run.dir"; then
    MANAGE_HYDRA_RUN_DIR=0
fi

if [[ "$PIPELINE_PRESET" == "qwen_3b_tarp" || "$PIPELINE_PRESET" == "llama_3b_tarp" ]]; then
    # Keep the online Fisher export settings aligned with the original CD prior recipe,
    # unless the caller explicitly overrides any of them.
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_target_valid_samples" "32"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_n_traj" "4"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_sem_metric" "lexical"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_min_correct_ratio" "0.25"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_fisher_fast_export" "False"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_fisher_generate_use_cache" "True"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_reasoning_cache_enabled" "True"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_fisher_allow_weak_samples_export" "True"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_fisher_weak_min_score_export" "0.0"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_max_new_tokens" "512"
    maybe_append_pipeline_override "actor_rollout_ref.actor.tarp_max_fisher_seq_len" "512"
    maybe_append_pipeline_override "actor_rollout_ref.rollout.force_answer_stop" "True"
    maybe_append_pipeline_override "actor_rollout_ref.rollout.gpu_memory_utilization" "0.35"
    maybe_append_pipeline_override "actor_rollout_ref.rollout.max_num_batched_tokens" "16384"
fi

PIPELINE_LORA_RANK_RAW="$(get_extra_override_value "actor_rollout_ref.model.lora_rank" || true)"
PIPELINE_LORA_RANK="${PIPELINE_LORA_RANK_RAW%\"}"
PIPELINE_LORA_RANK="${PIPELINE_LORA_RANK#\"}"
PIPELINE_LORA_RANK="${PIPELINE_LORA_RANK%\'}"
PIPELINE_LORA_RANK="${PIPELINE_LORA_RANK#\'}"
PIPELINE_USE_LORA=0
if [[ "$PIPELINE_LORA_RANK" =~ ^[0-9]+$ ]] && (( PIPELINE_LORA_RANK > 0 )); then
    PIPELINE_USE_LORA=1
fi
LORA_MERGE_DEVICE="${LORA_MERGE_DEVICE:-cpu}"
LORA_MERGE_DTYPE="${LORA_MERGE_DTYPE:-bfloat16}"
LORA_MERGE_TRUST_REMOTE_CODE="${LORA_MERGE_TRUST_REMOTE_CODE:-0}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-data_rl}"
INITIAL_MODEL="${INITIAL_MODEL:-$DEFAULT_INITIAL_MODEL}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/outputs/models}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES

STAGE_ORDER_STRING="${STAGE_ORDER:-$DEFAULT_STAGE_ORDER}"
read -r -a STAGE_ORDER <<< "$STAGE_ORDER_STRING"

NORMALIZED_STAGE_ORDER=()
for raw_stage in "${STAGE_ORDER[@]}"; do
    NORMALIZED_STAGE_ORDER+=("${raw_stage^^}")
done
FINAL_STAGE_INDEX=$((${#NORMALIZED_STAGE_ORDER[@]} - 1))
FINAL_STAGE="${NORMALIZED_STAGE_ORDER[$FINAL_STAGE_INDEX]}"

MODEL="${MODEL:-$DEFAULT_MODEL_ENV}"
case "$MODEL" in
    qwen)
        DEFAULT_MODEL_VARIANT_DIR="base"
        ;;
    *)
        DEFAULT_MODEL_VARIANT_DIR="instruct"
        ;;
esac
MODEL_VARIANT_DIR="${MODEL_VARIANT_DIR:-$DEFAULT_MODEL_VARIANT_DIR}"
if [[ -n "$DEFAULT_MODEL_TAG_PREFIX" ]]; then
    MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-$DEFAULT_MODEL_TAG_PREFIX}"
else
    MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-$(detect_model_tag "$INITIAL_MODEL")}"
fi
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-$DEFAULT_EXPERIMENT_PREFIX}"

N_GPUS="${N_GPUS:-${#VISIBLE_GPU_LIST[@]}}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
PROJECT_NAME="${PROJECT_NAME:-ICML26}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-4}"
FORCE_ANSWER_STOP_TRAIN="${FORCE_ANSWER_STOP_TRAIN:-$DEFAULT_FORCE_ANSWER_STOP_TRAIN}"
LOGGER="${LOGGER:-['console']}"
VAL_ONLY="${VAL_ONLY:-False}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_FREQ="${SAVE_FREQ:-25}"
# SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
DEFAULT_LR="${DEFAULT_LR:-5e-6}"

STOP_RAY_BEFORE_STAGE="${STOP_RAY_BEFORE_STAGE:-0}"
START_FRESH="${START_FRESH:-0}"
TRAIN_RESUME_MODE="${TRAIN_RESUME_MODE:-auto}"
REUSE_TARGET_CHECKPOINT="${REUSE_TARGET_CHECKPOINT:-1}"
ALLOW_EXISTING_STAGE_OUTPUT="${ALLOW_EXISTING_STAGE_OUTPUT:-1}"
SKIP_IF_MERGED_EXISTS="${SKIP_IF_MERGED_EXISTS:-1}"
OVERWRITE_MERGED_MODEL="${OVERWRITE_MERGED_MODEL:-1}"

declare -A STAGE_STEPS=(
    [CD]="${CD_STEPS:-200}"
    [KK]="${KK_STEPS:-300}"
    [R1]="${R1_STEPS:-300}"
)

declare -A STAGE_EXPERIMENT_NAME

declare -A STAGE_TRAIN_DATA=(
    [CD]="${CD_TRAIN_DATA:-data/countdown/train_ins.parquet}"
    [KK]="${KK_TRAIN_DATA:-data/kk/train_ins.parquet}"
    [R1]="${R1_TRAIN_DATA:-data/code-r1-12k/train_ins.parquet}"
)

declare -A STAGE_TEST_DATA=(
    [CD]="${CD_TEST_DATA:-data/countdown/test_ins.parquet,data/kk/test_ins.parquet,data/code-r1-12k/test_ins.parquet}"
    [KK]="${KK_TEST_DATA:-data/countdown/test_ins.parquet,data/kk/test_ins.parquet,data/code-r1-12k/test_ins.parquet}"
    [R1]="${R1_TEST_DATA:-data/countdown/test_ins.parquet,data/kk/test_ins.parquet,data/code-r1-12k/test_ins.parquet}"
)

declare -A STAGE_LR=(
    [CD]="${CD_LR:-$DEFAULT_LR}"
    [KK]="${KK_LR:-$DEFAULT_LR}"
    [R1]="${R1_LR:-$DEFAULT_LR}"
)

declare -A STAGE_SAVE_FREQ=(
    [CD]="${CD_SAVE_FREQ:-$SAVE_FREQ}"
    [KK]="${KK_SAVE_FREQ:-$SAVE_FREQ}"
    [R1]="${R1_SAVE_FREQ:-$SAVE_FREQ}"
)

declare -A STAGE_TEST_FREQ=(
    [CD]="${CD_TEST_FREQ:-$TEST_FREQ}"
    [KK]="${KK_TEST_FREQ:-$TEST_FREQ}"
    [R1]="${R1_TEST_FREQ:-$TEST_FREQ}"
)

declare -A STAGE_VAL_BEFORE_TRAIN=(
    [CD]="${CD_VAL_BEFORE_TRAIN:-$VAL_BEFORE_TRAIN}"
    [KK]="${KK_VAL_BEFORE_TRAIN:-$VAL_BEFORE_TRAIN}"
    [R1]="${R1_VAL_BEFORE_TRAIN:-$VAL_BEFORE_TRAIN}"
)

TARP_USE_ONLINE_TARP=0
if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    TARP_ANCHOR_STAGE="${TARP_ANCHOR_STAGE:-${NORMALIZED_STAGE_ORDER[0]}}"
    ONLINE_EXPERIMENT_PREFIX_EXPLICIT=0
    if [[ -n "${ONLINE_EXPERIMENT_PREFIX+x}" ]]; then
        ONLINE_EXPERIMENT_PREFIX_EXPLICIT=1
    fi
    ONLINE_EXPERIMENT_PREFIX="${ONLINE_EXPERIMENT_PREFIX:-${EXPERIMENT_PREFIX}-TARP}"
    PIPELINE_TARP_MODE="${PIPELINE_TARP_MODE:-continual}"
    PIPELINE_TARP_STRATEGY="${PIPELINE_TARP_STRATEGY:-online}"
    TARP_ONLINE_DECAY="${TARP_ONLINE_DECAY:-1.0}"

    declare -A STAGE_TARP_TRAIN_MODE=(
        [CD]="${CD_TARP_TRAIN_MODE:-$PIPELINE_TARP_MODE}"
        [KK]="${KK_TARP_TRAIN_MODE:-$PIPELINE_TARP_MODE}"
        [R1]="${R1_TARP_TRAIN_MODE:-$PIPELINE_TARP_MODE}"
    )

    declare -A STAGE_TARP_MODE=(
        [CD]="${CD_TARP_MODE:-TARP}"
        [KK]="${KK_TARP_MODE:-TARP}"
        [R1]="${R1_TARP_MODE:-TARP}"
    )

    declare -A STAGE_TARP_LAMBDA=(
        [CD]="${CD_TARP_LAMBDA:-0.01}"
        [KK]="${KK_TARP_LAMBDA:-0.01}"
        [R1]="${R1_TARP_LAMBDA:-0.01}"
    )
fi

if is_true "$START_FRESH"; then
    TRAIN_RESUME_MODE="disable"
    REUSE_TARGET_CHECKPOINT="0"
    ALLOW_EXISTING_STAGE_OUTPUT="0"
    SKIP_IF_MERGED_EXISTS="0"
fi

for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
    validate_stage "$stage"
done

if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    case "${PIPELINE_TARP_STRATEGY,,}" in
        online|online_tarp|online-tarp)
            TARP_USE_ONLINE_TARP=1
            ;;
        anchor|single_anchor|single-anchor)
            TARP_USE_ONLINE_TARP=0
            validate_stage "${TARP_ANCHOR_STAGE^^}"
            TARP_ANCHOR_STAGE="${TARP_ANCHOR_STAGE^^}"
            ;;
        *)
            echo "Unsupported PIPELINE_TARP_STRATEGY: $PIPELINE_TARP_STRATEGY" >&2
            exit 1
            ;;
    esac
fi

PPO_TRAINER_CONFIG_PATH="${PPO_TRAINER_CONFIG_PATH:-${ROOT_DIR}/verl/trainer/config/ppo_trainer.yaml}"
TARP_TARGET_VALID_SAMPLES_DEFAULT="$(read_yaml_scalar_default "tarp_target_valid_samples" "$PPO_TRAINER_CONFIG_PATH" || true)"
if [[ -z "$TARP_TARGET_VALID_SAMPLES_DEFAULT" ]]; then
    TARP_TARGET_VALID_SAMPLES_DEFAULT="64"
fi
TARP_TO_PG_RATIO_CAP_DEFAULT="$(read_yaml_scalar_default "tarp_to_pg_ratio_cap" "$PPO_TRAINER_CONFIG_PATH" || true)"
if [[ -z "$TARP_TO_PG_RATIO_CAP_DEFAULT" ]]; then
    TARP_TO_PG_RATIO_CAP_DEFAULT="0"
fi

TARP_LAMBDA_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_lambda" || true)"
TARP_TO_PG_RATIO_CAP_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_to_pg_ratio_cap" || true)"
if [[ -z "$TARP_TO_PG_RATIO_CAP_TAG_VALUE" ]]; then
    TARP_TO_PG_RATIO_CAP_TAG_VALUE="$TARP_TO_PG_RATIO_CAP_DEFAULT"
fi
TARP_ONLINE_DECAY_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_online_decay" || true)"
TARP_TARGET_VALID_SAMPLES_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_target_valid_samples" || true)"
if [[ -z "$TARP_TARGET_VALID_SAMPLES_TAG_VALUE" ]]; then
    TARP_TARGET_VALID_SAMPLES_TAG_VALUE="$TARP_TARGET_VALID_SAMPLES_DEFAULT"
fi
TARP_N_TRAJ_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_n_traj" || true)"
if [[ -z "$TARP_N_TRAJ_TAG_VALUE" ]]; then
    TARP_N_TRAJ_TAG_VALUE="4"
fi
TARP_MAX_NEW_TOKENS_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_max_new_tokens" || true)"
TARP_MAX_FISHER_SEQ_LEN_TAG_VALUE="$(get_extra_override_value "actor_rollout_ref.actor.tarp_max_fisher_seq_len" || true)"
TARP_REASONING_CACHE_ENABLED="${TARP_REASONING_CACHE_ENABLED:-auto}"
TARP_REASONING_CACHE_PATH="${TARP_REASONING_CACHE_PATH:-}"
TARP_REASONING_CACHE_KEY_FIELDS="${TARP_REASONING_CACHE_KEY_FIELDS:-}"
TARP_REASONING_CACHE_VERSION="${TARP_REASONING_CACHE_VERSION:-}"
TARP_ANCHOR_ROOT="${TARP_ANCHOR_ROOT:-${TARP_ROOT:-/path/to/outputs/anchor}}"
MODEL_TAG_TUNING_SIGNATURE="$(build_tarp_tuning_signature)"
if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    TARP_PRIMARY_STAGE_FOR_TAG="$(select_primary_tarp_stage_for_tag || true)"
    if [[ -n "${TARP_LAMBDA_TAG_VALUE:-}" ]]; then
        TARP_PRIMARY_LAMBDA_TAG_VALUE="$TARP_LAMBDA_TAG_VALUE"
    elif [[ -n "${TARP_PRIMARY_STAGE_FOR_TAG:-}" ]]; then
        TARP_PRIMARY_LAMBDA_TAG_VALUE="${STAGE_TARP_LAMBDA[$TARP_PRIMARY_STAGE_FOR_TAG]}"
    else
        TARP_PRIMARY_LAMBDA_TAG_VALUE="0"
    fi
    TARP_CAP_LAM_SIGNATURE="TARP_cap$(compact_tag_value "$TARP_TO_PG_RATIO_CAP_TAG_VALUE")_lam$(compact_tag_value "$TARP_PRIMARY_LAMBDA_TAG_VALUE")"
fi
if [[ "$PIPELINE_PRESET" == "qwen_3b_tarp" || "$PIPELINE_PRESET" == "llama_3b_tarp" ]] && [[ "$PIPELINE_MODE" == "tarp" ]]; then
    MODEL_TAG_TUNING_SIGNATURE="$TARP_CAP_LAM_SIGNATURE"
    if (( ! ONLINE_EXPERIMENT_PREFIX_EXPLICIT )); then
        ONLINE_EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}-${TARP_CAP_LAM_SIGNATURE}"
    fi
fi

cumulative_stage_names=()
stage_position=0
for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
    cumulative_stage_names+=("$stage")

    if [[ "$PIPELINE_MODE" == "tarp" ]]; then
        stage_position=$((stage_position + 1))
        if (( TARP_USE_ONLINE_TARP )) && (( stage_position > 1 )); then
            default_experiment_name="${ONLINE_EXPERIMENT_PREFIX}-$(join_by - "${cumulative_stage_names[@]}")"
        else
            default_experiment_name="${EXPERIMENT_PREFIX}-$(join_by - "${cumulative_stage_names[@]}")"
        fi
    else
        default_experiment_name="${EXPERIMENT_PREFIX}-$(join_by - "${cumulative_stage_names[@]}")"
    fi

    case "$stage" in
        CD)
            STAGE_EXPERIMENT_NAME[$stage]="${CD_EXPERIMENT_NAME:-$default_experiment_name}"
            ;;
        KK)
            STAGE_EXPERIMENT_NAME[$stage]="${KK_EXPERIMENT_NAME:-$default_experiment_name}"
            ;;
        R1)
            STAGE_EXPERIMENT_NAME[$stage]="${R1_EXPERIMENT_NAME:-$default_experiment_name}"
            ;;
    esac
done

declare -A STAGE_ONLINE_SAVE_TAG
declare -A STAGE_ONLINE_LOAD_TAG
declare -A STAGE_ONLINE_SAVE_DIR
declare -A STAGE_ONLINE_LOAD_DIR
TARP_ANCHOR_DIR="${TARP_ANCHOR_DIR:-}"
TARP_ANCHOR_TAG="${TARP_ANCHOR_TAG:-}"
if [[ "$PIPELINE_MODE" == "tarp" ]] && (( ! TARP_USE_ONLINE_TARP )); then
    TARP_ANCHOR_DIR="${TARP_ANCHOR_DIR:-$(build_anchor_dir_for_sequence 0.0 "$TARP_ANCHOR_STAGE")}"
    TARP_ANCHOR_TAG="${TARP_ANCHOR_TAG:-TARP__${TARP_ANCHOR_STAGE}__orig}"
fi
if [[ "$PIPELINE_MODE" == "tarp" ]] && (( TARP_USE_ONLINE_TARP )); then
    continual_stage_names=()
    previous_online_tag=""
    previous_online_dir=""
    explicit_first_load_tag="${TARP_ONLINE_LOAD_TAG:-}"
    explicit_first_load_dir="${TARP_ONLINE_LOAD_DIR:-}"
    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        if [[ "${STAGE_TARP_TRAIN_MODE[$stage],,}" == "baseline" ]]; then
            continue
        fi
        continual_stage_names+=("$stage")
        if [[ -n "${TARP_ONLINE_TAG:-}" ]]; then
            STAGE_ONLINE_SAVE_TAG[$stage]="$TARP_ONLINE_TAG"
        else
            STAGE_ONLINE_SAVE_TAG[$stage]="$(build_stage_tarp_online_tag "${STAGE_TARP_LAMBDA[$stage]}" "${continual_stage_names[@]}")"
        fi
        if [[ -n "${TARP_ONLINE_DIR:-}" ]]; then
            STAGE_ONLINE_SAVE_DIR[$stage]="$TARP_ONLINE_DIR"
        else
            STAGE_ONLINE_SAVE_DIR[$stage]="$(build_anchor_dir_for_sequence "${STAGE_TARP_LAMBDA[$stage]}" "${continual_stage_names[@]}")"
        fi
        if [[ -n "$previous_online_tag" ]]; then
            STAGE_ONLINE_LOAD_TAG[$stage]="$previous_online_tag"
            STAGE_ONLINE_LOAD_DIR[$stage]="$previous_online_dir"
        elif [[ -n "$explicit_first_load_tag" ]]; then
            STAGE_ONLINE_LOAD_TAG[$stage]="$explicit_first_load_tag"
            STAGE_ONLINE_LOAD_DIR[$stage]="$explicit_first_load_dir"
        else
            STAGE_ONLINE_LOAD_TAG[$stage]=""
            STAGE_ONLINE_LOAD_DIR[$stage]=""
        fi
        previous_online_tag="${STAGE_ONLINE_SAVE_TAG[$stage]}"
        previous_online_dir="${STAGE_ONLINE_SAVE_DIR[$stage]}"
    done
fi

PIPELINE_ORDER_TAG="$(join_by - "${NORMALIZED_STAGE_ORDER[@]}")"
if [[ "$PIPELINE_MODE" == "tarp" ]] && (( TARP_USE_ONLINE_TARP )); then
    DEFAULT_PIPELINE_NAME_CORE="${PIPELINE_ORDER_TAG}-online"
else
    DEFAULT_PIPELINE_NAME_CORE="$PIPELINE_ORDER_TAG"
fi
if [[ -n "$DEFAULT_PIPELINE_NAME_TAG" ]]; then
    DEFAULT_PIPELINE_NAME="${DEFAULT_PIPELINE_NAME_TAG}-${DEFAULT_PIPELINE_NAME_CORE}"
    DEFAULT_TARP_BOOTSTRAP_PIPELINE_NAME="${DEFAULT_PIPELINE_NAME_TAG}-${PIPELINE_ORDER_TAG}"
else
    DEFAULT_PIPELINE_NAME="$DEFAULT_PIPELINE_NAME_CORE"
    DEFAULT_TARP_BOOTSTRAP_PIPELINE_NAME="$PIPELINE_ORDER_TAG"
fi
PIPELINE_NAME="${PIPELINE_NAME:-$DEFAULT_PIPELINE_NAME}"

if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    if (( TARP_USE_ONLINE_TARP )); then
        DEFAULT_LOG_EXPERIMENT_PREFIX="$ONLINE_EXPERIMENT_PREFIX"
    else
        DEFAULT_LOG_EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}-TARP"
    fi
else
    DEFAULT_LOG_EXPERIMENT_PREFIX="$EXPERIMENT_PREFIX"
fi
LOG_EXPERIMENT_PREFIX="${LOG_EXPERIMENT_PREFIX:-$DEFAULT_LOG_EXPERIMENT_PREFIX}"
PPO_RUNTIME_BUCKET="${PPO_RUNTIME_BUCKET:-$(build_default_runtime_bucket "$PIPELINE_ORDER_TAG")}"

MERGE_BASE_ROOT="${MERGE_BASE_ROOT:-/path/to/outputs}"
DEFAULT_MERGE_ROOT="${MERGE_BASE_ROOT}/${PIPELINE_NAME}"
MERGE_ROOT_EXPLICIT=0
if [[ -n "${MERGE_ROOT+x}" ]]; then
    MERGE_ROOT_EXPLICIT=1
fi
MERGE_ROOT="${MERGE_ROOT:-$DEFAULT_MERGE_ROOT}"
if [[ "$PIPELINE_PRESET" == "qwen_3b_tarp" ]] && (( ! MERGE_ROOT_EXPLICIT )); then
    MERGE_ROOT="${MERGE_BASE_ROOT}/CD-KK-R1-online"
fi
if [[ "$PIPELINE_MODE" == "tarp" ]] && (( TARP_USE_ONLINE_TARP )); then
    TARP_BOOTSTRAP_FIRST_CONTINUAL_STAGE="${TARP_BOOTSTRAP_FIRST_CONTINUAL_STAGE:-1}"
    TARP_BOOTSTRAP_MERGE_ROOT="${TARP_BOOTSTRAP_MERGE_ROOT:-${MERGE_BASE_ROOT}/${DEFAULT_TARP_BOOTSTRAP_PIPELINE_NAME}}"
fi

LOG_BASE_ROOT="${LOG_BASE_ROOT:-$(tarp_runtime_log_dir "$PPO_RUNTIME_BUCKET")}"
DEFAULT_LOG_RUN_TAG="$(build_default_log_run_tag)"
LOG_RUN_TAG="${LOG_RUN_TAG:-$DEFAULT_LOG_RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-$LOG_BASE_ROOT/$PIPELINE_NAME/$LOG_RUN_TAG}"
HYDRA_RUN_BASE_DIR="${HYDRA_RUN_BASE_DIR:-$(tarp_runtime_hydra_dir "$PPO_RUNTIME_BUCKET")}"

if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    TARP_ONLINE_DIR="${TARP_ONLINE_DIR:-}"
    TARP_ONLINE_TAG="${TARP_ONLINE_TAG:-}"
fi

activate_conda
mkdir -p "$LOG_ROOT" "$MERGE_ROOT"
if (( MANAGE_HYDRA_RUN_DIR )); then
    mkdir -p "$HYDRA_RUN_BASE_DIR"
fi
if [[ "$PIPELINE_MODE" == "tarp" ]] && (( TARP_USE_ONLINE_TARP )); then
    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        if [[ -n "${STAGE_ONLINE_SAVE_DIR[$stage]:-}" ]]; then
            mkdir -p "${STAGE_ONLINE_SAVE_DIR[$stage]}"
        fi
    done
elif [[ "$PIPELINE_MODE" == "tarp" ]] && [[ -n "${TARP_ANCHOR_DIR:-}" ]]; then
    mkdir -p "$TARP_ANCHOR_DIR"
fi

if [[ "$PIPELINE_MODE" == "tarp" ]] && (( TARP_USE_ONLINE_TARP )) && is_true "$START_FRESH"; then
    existing_online_tags=()
    for stage in "${NORMALIZED_STAGE_ORDER[@]}"; do
        tag="${STAGE_ONLINE_SAVE_TAG[$stage]:-}"
        [[ -n "$tag" ]] || continue
        dir="${STAGE_ONLINE_SAVE_DIR[$stage]:-}"
        if [[ -n "$dir" ]] && find_online_prior_rank0_path "$tag" "$dir" >/dev/null; then
            existing_online_tags+=("$tag")
        fi
    done
    if (( ${#existing_online_tags[@]} > 0 )); then
        echo "Existing online TARP priors found under ${TARP_ANCHOR_ROOT}: $(join_by , "${existing_online_tags[@]}")." >&2
        echo "START_FRESH=1 would otherwise reuse those priors and contaminate the new run." >&2
        echo "Use a new TARP_ONLINE_DIR, a new explicit TARP_ONLINE_TAG, or remove the old priors manually." >&2
        exit 1
    fi
fi

echo "Pipeline root: $ROOT_DIR"
echo "Stage order: ${STAGE_ORDER[*]}"
echo "Initial model: $INITIAL_MODEL"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "N_GPUS: $N_GPUS"
echo "Model variant dir: $MODEL_VARIANT_DIR"
echo "Pipeline name: $PIPELINE_NAME"
echo "Start fresh: $START_FRESH"
echo "Resume mode: $TRAIN_RESUME_MODE"
echo "Merged model root: $MERGE_ROOT"
if (( PIPELINE_USE_LORA )); then
    echo "LoRA mode: enabled (rank=${PIPELINE_LORA_RANK}, merge_device=${LORA_MERGE_DEVICE}, merge_dtype=${LORA_MERGE_DTYPE})"
else
    echo "LoRA mode: disabled"
fi
echo "Logs: $LOG_ROOT"
if (( MANAGE_HYDRA_RUN_DIR )); then
    echo "Hydra root: $HYDRA_RUN_BASE_DIR"
fi
if [[ -n "${MODEL_TAG_TUNING_SIGNATURE:-}" ]]; then
    echo "Model tag tuning signature: $MODEL_TAG_TUNING_SIGNATURE"
fi
if [[ "$PIPELINE_MODE" == "tarp" ]]; then
    echo "Pipeline mode: TARP"
    echo "Pipeline TARP mode: $PIPELINE_TARP_MODE"
    echo "Pipeline TARP strategy: $PIPELINE_TARP_STRATEGY"
    if (( TARP_USE_ONLINE_TARP )); then
        echo "TARP online experiment prefix: $ONLINE_EXPERIMENT_PREFIX"
        echo "TARP anchor root: $TARP_ANCHOR_ROOT"
        if [[ -n "${TARP_ONLINE_TAG:-}" ]]; then
            echo "TARP online prior tag override: $TARP_ONLINE_TAG"
        else
            echo "TARP online prior tags: stage-specific cumulative names (for example: CD_TARP_..., CD-KK_TARP_...)"
        fi
        echo "TARP online decay: $TARP_ONLINE_DECAY"
        echo "TARP bootstrap first continual stage: $TARP_BOOTSTRAP_FIRST_CONTINUAL_STAGE"
        echo "TARP bootstrap merged root: $TARP_BOOTSTRAP_MERGE_ROOT"
    else
        echo "TARP anchor stage: $TARP_ANCHOR_STAGE"
        echo "TARP anchor dir: $TARP_ANCHOR_DIR"
        echo "TARP anchor tag: $TARP_ANCHOR_TAG"
    fi
else
    echo "Pipeline mode: baseline"
fi

current_base_model="$INITIAL_MODEL"
model_tag="$MODEL_TAG_PREFIX"
continual_stage_counter=0

for raw_stage in "${STAGE_ORDER[@]}"; do
    stage="${raw_stage^^}"
    validate_stage "$stage"
    stage_is_final=0
    if [[ "$stage" == "$FINAL_STAGE" ]]; then
        stage_is_final=1
    fi

    steps="${STAGE_STEPS[$stage]}"
    experiment_name="${STAGE_EXPERIMENT_NAME[$stage]}"
    stage_dir="${MODEL_ROOT}/${experiment_name}/${MODEL_VARIANT_DIR}"
    stage_model_tag_base="$model_tag"
    target_model_tag="${stage_model_tag_base}-${stage}${steps}"
    model_tag="$target_model_tag"
    checkpoint_actor_dir="${stage_dir}/global_step_${steps}/actor"
    tagged_model_name="$(decorate_model_tag "$target_model_tag")"
    legacy_merged_model_path="${MERGE_ROOT}/${target_model_tag}"
    merged_model_path="${MERGE_ROOT}/${tagged_model_name}"
    log_file="${LOG_ROOT}/$(sanitize_name "${tagged_model_name}__${experiment_name}").log"
    stage_hydra_run_dir=""
    if (( MANAGE_HYDRA_RUN_DIR )); then
        stage_hydra_run_dir="${HYDRA_RUN_BASE_DIR}/${PIPELINE_NAME}/${LOG_RUN_TAG}/$(sanitize_name "$experiment_name")"
    fi

    if [[ "$PIPELINE_MODE" == "tarp" ]]; then
        stage_tarp_train_mode="${STAGE_TARP_TRAIN_MODE[$stage]}"
        tarp_load_dataset_key="null"
        stage_tarp_load_dir="null"
        stage_tarp_load_tag="null"
        stage_tarp_dir="null"
        stage_tarp_tag="null"
        stage_online_tarp="False"
        stage_tarp_online_decay="$TARP_ONLINE_DECAY"
        stage_tarp_export_after_train="True"
        current_continual_stage_index=0

        if [[ "${stage_tarp_train_mode,,}" != "baseline" ]]; then
            continual_stage_counter=$((continual_stage_counter + 1))
            current_continual_stage_index=$continual_stage_counter
            if (( stage_is_final )); then
                stage_tarp_export_after_train="False"
            fi
            if (( TARP_USE_ONLINE_TARP )); then
                stage_tarp_dir="${STAGE_ONLINE_SAVE_DIR[$stage]}"
                stage_tarp_tag="${STAGE_ONLINE_SAVE_TAG[$stage]}"
                stage_tarp_load_tag="${STAGE_ONLINE_LOAD_TAG[$stage]:-null}"
                stage_tarp_load_dir="${STAGE_ONLINE_LOAD_DIR[$stage]:-null}"
                [[ -n "$stage_tarp_load_dir" ]] || stage_tarp_load_dir="null"
                stage_online_tarp="True"
            elif [[ "$stage" == "$TARP_ANCHOR_STAGE" ]]; then
                stage_tarp_dir="$TARP_ANCHOR_DIR"
                stage_tarp_tag="$TARP_ANCHOR_TAG"
            elif [[ "$stage" != "$TARP_ANCHOR_STAGE" ]]; then
                stage_tarp_load_dir="$TARP_ANCHOR_DIR"
                stage_tarp_load_tag="$TARP_ANCHOR_TAG"
            fi
        fi
    fi

    echo
    echo "============================================================"
    echo "Stage: $stage"
    echo "Experiment: $experiment_name"
    echo "Base model: $current_base_model"
    echo "Target step: $steps"
    echo "Stage model tag base: $stage_model_tag_base"
    if [[ "$PIPELINE_MODE" == "tarp" ]]; then
        echo "TARP train mode: $stage_tarp_train_mode"
        echo "TARP export after train: $stage_tarp_export_after_train"
        if (( TARP_USE_ONLINE_TARP )); then
            echo "TARP continual stage index: $current_continual_stage_index"
            echo "TARP online prior save: ${stage_tarp_dir}/${stage_tarp_tag}"
            if [[ "$stage_tarp_load_tag" == "null" ]]; then
                echo "TARP online prior load: none (initial online stage)"
            else
                echo "TARP online prior load: ${stage_tarp_load_dir}/${stage_tarp_load_tag}"
            fi
            echo "TARP load dataset key: $tarp_load_dataset_key (dataset-key lookup disabled; explicit prior is used)"
        else
            if [[ "$stage" == "$TARP_ANCHOR_STAGE" && "$stage_tarp_dir" != "null" ]]; then
                echo "TARP anchor export: ${stage_tarp_dir}/${stage_tarp_tag}"
            elif [[ "$stage_tarp_load_tag" != "null" ]]; then
                echo "TARP anchor prior load: ${stage_tarp_load_dir}/${stage_tarp_load_tag}"
            fi
            echo "TARP load dataset key: $tarp_load_dataset_key"
        fi
    fi
    echo "Val before train: ${STAGE_VAL_BEFORE_TRAIN[$stage]}"
    echo "Force answer stop (train): ${FORCE_ANSWER_STOP_TRAIN}"
    echo "Save/Test freq: ${STAGE_SAVE_FREQ[$stage]}/${STAGE_TEST_FREQ[$stage]}"
    echo "Allow existing stage output: $ALLOW_EXISTING_STAGE_OUTPUT"
    echo "Reuse existing target checkpoint: $REUSE_TARGET_CHECKPOINT"
    echo "Checkpoint dir: $checkpoint_actor_dir"
    echo "Merged output: $merged_model_path"
    echo "============================================================"

    reusable_merged_model_path=""
    if is_true "$SKIP_IF_MERGED_EXISTS"; then
        reusable_merged_model_path="$(find_reusable_merged_model_path "$merged_model_path" "$legacy_merged_model_path" || true)"
    fi
    if [[ -n "$reusable_merged_model_path" ]]; then
        if [[ "$reusable_merged_model_path" != "$merged_model_path" ]]; then
            echo "Found reusable bootstrap merged model for $stage: $reusable_merged_model_path"
        fi
        if [[ -d "$checkpoint_actor_dir" ]]; then
            process_stale_stage_checkpoints "$steps"
        fi
        if (( stage_is_final )) && [[ "$PIPELINE_MODE" == "tarp" ]] && [[ "${stage_tarp_train_mode,,}" != "baseline" ]]; then
            echo "Skipping TARP export-only bootstrap for final stage $stage; no later stage depends on its Fisher."
        else
            maybe_bootstrap_online_tarp_from_merged_model "$reusable_merged_model_path"
        fi
        echo "Merged model already exists. Skipping training and merge for $stage."
        current_base_model="$reusable_merged_model_path"
        continue
    fi

    stage_has_existing_outputs=0
    bootstrap_online_tarp_after_merge=0
    if compgen -G "${stage_dir}/global_step_*" >/dev/null; then
        stage_has_existing_outputs=1
    fi

    if (( stage_has_existing_outputs )) && [[ -d "$checkpoint_actor_dir" ]] && is_true "$REUSE_TARGET_CHECKPOINT"; then
        bootstrap_online_tarp_after_merge=1
        process_stale_stage_checkpoints "$steps"
        echo "Found target checkpoint. Reusing it directly for merge."
    else
        if (( stage_has_existing_outputs )) && ! is_true "$ALLOW_EXISTING_STAGE_OUTPUT"; then
            echo "Existing checkpoints found under $stage_dir." >&2
            echo "This run is currently configured not to reuse existing stage outputs." >&2
            echo "If you want to continue automatically, rerun with START_FRESH=0 or ALLOW_EXISTING_STAGE_OUTPUT=1." >&2
            echo "If you want a truly fresh run, keep START_FRESH=1 and use a new EXPERIMENT_PREFIX/MODEL_ROOT, or clean the old stage outputs yourself." >&2
            exit 1
        fi

        if is_true "$STOP_RAY_BEFORE_STAGE"; then
            ray stop --force >/dev/null 2>&1 || true
            sleep 1
        fi

        stage_args=(
            "trainer.val_before_train=${STAGE_VAL_BEFORE_TRAIN[$stage]}"
            "trainer.total_training_steps=$steps"
            "trainer.resume_mode=$TRAIN_RESUME_MODE"
            "actor_rollout_ref.rollout.force_answer_stop=${FORCE_ANSWER_STOP_TRAIN}"
        )
        if [[ "$PIPELINE_MODE" == "tarp" ]]; then
            stage_args+=(
                "actor_rollout_ref.actor.tarp_root=${TARP_ANCHOR_ROOT}"
                "actor_rollout_ref.actor.tarp_train_mode=${stage_tarp_train_mode}"
                "actor_rollout_ref.actor.tarp_load_dataset_key=${tarp_load_dataset_key}"
                "actor_rollout_ref.actor.tarp_load_dir=${stage_tarp_load_dir}"
                "actor_rollout_ref.actor.tarp_load_tag=${stage_tarp_load_tag}"
                "actor_rollout_ref.actor.tarp_dir=${stage_tarp_dir}"
                "actor_rollout_ref.actor.tarp_tag=${stage_tarp_tag}"
                "actor_rollout_ref.actor.online_tarp=${stage_online_tarp}"
                "actor_rollout_ref.actor.tarp_online_decay=${stage_tarp_online_decay}"
                "actor_rollout_ref.actor.tarp_export_after_train=${stage_tarp_export_after_train}"
                "actor_rollout_ref.actor.tarp_mode=${STAGE_TARP_MODE[$stage]}"
                "actor_rollout_ref.actor.tarp_lambda=${STAGE_TARP_LAMBDA[$stage]}"
            )
            append_tarp_reasoning_cache_args stage_args
        fi
        (
            if [[ -n "$stage_hydra_run_dir" ]]; then
                export HYDRA_RUN_DIR="$stage_hydra_run_dir"
            fi
            run_math_and_code "$log_file" "${stage_args[@]}" "${PIPELINE_EXTRA_ARGS[@]}"
        ) &
        training_pid=$!

        if ! watch_stage_checkpoints "$training_pid"; then
            echo "Intermediate checkpoint watcher failed for stage $stage. Stopping training." >&2
            kill "$training_pid" 2>/dev/null || true
            wait "$training_pid" 2>/dev/null || true
            exit 1
        fi

        if ! wait "$training_pid"; then
            echo "Stage training failed for $stage." >&2
            exit 1
        fi
    fi

    if [[ ! -d "$checkpoint_actor_dir" ]]; then
        echo "Expected checkpoint not found after training: $checkpoint_actor_dir" >&2
        exit 1
    fi

    process_stale_stage_checkpoints "$steps"

    run_stage_merge "$checkpoint_actor_dir" "$merged_model_path" "$current_base_model"
    if (( bootstrap_online_tarp_after_merge )); then
        if (( stage_is_final )) && [[ "$PIPELINE_MODE" == "tarp" ]] && [[ "${stage_tarp_train_mode,,}" != "baseline" ]]; then
            echo "Skipping TARP export-only bootstrap after final stage $stage; no later stage depends on its Fisher."
        else
            maybe_bootstrap_online_tarp_from_merged_model
        fi
    fi
    current_base_model="$merged_model_path"
done

echo
echo "Pipeline finished."
echo "Final merged model: $current_base_model"
