set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

tarp_args=(
    actor_rollout_ref.actor.tarp_train_mode=${TARP_TRAIN_MODE:-baseline}
    actor_rollout_ref.actor.tarp_load_dataset_key=${TARP_LOAD_DATASET_KEY:-null}
    actor_rollout_ref.actor.tarp_mode=${TARP_MODE:-TARP}
    actor_rollout_ref.actor.tarp_lambda=${TARP_LAMBDA:-0.0}
)

hydra_args=()
if [[ -n "${HYDRA_RUN_DIR:-}" ]]; then
    mkdir -p "${HYDRA_RUN_DIR}"
    hydra_args+=("hydra.run.dir=${HYDRA_RUN_DIR}")
fi

if [[ -n "${TARP_ROOT:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_root=${TARP_ROOT}")
fi
if [[ -n "${TARP_DATASET_KEY:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_dataset_key=${TARP_DATASET_KEY}")
fi
if [[ -n "${TARP_DIR:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_dir=${TARP_DIR}")
fi
if [[ -n "${TARP_TAG:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_tag=${TARP_TAG}")
fi
if [[ -n "${TARP_LOAD_DIR:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_load_dir=${TARP_LOAD_DIR}")
fi
if [[ -n "${TARP_LOAD_TAG:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_load_tag=${TARP_LOAD_TAG}")
fi
if [[ -n "${TARP_ONLINE_TARP:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.online_tarp=${TARP_ONLINE_TARP}")
fi
if [[ -n "${TARP_ONLINE_DECAY:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_online_decay=${TARP_ONLINE_DECAY}")
fi
if [[ -n "${TARP_EXPORT_AFTER_TRAIN:-}" ]]; then
    tarp_args+=("actor_rollout_ref.actor.tarp_export_after_train=${TARP_EXPORT_AFTER_TRAIN}")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[$TRAIN_DATA] \
    data.val_files=[$TEST_DATA] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=256 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    critic.model.path=$BASE_MODEL \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=$LOGGER \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.default_local_dir=$DIR \
    trainer.nnodes=1 \
    +trainer.val_only=$VAL_ONLY \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.total_epochs=$EPOCHS \
    reward_model.reward_manager=prime \
    "${hydra_args[@]}" \
    "${tarp_args[@]}" \
    "$@"
