# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import hydra
import ray
from omegaconf import open_dict

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.workers.actor.tarp_utils import (
    build_anchor_storage_dir,
    build_online_tarp_tag,
    extend_stage_sequence,
    infer_primary_dataset_key,
    infer_model_size_from_path,
    infer_stage_sequence_from_path,
    is_none_like,
    is_online_tarp_enabled,
    resolve_tarp_train_mode,
)
from verl.workers.actor.ewc_utils import is_ewc_export_only


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def _configure_online_tarp_identity(config, dataset_key: str | None) -> None:
    actor_cfg = config.actor_rollout_ref.actor
    if not is_online_tarp_enabled(actor_cfg):
        return

    current_stage = dataset_key or infer_primary_dataset_key(config.data.train_files)
    if current_stage is None:
        return

    base_model_path = config.actor_rollout_ref.model.get("path", None)
    stage_history = extend_stage_sequence(infer_stage_sequence_from_path(base_model_path), current_stage)
    if not stage_history:
        return

    save_dir_cfg = actor_cfg.get("tarp_dir", None)
    load_dir_cfg = actor_cfg.get("tarp_load_dir", None)
    save_tag_cfg = actor_cfg.get("tarp_tag", None)
    load_tag_cfg = actor_cfg.get("tarp_load_tag", None)

    tarp_cap = actor_cfg.get("tarp_to_pg_ratio_cap", 0)
    tarp_lambda = actor_cfg.get("tarp_lambda", 0.0)
    save_tag = build_online_tarp_tag(stage_history, tarp_cap, tarp_lambda)
    load_tag = build_online_tarp_tag(stage_history[:-1], tarp_cap, tarp_lambda) if len(stage_history) > 1 else None
    tarp_root = actor_cfg.get("tarp_root", "/path/to/outputs/anchor")
    model_size = actor_cfg.get("tarp_model_size", infer_model_size_from_path(base_model_path))

    with open_dict(actor_cfg):
        if is_none_like(save_dir_cfg) or str(save_dir_cfg).strip().lower() == "auto":
            if model_size is None:
                actor_cfg.tarp_dir = tarp_root
            else:
                actor_cfg.tarp_dir = build_anchor_storage_dir(
                    root=tarp_root,
                    model_size=model_size,
                    stage_sequence=stage_history,
                    cap_value=tarp_cap,
                    lam_value=tarp_lambda,
                )
        if is_none_like(save_tag_cfg) or str(save_tag_cfg).strip().lower() == "auto":
            actor_cfg.tarp_tag = save_tag
        if load_tag is None:
            if is_none_like(load_dir_cfg) or str(load_dir_cfg).strip().lower() == "auto":
                actor_cfg.tarp_load_dir = None
            if is_none_like(load_tag_cfg) or str(load_tag_cfg).strip().lower() == "auto":
                actor_cfg.tarp_load_tag = None
        else:
            if is_none_like(load_dir_cfg) or str(load_dir_cfg).strip().lower() == "auto":
                if model_size is None:
                    actor_cfg.tarp_load_dir = actor_cfg.tarp_dir
                else:
                    actor_cfg.tarp_load_dir = build_anchor_storage_dir(
                        root=tarp_root,
                        model_size=model_size,
                        stage_sequence=stage_history[:-1],
                        cap_value=tarp_cap,
                        lam_value=tarp_lambda,
                    )
            if is_none_like(load_tag_cfg) or str(load_tag_cfg).strip().lower() == "auto":
                actor_cfg.tarp_load_tag = load_tag

    print(
        "Resolved online TARP identity: "
        f"save={actor_cfg.get('tarp_dir')}/{actor_cfg.get('tarp_tag')} "
        f"load={actor_cfg.get('tarp_load_dir')}/{actor_cfg.get('tarp_load_tag')}",
        flush=True,
    )


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        # Prefer a temp dir on a spacious filesystem. If Ray defaults to /tmp on a nearly-full
        # root partition, object spilling can fail.
        temp_dir = os.environ.get("RAY_TMPDIR") or config.ray_init.get("temp_dir", None)

        init_kwargs = dict(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )
        if temp_dir:
            init_kwargs["_temp_dir"] = temp_dir
        ray.init(**init_kwargs)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    # create a timeline trace file to analyze the performance
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        dataset_key = infer_primary_dataset_key(train_dataset.dataframe.get("data_source", []))
        with open_dict(config.actor_rollout_ref.actor):
            trainer_tarp_export_only = bool(config.trainer.get("tarp_export_only", False))
            trainer_ewc_export_only = bool(config.trainer.get("ewc_export_only", False))
            if trainer_tarp_export_only and trainer_ewc_export_only:
                raise ValueError("trainer.tarp_export_only and trainer.ewc_export_only cannot both be True.")
            if trainer_tarp_export_only:
                config.actor_rollout_ref.actor.tarp_export_only = True
            if trainer_ewc_export_only:
                config.actor_rollout_ref.actor.ewc_export_only = True
            if not config.actor_rollout_ref.actor.get("tarp_dataset_key"):
                config.actor_rollout_ref.actor.tarp_dataset_key = dataset_key
            if not config.actor_rollout_ref.actor.get("tarp_root"):
                config.actor_rollout_ref.actor.tarp_root = "/path/to/outputs/anchor"
            if not config.actor_rollout_ref.actor.get("tarp_model_size"):
                config.actor_rollout_ref.actor.tarp_model_size = infer_model_size_from_path(config.actor_rollout_ref.model.path)
            if not config.actor_rollout_ref.actor.get("tarp_experiment_name"):
                config.actor_rollout_ref.actor.tarp_experiment_name = config.trainer.get("experiment_name", "")
            if not config.actor_rollout_ref.actor.get("ewc_dataset_key"):
                config.actor_rollout_ref.actor.ewc_dataset_key = dataset_key
            if not config.actor_rollout_ref.actor.get("ewc_experiment_name"):
                config.actor_rollout_ref.actor.ewc_experiment_name = config.trainer.get("experiment_name", "")
            if not config.actor_rollout_ref.actor.get("ewc_source_model_path"):
                config.actor_rollout_ref.actor.ewc_source_model_path = config.actor_rollout_ref.model.path
        _configure_online_tarp_identity(config, dataset_key)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        tarp_export_only = bool(config.trainer.get("tarp_export_only", False))
        ewc_export_only = bool(config.trainer.get("ewc_export_only", False)) or is_ewc_export_only(config.actor_rollout_ref.actor)
        if tarp_export_only and ewc_export_only:
            raise ValueError("TARP export-only mode and EWC export-only mode cannot be enabled together.")
        if tarp_export_only:
            if resolve_tarp_train_mode(config.actor_rollout_ref.actor) != "continual":
                raise ValueError("TARP export-only mode requires actor.tarp_train_mode=continual")
            print("TARP export-only mode: using the requested actor weights and updating TARP artifacts without PPO training...")
            trainer.get_Fisher_diagonal()
            return
        if ewc_export_only:
            print("EWC export-only mode: using the requested actor weights and exporting a diagonal Fisher prior without PPO training...")
            trainer.get_Fisher_diagonal()
            return
        trainer.fit()
        tarp_train_mode = resolve_tarp_train_mode(config.actor_rollout_ref.actor)
        online_tarp = is_online_tarp_enabled(config.actor_rollout_ref.actor)
        tarp_load_dataset_key = config.actor_rollout_ref.actor.get("tarp_load_dataset_key", None)
        raw_export_after_train = config.actor_rollout_ref.actor.get("tarp_export_after_train", None)
        if raw_export_after_train is None:
            should_export_tarp_after_train = online_tarp or tarp_load_dataset_key in (None, "", "null", "None")
        else:
            should_export_tarp_after_train = str(raw_export_after_train).strip().lower() in {"1", "true", "yes", "y", "on"}
        if (
            tarp_train_mode == "continual"
            and should_export_tarp_after_train
            and not config.trainer.get("val_only", False)
        ):
            if online_tarp:
                print("Online TARP mode: exporting Fisher and updating the consolidated prior...")
            else:
                print("Continual TARP mode: anchor task finished, exporting Fisher for the current task...")
            trainer.get_Fisher_diagonal()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")
    print("#"*100)
    print("data_config", data_config)
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    print(dataset)
    return dataset
    """
    (TaskRunner pid=750831) dataframe        target  ...                                               meta
    (TaskRunner pid=750831) 0        98.0  ...                                                NaN
    (TaskRunner pid=750831) 1        64.0  ...                                                NaN
    (TaskRunner pid=750831) 2        28.0  ...                                                NaN
    (TaskRunner pid=750831) 3        48.0  ...                                                NaN
    (TaskRunner pid=750831) 4        17.0  ...                                                NaN
    (TaskRunner pid=750831) ...       ...  ...                                                ...
    (TaskRunner pid=750831) 12450     NaN  ...  {'question_id': 1064.0, 'difficulty': 'Medium'...
    (TaskRunner pid=750831) 12451     NaN  ...                                               None
    (TaskRunner pid=750831) 12452     NaN  ...                                               None
    (TaskRunner pid=750831) 12453     NaN  ...                                               None
    (TaskRunner pid=750831) 12455     NaN  ...                                               None
    (TaskRunner pid=750831)
    (TaskRunner pid=750831) [26703 rows x 14 columns]
    (TaskRunner pid=750831) filter dataset len: 26703
    (TaskRunner pid=750831) dict_keys(['target', 'nums', 'data_source', 'ability', 'reward_model', 'extra_info',
    'task_id', 'entry_point', 'test', 'completion', 'examples', 'src', 'meta', 'input_ids', 'attention_mask', 'position_ids', 'index'])

    (TaskRunner pid=750831)      target             nums data_source  ... examples   src  meta
    (TaskRunner pid=750831) 0      45.0  [30, 71, 88, 2]   countdown  ...      NaN   NaN   NaN
    (TaskRunner pid=750831) 1      33.0      [9, 37, 79]   countdown  ...      NaN   NaN   NaN
    (TaskRunner pid=750831) 2      20.0      [9, 49, 78]   countdown  ...      NaN   NaN   NaN
    (TaskRunner pid=750831) 3      62.0     [77, 29, 44]   countdown  ...      NaN   NaN   NaN
    (TaskRunner pid=750831) 4      67.0     [45, 35, 57]   countdown  ...      NaN   NaN   NaN
    (TaskRunner pid=750831) ..      ...              ...         ...  ...      ...   ...   ...
    (TaskRunner pid=750831) 700     NaN              NaN        code  ...     None  None  None
    (TaskRunner pid=750831) 701     NaN              NaN        code  ...     None  None  None
    (TaskRunner pid=750831) 703     NaN              NaN        code  ...     None  None  None
    (TaskRunner pid=750831) 709     NaN              NaN        code  ...     None  None  None
    (TaskRunner pid=750831) 710     NaN              NaN        code  ...     None  None  None
    (TaskRunner pid=750831)
    (TaskRunner pid=750831) [1976 rows x 14 columns]
    (TaskRunner pid=750831) filter dataset len: 1976
    (TaskRunner pid=750831) dict_keys(['target', 'nums', 'data_source', 'ability', 'reward_model', 'extra_info',
    'task_id', 'entry_point', 'test', 'completion', 'examples', 'src', 'meta', 'input_ids', 'attention_mask', 'position_ids', 'index'])
    """

def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
