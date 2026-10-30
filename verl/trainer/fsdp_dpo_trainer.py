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

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import math

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import DPODataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.workers.actor.ewc_utils import (
    DiagonalFisherEWC,
    build_ewc_state_meta,
    consolidate_ewc_state_dict,
    get_ewc_dataset_key,
    infer_primary_dataset_key,
    is_ewc_export_only,
    is_ewc_loss_enabled,
    resolve_ewc_load_candidates,
    resolve_ewc_mode,
    resolve_ewc_save_paths,
    validate_ewc_state_dict,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_DPO_LOGGING_LEVEL", "WARN"))


class FSDPDPOTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self.ewc_module = None
        self._ewc_state_loaded = False
        self._ewc_state_summary_printed = False

        if getattr(self.config, "ulysses_sequence_parallel_size", 1) != 1:
            raise ValueError("FSDP DPO trainer currently supports ulysses_sequence_parallel_size=1 only.")
        if getattr(self.config, "use_remove_padding", False):
            raise ValueError("FSDP DPO trainer currently supports use_remove_padding=False only.")

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_models_optimizer()

        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _maybe_enable_lora(self, model):
        if self.config.model.get("lora_rank", 0) <= 0:
            return model
        model.enable_input_require_grads()
        lora_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": self.config.model.lora_rank,
            "lora_alpha": self.config.model.lora_alpha,
            "target_modules": convert_to_regular_types(self.config.model.target_modules),
            "bias": "none",
        }
        return get_peft_model(model, LoraConfig(**lora_config))

    def _build_single_model(self, model_path, *, trainable: bool):
        local_model_path = copy_to_local(src=model_path, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32"))
        model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=model)

            if trainable:
                model = self._maybe_enable_lora(model)

        if trainable and self.config.model.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        for param in model.parameters():
            param.requires_grad_(trainable and param.requires_grad)

        return model, model_config

    def _wrap_fsdp(self, model):
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(
            model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            return FSDP(
                model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        if fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for fully_shard"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = model.state_dict()
            apply_fsdp2(model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(model, full_state, self.device_mesh, cpu_offload)
            return model
        raise NotImplementedError(f"not implement {fsdp_strategy}")

    def _build_models_optimizer(self):
        log_gpu_memory_usage("Before DPO model allocation", logger=logger)

        policy_path = self.config.model.partial_pretrain
        ref_path = self.config.ref_model.get("partial_pretrain", None) or policy_path

        self.policy_model, self.policy_model_config = self._build_single_model(policy_path, trainable=True)
        self.ref_model, _ = self._build_single_model(ref_path, trainable=False)
        self.ref_model.eval()

        log_gpu_memory_usage("After DPO model allocation", logger=logger)

        self.policy_fsdp_model = self._wrap_fsdp(self.policy_model)
        self.ref_fsdp_model = self._wrap_fsdp(self.ref_model)

        log_gpu_memory_usage("After DPO FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.policy_fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            self.total_steps = self.config.trainer.total_training_steps
        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, configured epochs {self.config.trainer.total_epochs}, total training steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _move_batch_to_device(self, batch):
        return {key: value.to(self.device_name) for key, value in batch.items()}

    def _micro_batches(self, batch):
        batch_size = batch["chosen_input_ids"].shape[0]
        micro_batch_size = self.config.data.micro_batch_size_per_gpu
        for start in range(0, batch_size, micro_batch_size):
            end = start + micro_batch_size
            yield {key: value[start:end] for key, value in batch.items()}

    def _sequence_logps(self, model, input_ids, attention_mask, position_ids, loss_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        shift_logits = output.logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_loss_mask = loss_mask[:, :-1].to(shift_logits.device)

        token_logps = F.log_softmax(shift_logits, dim=-1)
        token_logps = torch.gather(token_logps, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return (token_logps * shift_loss_mask).sum(dim=-1)

    def _compute_batch_loss(self, batch):
        batch = self._move_batch_to_device(batch)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            chosen_logps = self._sequence_logps(
                self.policy_fsdp_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_position_ids"],
                batch["chosen_loss_mask"],
            )
            rejected_logps = self._sequence_logps(
                self.policy_fsdp_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_position_ids"],
                batch["rejected_loss_mask"],
            )

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            ref_chosen_logps = self._sequence_logps(
                self.ref_fsdp_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_position_ids"],
                batch["chosen_loss_mask"],
            )
            ref_rejected_logps = self._sequence_logps(
                self.ref_fsdp_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_position_ids"],
                batch["rejected_loss_mask"],
            )

        beta = self.config.dpo.beta
        policy_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = policy_logratios - ref_logratios
        losses = -F.logsigmoid(beta * logits)
        loss = losses.mean()

        chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()

        metrics = {
            "loss": loss.detach(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "pair_accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        }
        return loss, metrics

    def _initialize_ewc_module(self):
        if self.ewc_module is not None:
            return
        self.ewc_module = DiagonalFisherEWC(
            model=self.policy_fsdp_model,
            store_dtype=getattr(self.config, "ewc_store_dtype", torch.bfloat16),
        )

    def _log_ewc_state_summary(self):
        if self.device_mesh.get_rank() != 0 or self.ewc_module is None or self._ewc_state_summary_printed:
            return
        self._ewc_state_summary_printed = True
        summary = self.ewc_module.state_magnitude_summary()
        print(
            "[DPO EWC State] "
            f"fisher_mean={summary.get('fisher_global_mean', 0.0):.6e} "
            f"fisher_max={summary.get('fisher_global_max', 0.0):.6e} "
            f"fisher_count={int(summary.get('fisher_count', 0))}",
            flush=True,
        )

    def _load_ewc_state_if_needed(self):
        if not is_ewc_loss_enabled(self.config):
            return
        if resolve_ewc_mode(self.config) != "DIAG_FISHER":
            raise ValueError(f"Unsupported DPO ewc_mode: {getattr(self.config, 'ewc_mode', None)}")

        self._initialize_ewc_module()
        if self._ewc_state_loaded:
            return

        rank = self.device_mesh.get_rank()
        world = self.device_mesh.size()
        train_files = self.config.data.train_files
        train_sources = train_files if isinstance(train_files, (list, tuple)) else [train_files]
        primary_source = next((item for item in train_sources if item), None)
        candidates = resolve_ewc_load_candidates(self.config, rank, world, data_source=primary_source)
        fpath = next((path for path in candidates if os.path.exists(path)), None)
        if fpath is None:
            raise FileNotFoundError(f"DPO EWC load target not found. candidates={candidates}")

        state_dict = torch.load(fpath, map_location="cpu")
        validate_ewc_state_dict(
            state_dict,
            self.ewc_module.param_names,
            expected_mode=resolve_ewc_mode(self.config),
        )
        self.ewc_module.load_state_dict(state_dict)
        self._ewc_state_loaded = True
        if rank == 0:
            print(f"[DPO EWC] loaded diagonal Fisher shard: {fpath}", flush=True)
        self._log_ewc_state_summary()

    def _distributed_mean(self, tensor):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor /= torch.distributed.get_world_size()
        return tensor

    @staticmethod
    def _metric_local_scalar(tensor):
        value = tensor.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value

    def training_step(self, batch):
        self.policy_fsdp_model.train()
        self.ref_fsdp_model.eval()
        self.optimizer.zero_grad()
        self._load_ewc_state_if_needed()

        micro_batches = list(self._micro_batches(batch))
        agg_metrics = None
        for micro_batch in micro_batches:
            loss, metrics = self._compute_batch_loss(micro_batch)
            (loss / len(micro_batches)).backward()
            if agg_metrics is None:
                agg_metrics = {key: value.detach() for key, value in metrics.items()}
            else:
                for key, value in metrics.items():
                    agg_metrics[key] += value.detach()

        for key in agg_metrics:
            agg_metrics[key] /= len(micro_batches)

        ewc_penalty = torch.zeros((), device=self.device_name, dtype=torch.float32)
        if is_ewc_loss_enabled(self.config):
            ewc_penalty = self.ewc_module.penalty(coef=float(getattr(self.config, "ewc_lambda", 0.0)))
            ewc_penalty.backward()
        agg_metrics["ewc_penalty"] = self._metric_local_scalar(ewc_penalty)

        if self.config.model.strategy == "fsdp":
            grad_norm = self.policy_fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.policy_fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()

        metric_tensor = torch.stack(
            [
                agg_metrics["loss"],
                agg_metrics["reward_margin"],
                agg_metrics["chosen_reward"],
                agg_metrics["rejected_reward"],
                agg_metrics["pair_accuracy"],
                agg_metrics["ewc_penalty"],
            ]
        ).to(self.device_name)
        metric_tensor = self._distributed_mean(metric_tensor)
        lr = torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device_name)
        lr = self._distributed_mean(lr)

        return {
            "train/loss": metric_tensor[0].item(),
            "train/reward_margin": metric_tensor[1].item(),
            "train/chosen_reward": metric_tensor[2].item(),
            "train/rejected_reward": metric_tensor[3].item(),
            "train/pair_accuracy": metric_tensor[4].item(),
            "train/ewc_penalty": metric_tensor[5].item(),
            "train/lr(1e-6)": lr.item() * 1e6,
        }

    def validation_step(self, batch):
        self.policy_fsdp_model.eval()
        self.ref_fsdp_model.eval()
        with torch.no_grad():
            loss, metrics = self._compute_batch_loss(batch)

        metric_tensor = torch.stack(
            [
                metrics["loss"],
                metrics["reward_margin"],
                metrics["chosen_reward"],
                metrics["rejected_reward"],
                metrics["pair_accuracy"],
                torch.zeros((), device=self.device_name, dtype=metrics["loss"].dtype),
            ]
        ).to(self.device_name)
        metric_tensor = self._distributed_mean(metric_tensor)
        return metric_tensor

    def export_ewc_fisher(self):
        rank = self.device_mesh.get_rank()
        world = self.device_mesh.size()
        if resolve_ewc_mode(self.config) != "DIAG_FISHER":
            raise ValueError(f"Unsupported DPO ewc_mode for export: {getattr(self.config, 'ewc_mode', None)}")

        self.policy_fsdp_model.eval()
        self.ref_fsdp_model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        self._initialize_ewc_module()
        self.ewc_module.snapshot_theta_star()

        target_steps = int(getattr(self.config, "ewc_target_steps", 64))
        if target_steps <= 0:
            raise ValueError(f"ewc_target_steps must be > 0, got {target_steps}")

        total_collected = 0
        export_epochs = max(1, math.ceil(target_steps / max(self.steps_per_epoch, 1)))
        if rank == 0:
            print(f"[DPO EWC] exporting diagonal Fisher over {target_steps} micro-batches", flush=True)

        for epoch in range(export_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"DPO EWC Export {epoch + 1}/{export_epochs}",
                disable=rank != 0,
            ):
                micro_batches = list(self._micro_batches(data))
                for micro_batch in micro_batches:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._compute_batch_loss(micro_batch)
                    if torch.isfinite(loss):
                        loss.backward()
                        self.ewc_module.accumulate_fisher_after_backward()
                        total_collected += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_collected >= target_steps:
                        break
                if total_collected >= target_steps:
                    break
            if total_collected >= target_steps:
                break

        if total_collected <= 0:
            raise RuntimeError("DPO EWC export finished without any valid accumulated steps.")

        self.ewc_module.finalize_fisher()

        train_files = self.config.data.train_files
        train_sources = train_files if isinstance(train_files, (list, tuple)) else [train_files]
        dataset_key = (
            get_ewc_dataset_key(self.config, data_source=next((item for item in train_sources if item), None))
            or infer_primary_dataset_key(train_sources)
            or "UNKNOWN"
        )
        primary_source = next((item for item in train_sources if item), None)
        part_path, merged_path = resolve_ewc_save_paths(self.config, dataset_key, rank, world, total_collected)
        state = self.ewc_module.state_dict()
        state["_meta"] = build_ewc_state_meta(
            dataset_key=dataset_key,
            data_source=primary_source,
            config=self.config,
            rank=rank,
            world_size=world,
            valid_samples_used=total_collected,
        )
        state["_meta"]["experiment_name"] = str(getattr(self.config.trainer, "experiment_name", ""))

        if bool(getattr(self.config, "ewc_online", False)):
            prior_candidates = resolve_ewc_load_candidates(self.config, rank, world, data_source=primary_source)
            prior_path = next((path for path in prior_candidates if os.path.exists(path)), None)
            if prior_path is None:
                raise FileNotFoundError(f"DPO online EWC prior not found. candidates={prior_candidates}")
            prior_state = torch.load(prior_path, map_location="cpu")
            validate_ewc_state_dict(
                prior_state,
                self.ewc_module.param_names,
                expected_mode=resolve_ewc_mode(self.config),
            )
            state = consolidate_ewc_state_dict(
                prior_state,
                state,
                decay=float(getattr(self.config, "ewc_online_decay", 1.0)),
            )
            if rank == 0:
                print(f"[DPO EWC] online consolidated prior: {prior_path}", flush=True)

        os.makedirs(os.path.dirname(part_path), exist_ok=True)
        torch.save(state, part_path)
        torch.save(state, merged_path)
        torch.distributed.barrier()
        if rank == 0:
            print(f"[DPO EWC] saved part shard: {part_path}", flush=True)
            print(f"[DPO EWC] saved merged shard: {merged_path}", flush=True)

    def save_checkpoint(self, step):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.policy_fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.policy_fsdp_model.state_dict()

            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.policy_model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.policy_fsdp_model, options=options)
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.policy_model.save_pretrained(path, state_dict=state_dict)
                self.policy_model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        if is_ewc_export_only(self.config):
            self.export_ewc_fisher()
            return
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        self.total_training_steps = self.total_steps
        effective_total_epochs = max(self.config.trainer.total_epochs, math.ceil(self.total_training_steps / self.steps_per_epoch))
        if rank == 0:
            print(f"Total training steps: {self.total_training_steps}")
            print(f"Effective epochs: {effective_total_epochs}")

        for epoch in range(effective_total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader, total=self.steps_per_epoch, desc=f"Epoch {epoch + 1}/{effective_total_epochs}", disable=rank != 0):
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                should_run_validation = self.config.trainer.test_freq > 0 and (is_last_step or is_valid_step)
                if should_run_validation:
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        val_metrics.append(self.validation_step(val_data))
                    if rank == 0:
                        metric_tensor = torch.mean(torch.stack(val_metrics), dim=0)
                        metric = {
                            "val/loss": metric_tensor[0].item(),
                            "val/reward_margin": metric_tensor[1].item(),
                            "val/chosen_reward": metric_tensor[2].item(),
                            "val/rejected_reward": metric_tensor[3].item(),
                            "val/pair_accuracy": metric_tensor[4].item(),
                            "val/ewc_penalty": metric_tensor[5].item(),
                        }
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_dpo(config):
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_dpo_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_dpo_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPDPOTrainer(config=config, device_mesh=device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


def create_dpo_dataset(data_paths, data_config, tokenizer):
    return DPODataset(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)


if __name__ == "__main__":
    main()
