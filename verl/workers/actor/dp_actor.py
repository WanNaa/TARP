# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import contextlib
import itertools
import logging
import os
import re
import gc
import math
from typing import Tuple, Dict, List, Optional, Sequence, Any

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tensordict import TensorDict
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.trainer.ppo.reward import compute_reward

# =============================================================================
# [NEW] Import TARP Utils
# NOTE: 对应实现在 `verl/workers/actor/tarp_utils.py`
# =============================================================================
try:
    from verl.workers.actor.tarp_utils import (
        BlockExactSubspaceTARP,
        extract_between,
        split_reason_sentences,
        encode_sentence_embeds,
        semantic_consistency,
        make_exclude_ids,
        find_subseq,
        trim_end_by_pad,
        compute_tarp_line_weight_profile,
        build_tarp_token_weight_from_lines,
        build_tarp_state_meta,
        consolidate_tarp_state_dict,
        get_tarp_online_decay,
        get_tarp_dataset_key,
        infer_primary_dataset_key,
        is_online_tarp_enabled,
        resolve_tarp_load_candidates,
        resolve_tarp_part_id,
        resolve_tarp_train_mode,
        resolve_tarp_save_paths,
        validate_tarp_state_dict,
    )
except ImportError as e:
    # 如果导入失败，TARP 相关功能会失效，但不影响基础 PPO 训练
    print(f"[DP Actor] Warning: failed to import TARP utils: {e}. TARP features may fail.")

try:
    from verl.workers.actor.ewc_utils import (
        DiagonalFisherEWC,
        _param_name_aliases,
        build_ewc_state_meta,
        get_ewc_dataset_key,
        infer_primary_dataset_key as infer_primary_ewc_dataset_key,
        is_ewc_export_only,
        is_ewc_loss_enabled,
        resolve_ewc_load_candidates,
        resolve_ewc_mode,
        resolve_ewc_save_paths,
        validate_ewc_state_dict,
    )
except ImportError as e:
    print(f"[DP Actor] Warning: failed to import EWC utils: {e}. EWC features may fail.")

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EndAnswerCriteria(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids

    def _ends_with_stop(self, sequence_ids):
        Ls = len(self.stop_ids)
        if len(sequence_ids) < Ls:
            return False
        return sequence_ids[-Ls:] == self.stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in input_ids:
            if self._ends_with_stop(seq.tolist()):
                return True
        return False


class PerSequenceAnswerStop(LogitsProcessor):
    def __init__(self, stop_ids, eos_token_id: Optional[int], pad_token_id: Optional[int]):
        self.stop_ids = list(stop_ids or [])
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.force_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        if self.force_token_id is None:
            raise ValueError("PerSequenceAnswerStop requires either eos_token_id or pad_token_id.")

    def _ends_with_stop(self, sequence_ids: list[int]) -> bool:
        if not self.stop_ids:
            return False
        stop_len = len(self.stop_ids)
        if len(sequence_ids) < stop_len:
            return False
        return sequence_ids[-stop_len:] == self.stop_ids

    def _is_finished(self, sequence_ids: list[int]) -> bool:
        if not sequence_ids:
            return False
        if self.eos_token_id is not None and sequence_ids[-1] == self.eos_token_id:
            return True
        if self.pad_token_id is not None and sequence_ids[-1] == self.pad_token_id:
            return True
        return self._ends_with_stop(sequence_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        forced = scores
        for row_idx, seq in enumerate(input_ids):
            if not self._is_finished(seq.tolist()):
                continue
            forced[row_idx].fill_(torch.finfo(forced.dtype).min)
            forced[row_idx, self.force_token_id] = 0
        return forced


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

        # [TARP Init]
        self.tarp_module = None
        self._tarp_param_names = None
        self._tarp_shard_param_infos = None
        self._tarp_update_step = 0
        self._tarp_penalty_debug_calls = 0
        self._tarp_state_summary_printed = set()
        self.ewc_module = None
        self._ewc_param_names = None
        self._ewc_full_shapes = None
        self._ewc_shard_param_infos = None
        self._ewc_state_loaded = False
        self._ewc_state_summary_printed = set()
        self._ewc_local_theta_star = None
        self._ewc_penalty_debug_calls = 0

    def _use_orig_tarp_param_view(self) -> bool:
        return str(getattr(self.config, "tarp_param_view", "orig")).strip().lower() == "orig"

    def _tarp_full_param_ctx(self, with_grads: bool = False):
        if isinstance(self.actor_module, FSDP) and self._use_orig_tarp_param_view():
            return FSDP.summon_full_params(
                self.actor_module,
                writeback=False,
                recurse=True,
                with_grads=with_grads,
            )
        return contextlib.nullcontext()

    def _initialize_tarp_module(self):
        if self.tarp_module is not None:
            return

        with self._tarp_full_param_ctx():
            self.tarp_module = BlockExactSubspaceTARP(
                model=self.actor_module,
                rank_per_block=getattr(self.config, "tarp_rank_per_block", 32),
                block_size=getattr(self.config, "tarp_block_size", 2048),
                proj_dim=getattr(self.config, "tarp_proj_dim", 48),
                store_dtype=getattr(self.config, "tarp_store_dtype", torch.bfloat16),
                normalize_proj_columns=bool(getattr(self.config, "tarp_normalize_proj_columns", True)),
                average_over_proj_dim=bool(getattr(self.config, "tarp_average_over_proj_dim", True)),
            )
            self._tarp_param_names = [meta.name for meta in self.tarp_module.mat_metas] + [meta.name for meta in self.tarp_module.vec_metas]
        self._tarp_shard_param_infos = self._collect_tarp_shard_param_infos()
        self._tarp_state_loaded = False

    def _tarp_debug_enabled(self) -> bool:
        return bool(getattr(self.config, "tarp_penalty_debug", True))

    def _tarp_state_debug_on_load(self) -> bool:
        return bool(getattr(self.config, "tarp_state_debug_on_load", True))

    def _tarp_debug_interval(self) -> int:
        try:
            return max(1, int(getattr(self.config, "tarp_penalty_debug_interval", 1)))
        except Exception:
            return 1

    def _tarp_debug_top_layers(self) -> int:
        try:
            return max(1, int(getattr(self.config, "tarp_penalty_debug_top_layers", 3)))
        except Exception:
            return 3

    def _tarp_debug_delta_layers(self) -> int:
        try:
            return max(1, int(getattr(self.config, "tarp_penalty_debug_delta_layers", self._tarp_debug_top_layers())))
        except Exception:
            return self._tarp_debug_top_layers()

    @staticmethod
    def _top_terms(term_map: Dict[str, float], topk: int) -> List[Tuple[str, float]]:
        if not term_map:
            return []
        return sorted(term_map.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:topk]

    def _log_tarp_state_magnitude_summary(self, rank: int) -> None:
        if rank != 0 or not self._tarp_state_debug_on_load():
            return
        if self.tarp_module is None:
            return
        if rank in self._tarp_state_summary_printed:
            return
        self._tarp_state_summary_printed.add(rank)

        topk = self._tarp_debug_top_layers()
        summary = self.tarp_module.state_magnitude_summary()
        fisher_layers = summary.get("fisher_layer_stats", {})
        lam_layers = summary.get("lam_layer_stats", {})
        fisher_top = sorted(fisher_layers.items(), key=lambda kv: kv[1].get("max", 0.0), reverse=True)[:topk]
        lam_top = sorted(lam_layers.items(), key=lambda kv: kv[1].get("max", 0.0), reverse=True)[:topk]

        print(
            "[TARP State] "
            f"fisher_mean={summary.get('fisher_global_mean', 0.0):.6e} "
            f"fisher_max={summary.get('fisher_global_max', 0.0):.6e} "
            f"fisher_count={int(summary.get('fisher_count', 0))} "
            f"lam_mean={summary.get('lam_global_mean', 0.0):.6e} "
            f"lam_max={summary.get('lam_global_max', 0.0):.6e} "
            f"lam_count={int(summary.get('lam_count', 0))}",
            flush=True,
        )
        if fisher_top:
            fisher_str = ", ".join(
                f"{name}(mean={stat.get('mean', 0.0):.3e},max={stat.get('max', 0.0):.3e})" for name, stat in fisher_top
            )
            print(f"[TARP State] top_fisher_layers: {fisher_str}", flush=True)
        if lam_top:
            lam_str = ", ".join(
                f"{name}(mean={stat.get('mean', 0.0):.3e},max={stat.get('max', 0.0):.3e})" for name, stat in lam_top
            )
            print(f"[TARP State] top_lam_layers: {lam_str}", flush=True)

    def _log_tarp_penalty_breakdown(self, breakdown: Dict[str, Any], rank: int) -> None:
        if rank != 0 or not breakdown:
            return

        topk = self._tarp_debug_top_layers()
        delta_topk = self._tarp_debug_delta_layers()

        vector_total = float(breakdown.get("vector_total", 0.0))
        matrix_total = float(breakdown.get("matrix_total", 0.0))
        total = float(breakdown.get("total", 0.0))
        tiny = 1e-10
        if abs(vector_total) < tiny and abs(matrix_total) < tiny:
            diagnosis = "both_small"
        elif abs(vector_total) < 0.1 * max(abs(matrix_total), tiny):
            diagnosis = "vector_small_matrix_dominant"
        elif abs(matrix_total) < 0.1 * max(abs(vector_total), tiny):
            diagnosis = "matrix_small_vector_dominant"
        else:
            diagnosis = "both_nontrivial"

        print(
            "[TARP Breakdown] "
            f"call={self._tarp_penalty_debug_calls} "
            f"total={total:.6e} "
            f"vector_total={vector_total:.6e} "
            f"matrix_total={matrix_total:.6e} "
            f"diagnosis={diagnosis}",
            flush=True,
        )

        vector_terms = breakdown.get("vector_layer_terms", {}) or {}
        matrix_terms = breakdown.get("matrix_layer_terms", {}) or {}
        vector_top = self._top_terms(vector_terms, topk=topk)
        matrix_top = self._top_terms(matrix_terms, topk=topk)
        if vector_top:
            vector_str = ", ".join(f"{name}={value:.3e}" for name, value in vector_top)
            print(f"[TARP Breakdown] top_vector_terms: {vector_str}", flush=True)
        if matrix_top:
            matrix_str = ", ".join(f"{name}={value:.3e}" for name, value in matrix_top)
            print(f"[TARP Breakdown] top_matrix_terms: {matrix_str}", flush=True)

        delta_stats = breakdown.get("delta_stats", {}) or {}
        combined_terms = {**vector_terms, **matrix_terms}
        reps = self._top_terms(combined_terms, topk=delta_topk)
        if reps:
            delta_parts = []
            for layer_name, _ in reps:
                stat = delta_stats.get(layer_name, {})
                delta_parts.append(
                    f"{layer_name}("
                    f"group={stat.get('group', '?')},"
                    f"l2={float(stat.get('l2', 0.0)):.3e},"
                    f"mean_abs={float(stat.get('mean_abs', 0.0)):.3e},"
                    f"max_abs={float(stat.get('max_abs', 0.0)):.3e})"
                )
            if delta_parts:
                print(f"[TARP Delta] {'; '.join(delta_parts)}", flush=True)

    def _collect_tarp_shard_param_infos(self) -> Dict[str, object]:
        if not isinstance(self.actor_module, FSDP):
            return {}

        shard_infos: Dict[str, object] = {}
        for module in self.actor_module.modules():
            if not isinstance(module, FSDP):
                continue
            flat_param = getattr(module, "_flat_param", None)
            fqns = getattr(flat_param, "_fqns", None) if flat_param is not None else None
            info_list = getattr(flat_param, "_shard_param_infos", None) if flat_param is not None else None
            if not fqns or not info_list:
                continue
            for fqn, info in zip(fqns, info_list):
                if fqn is not None:
                    shard_infos.setdefault(fqn, info)
        return shard_infos

    def _slice_tarp_grad_for_current_param(self, name: str, param: torch.nn.Parameter, full_grad: torch.Tensor) -> Optional[torch.Tensor]:
        if tuple(param.shape) == tuple(full_grad.shape):
            return full_grad.reshape(param.shape)
        if param.numel() == full_grad.numel():
            return full_grad.reshape(param.shape)
        if param.numel() == 0:
            return None

        shard_info = (self._tarp_shard_param_infos or {}).get(name)
        if shard_info is None or not getattr(shard_info, "in_shard", False):
            return None

        start = getattr(shard_info, "intra_param_start_idx", None)
        end = getattr(shard_info, "intra_param_end_idx", None)
        if start is None or end is None or end < start:
            return None

        local_grad = full_grad.reshape(-1)[start:end + 1]
        if local_grad.numel() != param.numel():
            raise RuntimeError(
                f"TARP local grad slice mismatch for {name}: "
                f"slice_numel={local_grad.numel()} param_numel={param.numel()} "
                f"param_shape={tuple(param.shape)} full_grad_shape={tuple(full_grad.shape)}"
            )
        return local_grad.reshape(param.shape)

    def _apply_tarp_gradients(self, tarp_grads: Dict[str, torch.Tensor], scale: float) -> None:
        if not tarp_grads or scale == 0.0:
            return

        param_map = self.tarp_module._current_named_params()
        for name, full_grad in tarp_grads.items():
            param = self.tarp_module._current_param(name, param_map=param_map)
            local_grad = self._slice_tarp_grad_for_current_param(name, param, full_grad)
            if local_grad is None:
                continue

            target_dtype = param.grad.dtype if param.grad is not None else param.dtype
            local_grad = local_grad.to(device=param.device, dtype=target_dtype) * scale
            if param.grad is None:
                param.grad = local_grad.clone()
            else:
                if tuple(param.grad.shape) != tuple(local_grad.shape):
                    raise RuntimeError(
                        f"TARP grad add shape mismatch for {name}: "
                        f"existing_grad={tuple(param.grad.shape)} new_grad={tuple(local_grad.shape)}"
                    )
                param.grad.add_(local_grad)

    def _collect_tarp_local_targets(self) -> Dict[str, Dict[str, Any]]:
        if self.tarp_module is None or not self._tarp_param_names:
            return {}

        param_map = self.tarp_module._current_named_params()
        meta_shapes: Dict[str, Tuple[int, ...]] = {}
        for m in self.tarp_module.mat_metas:
            meta_shapes[m.name] = m.orig_shape
        for v in self.tarp_module.vec_metas:
            meta_shapes[v.name] = v.orig_shape

        targets: Dict[str, Dict[str, Any]] = {}
        for name in self._tarp_param_names:
            if name not in meta_shapes:
                continue
            param = self.tarp_module._current_param(name, param_map=param_map)
            full_numel = int(math.prod(meta_shapes[name]))
            if param.numel() == 0:
                continue

            shard_info = (self._tarp_shard_param_infos or {}).get(name)
            if shard_info is not None and getattr(shard_info, "in_shard", False):
                start = getattr(shard_info, "intra_param_start_idx", None)
                end = getattr(shard_info, "intra_param_end_idx", None)
                if start is None or end is None or end < start:
                    continue
                start = int(start)
                end = int(end)
                expected_numel = end - start + 1
                if expected_numel != param.numel():
                    if param.numel() == full_numel:
                        start = 0
                        end = full_numel - 1
                    else:
                        raise RuntimeError(
                            f"TARP local target mismatch for {name}: "
                            f"shard_numel={expected_numel} param_numel={param.numel()} "
                            f"param_shape={tuple(param.shape)} full_shape={meta_shapes[name]}"
                        )
            else:
                if param.numel() != full_numel:
                    continue
                start = 0
                end = full_numel - 1

            targets[name] = {
                "param": param,
                "shape": tuple(param.shape),
                "numel": int(param.numel()),
                "start": start,
                "end": end,
            }
        return targets

    def _add_tarp_local_gradient(self, name: str, target: Dict[str, Any], local_grad: torch.Tensor, scale: float) -> bool:
        param = target["param"]
        if local_grad.numel() != int(target["numel"]):
            raise RuntimeError(
                f"TARP local gradient numel mismatch for {name}: "
                f"grad_numel={local_grad.numel()} target_numel={target['numel']}"
            )

        target_shape = tuple(target["shape"])
        grad = param.grad
        if grad is None:
            if tuple(param.shape) != target_shape:
                return False
            target_dtype = param.dtype
            param.grad = (local_grad.reshape(target_shape).to(device=param.device, dtype=target_dtype) * scale).clone()
            return True

        if tuple(grad.shape) != target_shape:
            raise RuntimeError(
                f"TARP local grad add shape mismatch for {name}: "
                f"existing_grad={tuple(grad.shape)} target_shape={target_shape}"
            )
        local_grad = local_grad.reshape(target_shape).to(device=grad.device, dtype=grad.dtype) * scale
        grad.add_(local_grad)
        return True

    def _compute_or_apply_tarp_local_gradients(self, scale: float = 0.0, apply: bool = False) -> float:
        import torch.distributed as dist

        if self.tarp_module is None:
            return 0.0

        targets = self._collect_tarp_local_targets()
        if not targets:
            return 0.0

        device = get_torch_device().current_device()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        local_sq = torch.zeros((), device=dev, dtype=torch.float32)
        skipped_apply: List[str] = []
        alpha_proj = (1.0 / float(self.tarp_module.proj_dim)) if self.tarp_module.average_over_proj_dim else 1.0

        def consume(name: str, target: Dict[str, Any], local_grad: torch.Tensor) -> None:
            nonlocal local_sq
            if local_grad.numel() == 0:
                return
            local_sq = local_sq + local_grad.detach().float().pow(2).sum()
            if apply and scale != 0.0:
                if not self._add_tarp_local_gradient(name, target, local_grad, scale=scale):
                    skipped_apply.append(name)

        with torch.no_grad(), self._tarp_full_param_ctx():
            param_map = self.tarp_module._current_named_params()

            for v in self.tarp_module.vec_metas:
                target = targets.get(v.name)
                if target is None:
                    continue
                p = self.tarp_module._current_param(v.name, param_map=param_map)
                if self.tarp_module._shape_tuple(p) != v.orig_shape:
                    raise RuntimeError(
                        f"TARP local grad expected full vector param view for {v.name}, "
                        f"got shape={self.tarp_module._shape_tuple(p)} expected={v.orig_shape}"
                    )
                if self.tarp_module._cached_device:
                    ts = self.tarp_module.theta_star_gpu[v.name]
                    fd = self.tarp_module.fisher_diag_gpu[v.name]
                else:
                    ts = self.tarp_module.theta_star[v.name].to(p.device)
                    fd = self.tarp_module.fisher_diag_cpu[v.name].to(p.device)

                start = int(target["start"])
                end = int(target["end"]) + 1
                delta = p.reshape(-1)[start:end].float() - ts.reshape(-1)[start:end].float()
                local_grad = fd.reshape(-1)[start:end].float() * delta
                consume(v.name, target, local_grad)

            source_subspaces = self.tarp_module.subspaces_gpu if self.tarp_module._cached_device else self.tarp_module.subspaces_cpu
            for m in self.tarp_module.mat_metas:
                target = targets.get(m.name)
                if target is None or m.name not in source_subspaces:
                    continue
                p = self.tarp_module._current_param(m.name, param_map=param_map)
                if self.tarp_module._shape_tuple(p) != m.orig_shape:
                    raise RuntimeError(
                        f"TARP local grad expected full matrix param view for {m.name}, "
                        f"got shape={self.tarp_module._shape_tuple(p)} expected={m.orig_shape}"
                    )
                if self.tarp_module._cached_device:
                    ts = self.tarp_module.theta_star_gpu[m.name]
                    R = self.tarp_module.R_gpu[m.name]
                    subspace_blocks = self.tarp_module.subspaces_gpu[m.name]
                else:
                    ts = self.tarp_module.theta_star[m.name].to(p.device)
                    R = self.tarp_module.R_cpu[m.name].to(p.device)
                    subspace_blocks = self.tarp_module.subspaces_cpu[m.name]

                local_start = int(target["start"])
                local_end_excl = int(target["end"]) + 1
                local_grad_flat = torch.zeros((int(target["numel"]),), device=p.device, dtype=torch.float32)
                p_matrix = p.reshape(m.out_dim, m.in_dim)
                ts_matrix = ts.reshape(m.out_dim, m.in_dim)
                R_ = R.to(device=p.device, dtype=p.dtype)
                R_float_t = R_.float().t()

                for bi, (U, lam) in enumerate(subspace_blocks):
                    b0 = bi * m.block_size
                    b1 = min(m.out_dim, (bi + 1) * m.block_size)
                    block_start = b0 * m.in_dim
                    block_end_excl = b1 * m.in_dim
                    overlap_start = max(local_start, block_start)
                    overlap_end_excl = min(local_end_excl, block_end_excl)
                    if overlap_start >= overlap_end_excl:
                        continue

                    delta_block = p_matrix[b0:b1, :] - ts_matrix[b0:b1, :].to(dtype=p.dtype)
                    Z = torch.matmul(delta_block, R_).float()
                    U_ = U.to(device=Z.device, dtype=torch.float32)
                    lam_ = lam.to(device=Z.device, dtype=torch.float32)
                    projected = torch.matmul(U_.t(), Z)
                    grad_proj = torch.matmul(U_, lam_.unsqueeze(1) * projected).mul_(alpha_proj)
                    grad_block = torch.matmul(grad_proj, R_float_t)

                    src0 = overlap_start - block_start
                    src1 = overlap_end_excl - block_start
                    dst0 = overlap_start - local_start
                    dst1 = overlap_end_excl - local_start
                    local_grad_flat[dst0:dst1].copy_(grad_block.reshape(-1)[src0:src1])

                consume(m.name, target, local_grad_flat)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

        if skipped_apply:
            if not hasattr(self, "_tarp_skipped_local_apply_printed"):
                self._tarp_skipped_local_apply_printed = False
            if not self._tarp_skipped_local_apply_printed:
                self._tarp_skipped_local_apply_printed = True
                print(
                    f"[TARP] WARN skipped local TARP grad apply for params without local grad slots: {skipped_apply[:8]}",
                    flush=True,
                )

        return math.sqrt(max(float(local_sq.detach().item()), 0.0))

    def _compute_pg_grad_norm_for_tarp_params(self) -> float:
        import torch.distributed as dist

        targets = self._collect_tarp_local_targets()
        if not targets:
            return 0.0
        device = get_torch_device().current_device()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        local_sq = torch.zeros((), device=dev, dtype=torch.float32)
        for target in targets.values():
            param = target["param"]
            if param.grad is None:
                continue
            local_sq = local_sq + param.grad.detach().float().pow(2).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        return math.sqrt(max(float(local_sq.detach().item()), 0.0))

    def _initialize_ewc_module(self):
        if self.ewc_module is not None:
            return
        with self._tarp_full_param_ctx():
            self.ewc_module = DiagonalFisherEWC(
                model=self.actor_module,
                store_dtype=getattr(self.config, "ewc_store_dtype", torch.bfloat16),
            )
            self._ewc_param_names = list(self.ewc_module.param_names)
            self._ewc_full_shapes = {ref.name: tuple(ref.param.shape) for ref in self.ewc_module.param_refs}
        self._ewc_shard_param_infos = self._collect_tarp_shard_param_infos()
        self._ewc_state_loaded = False

    def _ewc_state_debug_on_load(self) -> bool:
        return bool(getattr(self.config, "ewc_state_debug_on_load", True))

    def _log_ewc_state_magnitude_summary(self, rank: int) -> None:
        if rank != 0 or not self._ewc_state_debug_on_load():
            return
        if self.ewc_module is None:
            return
        if rank in self._ewc_state_summary_printed:
            return
        self._ewc_state_summary_printed.add(rank)
        summary = self.ewc_module.state_magnitude_summary()
        topk = self._tarp_debug_top_layers()
        fisher_layers = summary.get("fisher_layer_stats", {})
        fisher_top = sorted(fisher_layers.items(), key=lambda kv: kv[1].get("max", 0.0), reverse=True)[:topk]
        print(
            "[EWC State] "
            f"fisher_mean={summary.get('fisher_global_mean', 0.0):.6e} "
            f"fisher_max={summary.get('fisher_global_max', 0.0):.6e} "
            f"fisher_count={int(summary.get('fisher_count', 0))}",
            flush=True,
        )
        if fisher_top:
            fisher_str = ", ".join(
                f"{name}(mean={stat.get('mean', 0.0):.3e},max={stat.get('max', 0.0):.3e})"
                for name, stat in fisher_top
            )
            print(f"[EWC State] top_fisher_layers: {fisher_str}", flush=True)

    @staticmethod
    def _is_fully_correct_score(score: float, data_source: Optional[str]) -> bool:
        source = "" if data_source is None else str(data_source).lower()
        if "kk" in source:
            return score >= 3.0 - 1e-6
        if source == "code" or "code-r1" in source:
            return score >= 1.1 - 1e-6
        return score >= 1.0 - 1e-6

    def _collect_ewc_local_targets(self) -> Dict[str, Dict[str, Any]]:
        if self.ewc_module is None or not self._ewc_param_names:
            return {}

        param_map: Dict[str, torch.nn.Parameter] = {}
        for name, param in self.actor_module.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            for alias in _param_name_aliases(name):
                param_map.setdefault(alias, param)
        full_shapes = self._ewc_full_shapes or {}
        targets: Dict[str, Dict[str, Any]] = {}

        for name in self._ewc_param_names:
            full_shape = full_shapes.get(name)
            param = param_map.get(name)
            if full_shape is None or param is None or param.numel() == 0:
                continue

            full_numel = int(math.prod(full_shape))
            shard_info = (self._ewc_shard_param_infos or {}).get(name)
            if shard_info is not None and getattr(shard_info, "in_shard", False):
                start = getattr(shard_info, "intra_param_start_idx", None)
                end = getattr(shard_info, "intra_param_end_idx", None)
                if start is None or end is None or end < start:
                    continue
                start = int(start)
                end = int(end)
                expected_numel = end - start + 1
                if expected_numel != param.numel():
                    if param.numel() == full_numel:
                        start = 0
                        end = full_numel - 1
                    else:
                        raise RuntimeError(
                            f"EWC local target mismatch for {name}: "
                            f"shard_numel={expected_numel} param_numel={param.numel()} "
                            f"param_shape={tuple(param.shape)} full_shape={full_shape}"
                        )
            else:
                if param.numel() != full_numel:
                    continue
                start = 0
                end = full_numel - 1

            targets[name] = {
                "param": param,
                "shape": tuple(param.shape),
                "numel": int(param.numel()),
                "start": start,
                "end": end,
            }

        if not targets:
            import torch.distributed as dist

            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
            if rank == 0 and not getattr(self, "_ewc_targets_empty_warned", False):
                self._ewc_targets_empty_warned = True
                preview = ", ".join(list(param_map.keys())[:4])
                wanted = ", ".join(list(self._ewc_param_names)[:4])
                print(
                    "[EWC] WARN local targets empty. "
                    f"local_param_alias_preview=[{preview}] "
                    f"wanted_param_preview=[{wanted}]",
                    flush=True,
                )

        return targets

    def _initialize_ewc_local_theta_star(self, targets: Dict[str, Dict[str, Any]]) -> None:
        if self.ewc_module is None or self._ewc_local_theta_star is not None or not targets:
            return

        theta_local: Dict[str, torch.Tensor] = {}
        with torch.no_grad(), self._tarp_full_param_ctx():
            param_map = self.ewc_module._current_param_map()
            for name, target in targets.items():
                param = param_map.get(name)
                if param is None:
                    continue
                start = int(target["start"])
                end_excl = int(target["end"]) + 1
                theta_local[name] = param.reshape(-1)[start:end_excl].detach().cpu().clone().to(torch.float32)
        self._ewc_local_theta_star = theta_local

    def _compute_ewc_penalty_value(self, apply_grad: bool = False, scale: float = 0.0) -> torch.Tensor:
        import torch.distributed as dist

        device = get_torch_device().current_device()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if not is_ewc_loss_enabled(self.config):
            return torch.zeros((), device=dev, dtype=torch.float32)

        if resolve_ewc_mode(self.config) != "DIAG_FISHER":
            raise ValueError(f"Unsupported ewc_mode for training: {getattr(self.config, 'ewc_mode', None)}")

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world = dist.get_world_size() if is_dist else 1

        self._initialize_ewc_module()

        if not self._ewc_state_loaded:
            candidates = resolve_ewc_load_candidates(self.config, rank, world)
            fpath = next((path for path in candidates if os.path.exists(path)), None)
            if fpath is None:
                if not hasattr(self, "_ewc_missing_printed"):
                    self._ewc_missing_printed = set()
                if rank not in self._ewc_missing_printed:
                    self._ewc_missing_printed.add(rank)
                    if candidates:
                        print(f"[EWC] INFO rank{rank} no diagonal Fisher shard found. candidates={candidates}", flush=True)
                    else:
                        print(f"[EWC] INFO rank{rank} no EWC load target configured; training without penalty.", flush=True)
                return torch.zeros((), device=dev, dtype=torch.float32)

            state_dict = torch.load(fpath, map_location="cpu")
            with self._tarp_full_param_ctx():
                validate_ewc_state_dict(
                    state_dict,
                    self.ewc_module.param_names,
                    expected_mode=resolve_ewc_mode(self.config),
                )
                self.ewc_module.load_state_dict(state_dict)
            self._ewc_state_loaded = True

            if not hasattr(self, "_ewc_loaded_printed"):
                self._ewc_loaded_printed = set()
            if rank not in self._ewc_loaded_printed:
                self._ewc_loaded_printed.add(rank)
                print(f"[EWC] rank{rank} loaded diagonal Fisher shard: {fpath}", flush=True)
            self._log_ewc_state_magnitude_summary(rank)

        targets = self._collect_ewc_local_targets()
        if not targets:
            return torch.zeros((), device=dev, dtype=torch.float32)
        self._initialize_ewc_local_theta_star(targets)

        pen = torch.zeros((), device=dev, dtype=torch.float32)
        skipped_apply: List[str] = []
        max_abs_delta = 0.0
        nonzero_delta_params = 0
        with torch.no_grad(), self._tarp_full_param_ctx():
            param_map = self.ewc_module._current_param_map()
            for name in self._ewc_param_names:
                target = targets.get(name)
                if target is None:
                    continue

                param = param_map.get(name)
                full_shape = (self._ewc_full_shapes or {}).get(name)
                if param is None or full_shape is None:
                    continue
                if tuple(param.shape) != tuple(full_shape):
                    raise RuntimeError(
                        f"EWC expected full parameter view for {name}, "
                        f"got shape={tuple(param.shape)} expected={tuple(full_shape)}"
                    )

                start = int(target["start"])
                end_excl = int(target["end"]) + 1
                theta_local = None if self._ewc_local_theta_star is None else self._ewc_local_theta_star.get(name)
                if theta_local is not None and theta_local.numel() == (end_excl - start):
                    theta_slice = theta_local.to(device=param.device, dtype=torch.float32)
                else:
                    theta_slice = self.ewc_module.theta_star[name].reshape(-1)[start:end_excl].to(
                        device=param.device, dtype=torch.float32
                    )
                fisher_slice = self.ewc_module.fisher_diag_cpu[name].reshape(-1)[start:end_excl].to(
                    device=param.device, dtype=torch.float32
                )
                delta = param.reshape(-1)[start:end_excl].float() - theta_slice
                if delta.numel() > 0:
                    delta_abs_max = float(delta.abs().max().item())
                    if delta_abs_max > max_abs_delta:
                        max_abs_delta = delta_abs_max
                    if delta_abs_max > 0.0:
                        nonzero_delta_params += 1
                pen = pen + 0.5 * (fisher_slice * delta.pow(2)).sum()

                if apply_grad and scale != 0.0:
                    local_grad = fisher_slice * delta
                    if not self._add_tarp_local_gradient(name, target, local_grad, scale=scale):
                        skipped_apply.append(name)

        if skipped_apply and rank == 0:
            preview = ", ".join(skipped_apply[:3])
            more = "" if len(skipped_apply) <= 3 else f", ... (+{len(skipped_apply) - 3} more)"
            print(f"[EWC] WARN skipped local grad apply for {preview}{more}", flush=True)
        self._ewc_penalty_debug_calls += 1
        if rank == 0 and self._ewc_penalty_debug_calls <= 4:
            print(
                "[EWC Debug] "
                f"call={self._ewc_penalty_debug_calls} "
                f"max_abs_delta={max_abs_delta:.6e} "
                f"nonzero_delta_params={nonzero_delta_params}",
                flush=True,
            )

        if is_dist:
            dist.all_reduce(pen, op=dist.ReduceOp.SUM)
        if not torch.isfinite(pen):
            if not hasattr(self, "_ewc_warned_nonfinite"):
                self._ewc_warned_nonfinite = set()
            if rank not in self._ewc_warned_nonfinite:
                self._ewc_warned_nonfinite.add(rank)
                print(f"[EWC] WARN rank{rank} penalty not finite: {pen.detach().item()}", flush=True)
            return torch.zeros((), device=dev, dtype=torch.float32)
        return pen

    def _estimate_diag_fisher_remote(
        self,
        dataloader,
        tokenizer,
        reward_fn,
    ):
        import gc
        import time
        import torch.nn.functional as F
        import torch.distributed as dist

        if resolve_ewc_mode(self.config) != "DIAG_FISHER":
            raise ValueError(f"Unsupported ewc_mode for export: {getattr(self.config, 'ewc_mode', None)}")

        pid = os.getpid()
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world = dist.get_world_size() if is_dist else 1

        try:
            if hasattr(self.actor_module, "config"):
                self.actor_module.config.use_cache = True
        except Exception:
            pass

        def _check_meta_params(module):
            return [name for name, param in module.named_parameters() if getattr(param, "is_meta", False)]

        if _check_meta_params(self.actor_module):
            raise RuntimeError("[EWC] Meta params found. Cannot proceed.")

        self.actor_module.eval()
        self._initialize_ewc_module()
        self.ewc_module.snapshot_theta_star()

        device_name = getattr(self, "device_name", "cuda")
        device = get_torch_device().current_device()
        target_valid_samples = int(getattr(self.config, "ewc_target_valid_samples", 32))
        n_traj = 1
        max_new_tokens = int(getattr(self.config, "ewc_max_new_tokens", 512))
        max_fisher_seq_len = int(getattr(self.config, "ewc_max_fisher_seq_len", 1024))
        skip_overlong_fisher_seq = bool(getattr(self.config, "ewc_skip_overlong_fisher_seq", False))
        autocast_dtype = torch.bfloat16
        use_autocast = device_name == "cuda"

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        answer_end_ids = tokenizer("</answer>", add_special_tokens=False)["input_ids"]
        logits_processor = LogitsProcessorList([PerSequenceAnswerStop(answer_end_ids, eos_token_id, pad_token_id)])

        valid_samples_count = 0
        attempted_samples_count = 0
        data_source_observed: List[str] = []

        for batch_idx, batch_dict in enumerate(dataloader):
            if valid_samples_count >= target_valid_samples:
                break

            original_batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            gen_batch = original_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_keys)
            batch_data = {**gen_batch.batch.to(device=device), **gen_batch.non_tensor_batch}

            input_ids = batch_data["input_ids"][:1]
            attention_mask = batch_data["attention_mask"][:1]
            prompt_len = int(input_ids.size(1))

            ground_truth = None
            extra_info = None
            data_source = None
            if "data_source" in original_batch.non_tensor_batch:
                data_source = original_batch.non_tensor_batch["data_source"][0]
            elif "reward_fn_key" in original_batch.non_tensor_batch:
                data_source = original_batch.non_tensor_batch["reward_fn_key"][0]
            if data_source is not None:
                data_source_observed.append(str(data_source))
            if "reward_model" in original_batch.non_tensor_batch:
                rm_item = original_batch.non_tensor_batch["reward_model"][0]
                if isinstance(rm_item, dict):
                    ground_truth = rm_item.get("ground_truth", None)
            if "extra_info" in original_batch.non_tensor_batch:
                extra_info = original_batch.non_tensor_batch["extra_info"][0]

            is_fsdp = isinstance(self.actor_module, FSDP)
            param_ctx = (
                FSDP.summon_full_params(self.actor_module, writeback=False, recurse=False)
                if is_fsdp
                else contextlib.nullcontext()
            )

            attempted_samples_count += 1
            if rank == 0:
                print(
                    f"[EWC Export pid={pid}] batch={batch_idx} "
                    f"attempt={attempted_samples_count} valid={valid_samples_count}/{target_valid_samples}",
                    flush=True,
                )

            generate_ctx = (
                torch.autocast(device_type=device_name, dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with param_ctx, torch.no_grad(), generate_ctx:
                outputs = self.actor_module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    num_beams=1,
                    num_return_sequences=n_traj,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                    logits_processor=logits_processor,
                    return_dict_in_generate=True,
                )
                seqs = outputs.sequences.detach().cpu()
                del outputs

            full_ids_cpu = seqs[0]
            gen_ids = full_ids_cpu[prompt_len:]
            response_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
            score = self._score_response(reward_fn, data_source, response_str, ground_truth, extra_info)
            if not self._is_fully_correct_score(score, data_source):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            rel_end = trim_end_by_pad(gen_ids, pad_token_id)
            seq_end = prompt_len + int(rel_end)
            if seq_end <= prompt_len:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            anchor_ids_cpu = full_ids_cpu[:seq_end]
            if anchor_ids_cpu.numel() > max_fisher_seq_len:
                if skip_overlong_fisher_seq:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                old_len = int(anchor_ids_cpu.numel())
                offset = old_len - int(max_fisher_seq_len)
                anchor_ids_cpu = anchor_ids_cpu[offset:]
                prompt_len = max(0, prompt_len - offset)

            if anchor_ids_cpu.numel() <= prompt_len:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            anchor_ids_gpu = anchor_ids_cpu.unsqueeze(0).to(device, non_blocking=True)
            self.actor_module.zero_grad(set_to_none=True)

            t0 = time.perf_counter()
            fisher_ctx = (
                torch.autocast(device_type=device_name, dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with fisher_ctx:
                out = self.actor_module(
                    input_ids=anchor_ids_gpu,
                    attention_mask=torch.ones_like(anchor_ids_gpu),
                    use_cache=False,
                )
                logits = out.logits[:, :-1, :]
                labels = anchor_ids_gpu[:, 1:]

                nll = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="none",
                    )
                    .view_as(labels)
                    .float()
                )
                token_weight = torch.zeros_like(nll, dtype=torch.float32)
                valid_start_idx = max(0, prompt_len - 1)
                if valid_start_idx < token_weight.size(1):
                    token_weight[:, valid_start_idx:] = 1.0

                valid_count = token_weight.sum().clamp(min=1.0)
                loss = (nll * token_weight).sum() / valid_count

            if torch.isfinite(loss) and float(loss.detach().item()) != 0.0:
                loss.backward()
                self.ewc_module.accumulate_fisher_after_backward()
                valid_samples_count += 1
                if rank == 0:
                    print(
                        f"[EWC Export] accumulated sample {valid_samples_count}/{target_valid_samples} "
                        f"loss={float(loss.detach().item()):.6e} dt={time.perf_counter() - t0:.2f}s",
                        flush=True,
                    )

            self.actor_module.zero_grad(set_to_none=True)
            del out, logits, labels, nll, token_weight, loss, anchor_ids_gpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if valid_samples_count <= 0:
            raise RuntimeError("[EWC] Export finished without any valid fully-correct samples.")

        if rank == 0:
            print(f"[EWC] Finalizing diagonal Fisher with {valid_samples_count} valid samples...", flush=True)
        self.ewc_module.finalize_fisher()

        primary_data_source = next((source for source in data_source_observed if source), None)
        dataset_key = (
            get_ewc_dataset_key(self.config, data_source=primary_data_source)
            or infer_primary_ewc_dataset_key(data_source_observed)
            or "UNKNOWN"
        )
        part_path, merged_path = resolve_ewc_save_paths(self.config, dataset_key, rank, world, valid_samples_count)
        state = self.ewc_module.state_dict()
        state["_meta"] = build_ewc_state_meta(
            dataset_key=dataset_key,
            data_source=primary_data_source,
            config=self.config,
            rank=rank,
            world_size=world,
            valid_samples_used=valid_samples_count,
        )
        state["_meta"]["pid"] = int(pid)

        def _atomic_save(path: str):
            tmp_path = path + f".tmp{pid}"
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)

        _atomic_save(part_path)
        _atomic_save(merged_path)

        if is_dist:
            dist.barrier()
        if rank == 0:
            print(f"[EWC] Saved part shard: {part_path}", flush=True)
            print(f"[EWC] Saved merged shard: {merged_path}", flush=True)

    def _score_response(self, reward_fn, data_source, response_str, ground_truth, extra_info):
        if ground_truth is None or data_source is None:
            return 0.0

        score_fn = getattr(reward_fn, "compute_score", None) if reward_fn is not None else None
        if score_fn is None:
            from verl.utils.reward_score import default_compute_score

            score_fn = default_compute_score

        try:
            score = score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except TypeError:
            score = score_fn(data_source, response_str, ground_truth, extra_info)
        except Exception:
            return 0.0

        if isinstance(score, dict):
            score = score.get("score", 0.0)
        return float(score)

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        nonfinite_grad_tensors = 0
        with torch.no_grad():
            for p in self.actor_module.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    nonfinite_grad_tensors += 1
                    g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if nonfinite_grad_tensors > 0:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            print(f"[TARP] WARN rank {rank} sanitized {nonfinite_grad_tensors} non-finite gradient tensors before clipping", flush=True)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        dist_is_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        dist_rank = torch.distributed.get_rank() if dist_is_initialized else 0

        def _is_finite_all_ranks(x: torch.Tensor) -> bool:
            finite_local = bool(torch.isfinite(x.detach()).all().item())
            if not dist_is_initialized:
                return finite_local
            finite_flag = torch.tensor(1 if finite_local else 0, device=x.device, dtype=torch.int32)
            torch.distributed.all_reduce(finite_flag, op=torch.distributed.ReduceOp.MIN)
            return bool(finite_flag.item())

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                tarp_enabled_for_batch = resolve_tarp_train_mode(self.config) == "continual"
                ewc_enabled_for_batch = is_ewc_loss_enabled(self.config)
                if tarp_enabled_for_batch and ewc_enabled_for_batch:
                    raise ValueError("TARP continual penalty and EWC penalty cannot be enabled at the same time.")
                # Warmup should track optimizer-update steps (mini-batches), not micro-batches.
                if tarp_enabled_for_batch:
                    self._tarp_update_step += 1
                tarp_update_step_for_batch = self._tarp_update_step
                tarp_batch_pg_loss_abs = 1e-8

                for micro_batch_idx, data in enumerate(micro_batches):
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                    pg_loss_scalar = float(pg_loss.detach().item())
                    pg_loss_finite = _is_finite_all_ranks(pg_loss)
                    if not pg_loss_finite:
                        if dist_rank == 0:
                            print(
                                f"[TARP] WARN non-finite pg_loss detected at epoch={epoch} mini_batch={batch_idx} micro_batch={micro_batch_idx}: {pg_loss_scalar}",
                                flush=True,
                            )
                        pg_loss = torch.nan_to_num(pg_loss, nan=0.0, posinf=0.0, neginf=0.0)
                        pg_loss_abs = 1e-8
                    else:
                        pg_loss_abs = max(abs(pg_loss_scalar), 1e-8)
                    tarp_batch_pg_loss_abs = max(tarp_batch_pg_loss_abs, pg_loss_abs)

                    # 3️⃣ optional TARP loss (Subspace Version)
                    tarp_enabled = tarp_enabled_for_batch
                    tarp_apply_on_this_micro_batch = tarp_enabled and (micro_batch_idx == len(micro_batches) - 1)
                    tarp_grad_scale = 0.0
                    tarp_penalty_value = 0.0
                    tarp_lambda_raw = 0.0
                    tarp_lambda_eff = 0.0
                    warmup_frac = 1.0
                    tarp_ratio_cap = 0.0
                    tarp_ratio_raw = 0.0
                    ratio_scale = 1.0
                    scaled_tarp_value = 0.0
                    tarp_pg_small_skip = 0.0
                    tarp_pg_loss_floor = float(getattr(self.config, "tarp_pg_loss_floor", 1e-2))
                    tarp_disable_when_pg_small = bool(getattr(self.config, "tarp_disable_when_pg_small", True))
                    tarp_grad_to_pg_ratio_cap = 0.0
                    tarp_grad_ratio_raw = 0.0
                    tarp_grad_ratio = 0.0
                    tarp_grad_norm = 0.0
                    pg_grad_norm = 0.0
                    ewc_apply_on_this_micro_batch = ewc_enabled_for_batch and (micro_batch_idx == len(micro_batches) - 1)
                    ewc_penalty_value = 0.0
                    ewc_lambda = 0.0
                    ewc_term_scaled = 0.0
                    if tarp_enabled:
                        tarp_lambda_raw = float(getattr(self.config, "tarp_lambda", 0.0))
                        tarp_warmup_steps = int(getattr(self.config, "tarp_warmup_steps", 0))
                        if tarp_warmup_steps > 0:
                            warmup_frac = min(1.0, float(tarp_update_step_for_batch) / float(tarp_warmup_steps))
                        else:
                            warmup_frac = 1.0
                        tarp_lambda_eff = tarp_lambda_raw * warmup_frac
                        tarp_ratio_cap = float(getattr(self.config, "tarp_to_pg_ratio_cap", 0.0))
                        tarp_grad_to_pg_ratio_cap = max(0.0, float(getattr(self.config, "tarp_grad_to_pg_ratio_cap", 0.0)))

                    # 4️⃣ optional KL loss
                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # if getattr(self.config, "use_tarp_loss", False):
                    #     print(f"TARP Loss: {self.config.tarp_lambda * tarp_penalty}, KL Loss: {kl_loss * self.config.kl_loss_coef}")

                    if self.config.use_dynamic_bsz:
                        loss_scale = len(data) / self.config.ppo_mini_batch_size
                        loss = policy_loss * loss_scale
                    else:
                        loss_scale = 1.0 / self.gradient_accumulation
                        loss = policy_loss * loss_scale
                    loss_finite = _is_finite_all_ranks(loss)
                    if not loss_finite:
                        if dist_rank == 0:
                            print(
                                f"[TARP] WARN non-finite policy loss detected at epoch={epoch} mini_batch={batch_idx} micro_batch={micro_batch_idx}; skipping backward for this micro-batch.",
                                flush=True,
                            )
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    loss.backward()

                    if ewc_apply_on_this_micro_batch:
                        ewc_lambda = float(getattr(self.config, "ewc_lambda", 0.0))
                        ewc_penalty = self._compute_ewc_penalty_value(apply_grad=ewc_lambda != 0.0, scale=ewc_lambda)
                        ewc_penalty_value = float(ewc_penalty.detach().item())
                        if ewc_penalty_value != 0.0 and ewc_lambda != 0.0:
                            ewc_term_scaled = ewc_penalty_value * ewc_lambda
                        ewc_metric = {
                            "actor/ewc_loss": ewc_penalty_value,
                            "actor/ewc_coef": ewc_lambda,
                            "actor/ewc_term_scaled": ewc_term_scaled,
                            f"actor/ewc_loss_mb_{micro_batch_idx}": ewc_penalty_value,
                        }
                        append_to_dict(metrics, ewc_metric)
                        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                            print(
                                "[EWC] "
                                f"epoch={epoch} mini_batch={batch_idx} micro_batch={micro_batch_idx} "
                                f"ewc_loss={ewc_penalty_value:.6e} "
                                f"ewc_coef={ewc_lambda:.6e} "
                                f"ewc_term_scaled={ewc_term_scaled:.6e}",
                                flush=True,
                            )

                    # NOTE:
                    # For FSDP + orig-view TARP we must avoid calling summon_full_params
                    # between policy forward and loss.backward(). Compute TARP penalty after
                    # backward to prevent autograd shape/view mismatches.
                    if tarp_apply_on_this_micro_batch:
                        pg_loss_abs_for_tarp = max(tarp_batch_pg_loss_abs, 1e-8)
                        tarp_penalty = self._compute_tarp_penalty_value()
                        tarp_penalty_value = float(tarp_penalty.detach().item())
                        if tarp_penalty_value != 0.0 and tarp_lambda_eff != 0.0:
                            if tarp_disable_when_pg_small and pg_loss_abs_for_tarp < tarp_pg_loss_floor:
                                tarp_pg_small_skip = 1.0
                                tarp_grad_scale = 0.0
                                ratio_scale = 0.0
                                tarp_ratio_raw = 0.0
                                scaled_tarp_value = 0.0
                            else:
                                scaled_tarp = tarp_lambda_eff * tarp_penalty
                                if torch.isfinite(scaled_tarp):
                                    scaled_tarp_value = float(scaled_tarp.detach().item())
                                else:
                                    print(
                                        f"[TARP] WARN scaled penalty not finite, skipping step contribution: {scaled_tarp.detach().item()}",
                                        flush=True,
                                    )
                                if scaled_tarp_value != 0.0:
                                    tarp_ratio_raw = abs(scaled_tarp_value) / pg_loss_abs_for_tarp
                                    if tarp_ratio_cap > 0.0 and tarp_ratio_raw > tarp_ratio_cap:
                                        ratio_scale = tarp_ratio_cap / max(tarp_ratio_raw, 1e-8)
                                    tarp_grad_scale = tarp_lambda_eff * ratio_scale
                                    # Optional extra safety: cap by gradient-norm ratio.
                                    # This uses local analytic TARP gradients and avoids materializing full-model grads.
                                    if tarp_grad_to_pg_ratio_cap > 0.0:
                                        tarp_grad_norm = self._compute_or_apply_tarp_local_gradients(scale=0.0, apply=False)
                                        pg_grad_norm = self._compute_pg_grad_norm_for_tarp_params()

                                        if pg_grad_norm > 0.0 and tarp_grad_norm > 0.0:
                                            tarp_grad_ratio_raw = (tarp_lambda_eff * tarp_grad_norm) / (pg_grad_norm + 1e-12)
                                            tarp_grad_ratio = (tarp_grad_scale * tarp_grad_norm) / (pg_grad_norm + 1e-12)
                                            if tarp_grad_ratio > tarp_grad_to_pg_ratio_cap:
                                                grad_scale_cap = tarp_grad_to_pg_ratio_cap * pg_grad_norm / (tarp_grad_norm + 1e-12)
                                                tarp_grad_scale = min(tarp_grad_scale, grad_scale_cap)
                                                tarp_grad_ratio = (tarp_grad_scale * tarp_grad_norm) / (pg_grad_norm + 1e-12)

                        if tarp_grad_scale != 0.0:
                            tarp_grad_norm = self._compute_or_apply_tarp_local_gradients(scale=tarp_grad_scale, apply=True)
                            if tarp_grad_to_pg_ratio_cap <= 0.0:
                                pg_grad_norm = 0.0
                                tarp_grad_ratio_raw = 0.0
                                tarp_grad_ratio = 0.0

                        tarp_metric = {
                            "actor/tarp_loss": tarp_penalty_value,
                            "actor/tarp_coef_raw": tarp_lambda_raw,
                            "actor/tarp_coef": tarp_grad_scale,
                            "actor/tarp_warmup": warmup_frac,
                            "actor/tarp_term_scaled": scaled_tarp_value,
                            "actor/tarp_to_pg_ratio_raw": tarp_ratio_raw,
                            "actor/tarp_to_pg_ratio": tarp_ratio_raw * ratio_scale,
                            "actor/tarp_pg_loss_abs": pg_loss_abs_for_tarp,
                            "actor/tarp_pg_small_skip": tarp_pg_small_skip,
                            "actor/tarp_grad_norm": tarp_grad_norm,
                            "actor/pg_grad_norm_proxy": pg_grad_norm,
                            "actor/tarp_grad_to_pg_ratio_raw": tarp_grad_ratio_raw,
                            "actor/tarp_grad_to_pg_ratio": tarp_grad_ratio,
                            "actor/tarp_grad_to_pg_ratio_cap": tarp_grad_to_pg_ratio_cap,
                            "actor/pg_loss_nonfinite": 0.0 if pg_loss_finite else 1.0,
                            f"actor/tarp_loss_mb_{micro_batch_idx}": tarp_penalty_value,
                        }
                        append_to_dict(metrics, tarp_metric)
                        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                            print(
                                "[TARP] "
                                f"epoch={epoch} mini_batch={batch_idx} micro_batch={micro_batch_idx} "
                                f"tarp_loss={tarp_penalty_value:.6e} "
                                f"tarp_coef_raw={tarp_lambda_raw:.6e} "
                                f"tarp_coef={tarp_grad_scale:.6e} "
                                f"tarp_term_scaled={scaled_tarp_value:.6e} "
                                f"tarp_to_pg_ratio_raw={tarp_ratio_raw:.6e} "
                                f"tarp_to_pg_ratio={(tarp_ratio_raw * ratio_scale):.6e} "
                                f"pg_loss_abs={pg_loss_abs_for_tarp:.6e} "
                                f"pg_small_skip={int(tarp_pg_small_skip)} "
                                f"tarp_grad_norm={tarp_grad_norm:.6e} "
                                f"pg_grad_norm={pg_grad_norm:.6e} "
                                f"tarp_grad_to_pg_ratio_raw={tarp_grad_ratio_raw:.6e} "
                                f"tarp_grad_to_pg_ratio={tarp_grad_ratio:.6e} "
                                f"tarp_grad_to_pg_ratio_cap={tarp_grad_to_pg_ratio_cap:.6e} "
                                f"tarp_to_pg_ratio_cap={tarp_ratio_cap:.6e}",
                                flush=True,
                            )

                    data_metric = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data_metric)

                grad_norm = self._optimizer_step()
                data_metric = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_metric)
        self.actor_optimizer.zero_grad()
        return metrics

    def _compute_tarp_penalty_value(self) -> torch.Tensor:
        import torch.distributed as dist
        from verl.utils.device import get_torch_device

        device = get_torch_device().current_device()
        dev = device if isinstance(device, torch.device) else torch.device(device)

        if resolve_tarp_train_mode(self.config) != "continual":
            return torch.zeros((), device=dev, dtype=torch.float32)

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world = dist.get_world_size() if is_dist else 1
        self._tarp_penalty_debug_calls += 1
        debug_enabled = self._tarp_debug_enabled()
        debug_interval = self._tarp_debug_interval()
        should_log_breakdown = debug_enabled and (self._tarp_penalty_debug_calls % debug_interval == 0)

        self._initialize_tarp_module()

        if not getattr(self, "_tarp_state_loaded", False):
            candidates = resolve_tarp_load_candidates(self.config, rank, world)
            fpath = next((path for path in candidates if os.path.exists(path)), None)
            if fpath is None:
                if not hasattr(self, "_tarp_missing_printed"):
                    self._tarp_missing_printed = set()
                if rank not in self._tarp_missing_printed:
                    self._tarp_missing_printed.add(rank)
                    if candidates:
                        print(f"[TARP] INFO rank{rank} no prior orig-view shard found for this stage. candidates={candidates}", flush=True)
                    else:
                        print(f"[TARP] INFO rank{rank} no TARP load target configured for this stage; training without penalty.", flush=True)
                return torch.zeros((), device=dev, dtype=torch.float32)

            state_dict = torch.load(fpath, map_location="cpu")
            validate_tarp_state_dict(
                state_dict,
                self._tarp_param_names or [],
                expected_config={
                    "rank_per_block": int(getattr(self.config, "tarp_rank_per_block", 32)),
                    "block_size": int(getattr(self.config, "tarp_block_size", 2048)),
                    "proj_dim": int(getattr(self.config, "tarp_proj_dim", 48)),
                },
            )
            self.tarp_module.load_state_dict(state_dict)
            self._tarp_state_loaded = True

            if not hasattr(self, "_tarp_loaded_printed"):
                self._tarp_loaded_printed = set()
            if rank not in self._tarp_loaded_printed:
                self._tarp_loaded_printed.add(rank)
                print(f"[TARP] rank{rank} loaded orig-view shard: {fpath}", flush=True)
            self._log_tarp_state_magnitude_summary(rank)

        cache_dtype = getattr(self.config, "tarp_cache_dtype", torch.bfloat16)
        if (getattr(self.tarp_module, "_cached_device", None) is None) or (self.tarp_module._cached_device != dev):
            self.tarp_module.cache_to_device(dev, dtype=cache_dtype)

        penalty_cap = float(getattr(self.config, "tarp_penalty_cap", 0.0))
        breakdown: Dict[str, Any] = {}
        with torch.no_grad(), self._tarp_full_param_ctx():
            if should_log_breakdown:
                pen, breakdown = self.tarp_module.penalty_with_breakdown(coef=1.0)
                self._log_tarp_penalty_breakdown(breakdown=breakdown, rank=rank)
            else:
                pen = self.tarp_module.penalty(coef=1.0)
            if penalty_cap > 0:
                pen = pen.clamp(max=penalty_cap)
            if not torch.isfinite(pen):
                pen = torch.nan_to_num(
                    pen,
                    nan=0.0,
                    posinf=(penalty_cap if penalty_cap > 0 else 0.0),
                    neginf=0.0,
                )
        pen = pen.float()

        if not torch.isfinite(pen):
            if not hasattr(self, "_tarp_warned_nonfinite"):
                self._tarp_warned_nonfinite = set()
            if rank not in self._tarp_warned_nonfinite:
                self._tarp_warned_nonfinite.add(rank)
                print(f"[TARP] WARN rank{rank} penalty not finite: {pen.detach().item()}", flush=True)
            return torch.zeros((), device=dev, dtype=torch.float32)

        return pen


    def estimate_fisher_remote(
        self,
        dataloader,
        tokenizer,
        reward_fn,
        save_dir: str = "/path/to/outputs/anchor",
        rollout=None,
        rollout_sharding_manager=None,
    ):
        import gc
        import json
        import hashlib
        import re
        import time
        import contextlib
        import torch
        import torch.nn.functional as F
        import torch.distributed as dist
        from datetime import datetime, timezone
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        pid = os.getpid()
        dynamo_ctx = contextlib.nullcontext()

        tarp_export_only = bool(getattr(self.config, "tarp_export_only", False))
        ewc_export_only = is_ewc_export_only(self.config)
        if tarp_export_only and ewc_export_only:
            raise ValueError("TARP export-only mode and EWC export-only mode cannot be enabled together.")
        if ewc_export_only:
            return self._estimate_diag_fisher_remote(
                dataloader=dataloader,
                tokenizer=tokenizer,
                reward_fn=reward_fn,
            )

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world = dist.get_world_size() if is_dist else 1

        def _norm_text(s: str) -> str:
            if s is None:
                return ""
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = s.replace("\u00a0", " ")
            s = s.replace("\\n", "\n")
            s = s.replace("’", "'").replace("“", '"').replace("”", '"')
            s = re.sub(r"[ \t]+", " ", s)
            return s

        def locate_line_spans(tokenizer, full_ids, lines, start_cursor=0):
            resp_text = tokenizer.decode(full_ids[start_cursor:], skip_special_tokens=False)
            resp_norm = _norm_text(resp_text)
            spans = []
            cursor_char = 0
            for line_idx, line in enumerate(lines):
                norm_line = _norm_text(line).strip()
                if not norm_line:
                    continue
                pos = resp_norm.find(norm_line, cursor_char)
                if pos == -1:
                    pattern = re.escape(norm_line).replace(r"\ ", r"\s+")
                    match = re.search(pattern, resp_norm[cursor_char:])
                    if match is not None:
                        pos = cursor_char + match.start()
                if pos == -1:
                    continue
                end = pos + len(norm_line)
                prefix = resp_norm[:pos]
                prefix_end = resp_norm[:end]
                start_tok = start_cursor + len(tokenizer.encode(prefix, add_special_tokens=False))
                end_tok = start_cursor + len(tokenizer.encode(prefix_end, add_special_tokens=False))
                spans.append((start_tok, end_tok, norm_line, line_idx))
                cursor_char = end
            return spans

        def _tokenize_simple(s: str):
            s = (s or "").lower()
            s = re.sub(r"\s+", " ", s).strip()
            return {t for t in re.split(r"[^a-z0-9\+\-\*/=<>\._]+", s) if t}

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return float(len(a & b)) / float(max(len(a | b), 1))

        def _text_sem_scores(anchor_lines, other_lines_list, agg="mean"):
            if not anchor_lines:
                return None
            anchor_sets = [_tokenize_simple(x) for x in anchor_lines]
            if not other_lines_list:
                return torch.full((len(anchor_lines),), sem_fallback, device=device, dtype=torch.float32)

            per_other = []
            for other_lines in other_lines_list:
                other_sets = [_tokenize_simple(x) for x in other_lines]
                if not other_sets:
                    continue
                best_scores = []
                for anchor_set in anchor_sets:
                    best = 0.0
                    for other_set in other_sets:
                        best = max(best, _jaccard(anchor_set, other_set))
                    best_scores.append(best)
                per_other.append(torch.tensor(best_scores, device=device, dtype=torch.float32))

            if not per_other:
                return torch.full((len(anchor_lines),), sem_fallback, device=device, dtype=torch.float32)

            stack = torch.stack(per_other, dim=0)
            if agg == "min":
                sem = stack.min(dim=0).values
            elif agg == "max":
                sem = stack.max(dim=0).values
            else:
                sem = stack.mean(dim=0)
            return sem.clamp(0.0, 1.0)

        def _is_fully_correct(score: float, data_source: Optional[str]) -> bool:
            source = "" if data_source is None else str(data_source).lower()
            if "kk" in source:
                return score >= 3.0 - 1e-6
            if source == "code" or "code-r1" in source:
                return score >= 1.1 - 1e-6
            return score >= 1.0 - 1e-6

        def _reason_lines_from_response(resp_text: str) -> List[str]:
            reason = extract_between(resp_text, "<think>", "</think>")
            if reason == resp_text:
                reason = extract_between(resp_text, None, "<answer>")
            return split_reason_sentences(reason)

        def _choose_anchor_index(correct_ids: List[int], reason_lines: List[List[str]], scores: List[float]) -> int:
            if len(correct_ids) <= 1 or anchor_strategy == "first":
                return correct_ids[0]

            if anchor_strategy in {"best", "best_score"}:
                return max(correct_ids, key=lambda idx: (float(scores[idx]), -idx))

            # default: consensus anchor to reduce single-trajectory bias
            best_idx = correct_ids[0]
            best_key = (-1.0, float(scores[best_idx]), -best_idx)
            for idx in correct_ids:
                anchor_lines = reason_lines[idx]
                if not anchor_lines:
                    sem_mean = 0.0
                else:
                    other_lines = [reason_lines[j] for j in correct_ids if j != idx]
                    sem = _text_sem_scores(anchor_lines, other_lines, agg="mean")
                    sem_mean = float(sem.mean().item()) if sem is not None and sem.numel() > 0 else 0.0
                candidate_key = (sem_mean, float(scores[idx]), -idx)
                if candidate_key > best_key:
                    best_key = candidate_key
                    best_idx = idx
            return best_idx

        def _merge_semantic_scores(lexical_scores: Optional[torch.Tensor], embed_scores: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if sem_metric == "embedding":
                return embed_scores if embed_scores is not None else lexical_scores
            if sem_metric == "lexical":
                return lexical_scores

            # hybrid
            if lexical_scores is None:
                return embed_scores
            if embed_scores is None:
                return lexical_scores

            max_len = max(lexical_scores.numel(), embed_scores.numel())
            lex = torch.full((max_len,), sem_fallback, device=lexical_scores.device, dtype=torch.float32)
            emb = torch.full((max_len,), sem_fallback, device=embed_scores.device, dtype=torch.float32)
            lex[: lexical_scores.numel()] = lexical_scores.float()
            emb[: embed_scores.numel()] = embed_scores.float().to(device=lex.device)
            return (sem_hybrid_alpha * lex + (1.0 - sem_hybrid_alpha) * emb).clamp(0.0, 1.0)

        try:
            if hasattr(self.actor_module, "config"):
                self.actor_module.config.use_cache = True
        except Exception:
            pass

        def _check_meta_params(module):
            return [name for name, param in module.named_parameters() if getattr(param, "is_meta", False)]

        if _check_meta_params(self.actor_module):
            raise RuntimeError("[TARP] Meta params found. Cannot proceed.")

        print(f"[TARP-Subspace pid={pid}] Initializing BlockExactSubspaceTARP with orig-view params...", flush=True)
        with self._tarp_full_param_ctx():
            tarp_module = BlockExactSubspaceTARP(
                model=self.actor_module,
                rank_per_block=int(getattr(self.config, "tarp_rank_per_block", 32)),
                block_size=int(getattr(self.config, "tarp_block_size", 2048)),
                proj_dim=int(getattr(self.config, "tarp_proj_dim", 48)),
                normalize_proj_columns=bool(getattr(self.config, "tarp_normalize_proj_columns", True)),
                average_over_proj_dim=bool(getattr(self.config, "tarp_average_over_proj_dim", True)),
                cov_dtype=torch.float32,
                store_dtype=torch.bfloat16,
            )
            tarp_module.snapshot_theta_star()

        self.actor_module.eval()
        device_name = getattr(self, "device_name", "cuda")
        device = get_torch_device().current_device()

        target_valid_samples = int(getattr(self.config, "tarp_target_valid_samples", 64))
        n_traj = int(getattr(self.config, "tarp_n_traj", 4))
        max_new_tokens = int(getattr(self.config, "tarp_max_new_tokens", 1024))
        max_fisher_seq_len = int(getattr(self.config, "tarp_max_fisher_seq_len", 1024))
        skip_overlong_fisher_seq = bool(getattr(self.config, "tarp_skip_overlong_fisher_seq", False))
        tarp_export_only = bool(getattr(self.config, "tarp_export_only", False))
        fisher_fast_export = bool(getattr(self.config, "tarp_fisher_fast_export", True))
        if tarp_export_only and fisher_fast_export:
            export_n_traj_cap = int(getattr(self.config, "tarp_fisher_max_n_traj_export", 1))
            export_new_tokens_cap = int(getattr(self.config, "tarp_fisher_max_new_tokens_export", 128))
            export_seq_len_cap = int(getattr(self.config, "tarp_fisher_max_seq_len_export", 384))
            n_traj = max(1, min(n_traj, export_n_traj_cap))
            max_new_tokens = max(16, min(max_new_tokens, export_new_tokens_cap))
            max_fisher_seq_len = max(64, min(max_fisher_seq_len, export_seq_len_cap))
        min_correct_ratio = float(getattr(self.config, "tarp_min_correct_ratio", 0.5))
        min_correct_count = int(getattr(self.config, "tarp_min_correct_count", 1))
        anchor_strategy = str(getattr(self.config, "tarp_anchor_strategy", "consensus")).strip().lower()
        sem_tau = float(getattr(self.config, "tarp_sem_tau", 0.2))
        sem_power = float(getattr(self.config, "tarp_sem_power", 1.0))
        sem_agg = str(getattr(self.config, "tarp_sem_agg", "mean"))
        sem_metric = str(getattr(self.config, "tarp_sem_metric", "hybrid")).strip().lower()
        if tarp_export_only and fisher_fast_export and sem_metric == "hybrid":
            sem_metric = str(getattr(self.config, "tarp_sem_metric_export", "lexical")).strip().lower()
        sem_hybrid_alpha = float(getattr(self.config, "tarp_sem_hybrid_alpha", 0.5))
        sem_hybrid_alpha = max(0.0, min(1.0, sem_hybrid_alpha))
        sem_fallback = float(getattr(self.config, "tarp_sem_fallback", 0.2))
        sem_fallback = max(0.0, min(1.0, sem_fallback))
        use_stability_weight = str(getattr(self.config, "tarp_use_stability_weight", False)).strip().lower() in {"1", "true", "yes", "y", "on"}
        use_line_type_weight = str(getattr(self.config, "tarp_use_line_type_weight", True)).strip().lower() in {"1", "true", "yes", "y", "on"}
        use_tail_bias = str(getattr(self.config, "tarp_use_tail_bias", True)).strip().lower() in {"1", "true", "yes", "y", "on"}
        line_weight_math_step = float(getattr(self.config, "tarp_line_weight_math_step", 1.0))
        line_weight_search_control = float(getattr(self.config, "tarp_line_weight_search_control", 0.3))
        line_weight_filler = float(getattr(self.config, "tarp_line_weight_filler", 0.05))
        kk_line_weight_logic_step = float(getattr(self.config, "tarp_kk_line_weight_logic_step", 1.0))
        kk_line_weight_consistency_check = float(getattr(self.config, "tarp_kk_line_weight_consistency_check", 0.8))
        kk_line_weight_statement_focus = float(getattr(self.config, "tarp_kk_line_weight_statement_focus", 0.55))
        kk_line_weight_summary = float(getattr(self.config, "tarp_kk_line_weight_summary", 0.3))
        kk_line_weight_filler = float(getattr(self.config, "tarp_kk_line_weight_filler", 0.1))
        tail_head_weight = float(getattr(self.config, "tarp_tail_head_weight", 0.8))
        tail_tail_weight = float(getattr(self.config, "tarp_tail_tail_weight", 1.2))
        kk_tail_head_weight = float(getattr(self.config, "tarp_kk_tail_head_weight", 0.9))
        kk_tail_tail_weight = float(getattr(self.config, "tarp_kk_tail_tail_weight", 1.1))
        if use_stability_weight:
            print(
                "[TARP] INFO tarp_use_stability_weight=True is ignored in this consistency-only TARP weighting mode.",
                flush=True,
            )
        answer_span_weight = float(getattr(self.config, "tarp_answer_span_weight", 1.0))
        answer_tag_weight = float(getattr(self.config, "tarp_answer_tag_weight", 1.0))
        debug_print_limit = int(getattr(self.config, "tarp_debug_print_limit", 0))
        gen_use_cache_cfg = getattr(self.config, "tarp_fisher_generate_use_cache", None)
        if gen_use_cache_cfg is None:
            # In export-only mode, keep KV cache on by default for both speed and stability.
            # Some HF generation paths with use_cache=False can intermittently trigger RoPE length mismatches.
            gen_use_cache = bool(tarp_export_only)
        else:
            gen_use_cache = bool(gen_use_cache_cfg)
        deterministic_export = bool(getattr(self.config, "tarp_fisher_deterministic_export", True)) if (tarp_export_only and fisher_fast_export) else False
        allow_weak_samples_export = bool(getattr(self.config, "tarp_fisher_allow_weak_samples_export", True))
        weak_min_score_export = float(getattr(self.config, "tarp_fisher_weak_min_score_export", 0.0))
        n_traj_effective = 1 if deterministic_export else n_traj

        def _is_none_like(value: Any) -> bool:
            if value is None:
                return True
            text = str(value).strip().lower()
            return text in {"", "none", "null", "nil"}

        def _parse_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            return default

        raw_sync_sampling_cfg = getattr(self.config, "tarp_sync_sampling_across_ranks", None)
        if _is_none_like(raw_sync_sampling_cfg):
            sync_sampling_across_ranks = bool(is_dist and world > 1 and tarp_export_only and not deterministic_export)
        else:
            sync_sampling_across_ranks = _parse_bool(raw_sync_sampling_cfg, default=False)
        try:
            sync_sampling_base_seed = int(getattr(self.config, "tarp_sync_sampling_base_seed", 0))
        except Exception:
            sync_sampling_base_seed = 0
        sampling_signature = (
            f"prompt_hash_seed_v1:"
            f"sync={int(sync_sampling_across_ranks)}:"
            f"seed={sync_sampling_base_seed}:"
            f"det={int(deterministic_export)}:"
            f"traj={n_traj_effective}:"
            f"new={max_new_tokens}:"
            f"cache={int(gen_use_cache)}"
        )

        if tarp_export_only and fisher_fast_export:
            print(
                "[TARP] Fast export mode enabled: "
                f"n_traj={n_traj_effective}, max_new_tokens={max_new_tokens}, max_fisher_seq_len={max_fisher_seq_len}, "
                f"sem_metric={sem_metric}, gen_use_cache={gen_use_cache}, deterministic_export={deterministic_export}",
                flush=True,
            )
        if sync_sampling_across_ranks and rank == 0:
            print(
                f"[TARP] Sync sampling across ranks enabled: base_seed={sync_sampling_base_seed} "
                f"signature={sampling_signature}",
                flush=True,
            )

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        answer_start_ids = tokenizer("<answer>", add_special_tokens=False)["input_ids"]
        answer_end_ids = tokenizer("</answer>", add_special_tokens=False)["input_ids"]
        end_think_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
        exclude_ids = make_exclude_ids(tokenizer, device=device)
        logits_processor = LogitsProcessorList([PerSequenceAnswerStop(answer_end_ids, eos_token_id, pad_token_id)])

        def _get_ids_and_span(full_ids_1d: torch.Tensor, prompt_len: int):
            gen_part = full_ids_1d[prompt_len:]
            rel_end = trim_end_by_pad(gen_part, pad_token_id)
            seq_end = prompt_len + rel_end
            gen_list = full_ids_1d[prompt_len:seq_end].tolist()
            think_end = find_subseq(gen_list, end_think_ids)
            ans_start = find_subseq(gen_list, answer_start_ids)
            ans_end = find_subseq(gen_list, answer_end_ids)
            span_end = seq_end
            if ans_end is not None:
                span_end = min(prompt_len + ans_end, seq_end)
            elif ans_start is not None:
                span_end = seq_end
            elif think_end is not None:
                span_end = min(prompt_len + think_end, seq_end)
            if span_end <= prompt_len + 1:
                return None
            return full_ids_1d[:span_end]

        def _find_tag_bounds(token_ids_list: list[int], tag_ids: list[int], start_cursor: int = 0):
            rel_end = find_subseq(token_ids_list[start_cursor:], tag_ids)
            if rel_end is None:
                return None
            end = start_cursor + rel_end
            return end - len(tag_ids), end

        def _build_aux_spans(token_ids_list: list[int], prompt_start: int):
            aux_spans = []
            think_close = _find_tag_bounds(token_ids_list, end_think_ids, start_cursor=prompt_start)
            answer_open = _find_tag_bounds(token_ids_list, answer_start_ids, start_cursor=prompt_start)
            answer_close = None

            if think_close is not None and answer_tag_weight > 0:
                aux_spans.append(
                    {
                        "start_tok": think_close[0],
                        "end_tok": think_close[1],
                        "label": "</think>",
                        "weight": answer_tag_weight,
                        "allow_excluded": True,
                    }
                )

            if answer_open is not None:
                if answer_tag_weight > 0:
                    aux_spans.append(
                        {
                            "start_tok": answer_open[0],
                            "end_tok": answer_open[1],
                            "label": "<answer>",
                            "weight": answer_tag_weight,
                            "allow_excluded": True,
                        }
                    )
                answer_close = _find_tag_bounds(token_ids_list, answer_end_ids, start_cursor=answer_open[1])
                answer_body_end = answer_close[0] if answer_close is not None else len(token_ids_list)
                if answer_span_weight > 0 and answer_body_end > answer_open[1]:
                    aux_spans.append(
                        {
                            "start_tok": answer_open[1],
                            "end_tok": answer_body_end,
                            "label": "answer_body",
                            "weight": answer_span_weight,
                            "allow_excluded": True,
                        }
                    )

            if answer_close is not None and answer_tag_weight > 0:
                aux_spans.append(
                    {
                        "start_tok": answer_close[0],
                        "end_tok": answer_close[1],
                        "label": "</answer>",
                        "weight": answer_tag_weight,
                        "allow_excluded": True,
                    }
                )

            aux_spans.sort(key=lambda span: (span["start_tok"], span["end_tok"]))
            return aux_spans

        def _is_rope_shape_mismatch(exc: Exception) -> bool:
            msg = str(exc)
            if "apply_rotary_pos_emb" in msg:
                return True
            if "size of tensor a" in msg and "must match the size of tensor b" in msg:
                return True
            return False

        def _generate_with_rope_fallback(
            sample_input_ids: torch.Tensor,
            sample_attention_mask: Optional[torch.Tensor],
            sample_gen_config: Dict[str, Any],
            batch_idx: int,
            sample_idx: int,
            data_source: Optional[str],
            sampling_seed: Optional[int] = None,
        ):
            rng_devices: List[int] = []
            if sample_input_ids.is_cuda and sample_input_ids.device.index is not None:
                rng_devices = [int(sample_input_ids.device.index)]

            def _run_generate(attention_mask_override: Optional[torch.Tensor], runtime_gen_config: Dict[str, Any]):
                if sampling_seed is None:
                    return self.actor_module.generate(
                        input_ids=sample_input_ids,
                        attention_mask=attention_mask_override,
                        **runtime_gen_config,
                    )

                with torch.random.fork_rng(devices=rng_devices, enabled=True):
                    torch.manual_seed(int(sampling_seed))
                    if sample_input_ids.is_cuda:
                        torch.cuda.manual_seed_all(int(sampling_seed))
                    return self.actor_module.generate(
                        input_ids=sample_input_ids,
                        attention_mask=attention_mask_override,
                        **runtime_gen_config,
                    )

            try:
                return _run_generate(sample_attention_mask, sample_gen_config)
            except RuntimeError as exc:
                if not _is_rope_shape_mismatch(exc):
                    raise
                print(
                    f"[TARP Warn pid={pid}] RoPE mismatch during rollout_generate "
                    f"(batch={batch_idx}, sample={sample_idx}, data_source={data_source}). "
                    f"Retry with use_cache=True and attention_mask=None. err={exc}",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                retry_gen_config = dict(sample_gen_config)
                retry_gen_config["use_cache"] = True
                return _run_generate(None, retry_gen_config)

        def _normalize_id_value(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float):
                if math.isnan(value):
                    return None
                if value.is_integer():
                    return str(int(value))
                return f"{value:.12g}"
            if isinstance(value, str):
                text = value.strip()
                if _is_none_like(text) or text.lower() == "nan":
                    return None
                return text
            if isinstance(value, (int, bool)):
                return str(value)
            return None

        def _nested_get(payload: Any, path_parts: Sequence[str]) -> Any:
            cur = payload
            for part in path_parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur

        def _sample_non_tensor_value(non_tensor_batch: Dict[str, Any], key: str, sample_idx: int) -> Any:
            if key not in non_tensor_batch:
                return None
            values = non_tensor_batch[key]
            try:
                return values[sample_idx]
            except Exception:
                return None

        def _extract_sample_id(non_tensor_batch: Dict[str, Any], sample_idx: int, key_fields: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
            for field in key_fields:
                field = str(field).strip()
                if not field:
                    continue
                if "." in field:
                    root, *rest = field.split(".")
                    root_val = _sample_non_tensor_value(non_tensor_batch, root, sample_idx)
                    if root_val is not None:
                        normalized = _normalize_id_value(_nested_get(root_val, rest))
                        if normalized is not None:
                            return normalized, field
                    continue

                direct_val = _sample_non_tensor_value(non_tensor_batch, field, sample_idx)
                normalized_direct = _normalize_id_value(direct_val)
                if normalized_direct is not None:
                    return normalized_direct, field

                for holder in ("extra_info", "meta", "reward_model"):
                    holder_val = _sample_non_tensor_value(non_tensor_batch, holder, sample_idx)
                    if holder_val is None:
                        continue
                    nested_val = _nested_get(holder_val, [field])
                    normalized_nested = _normalize_id_value(nested_val)
                    if normalized_nested is not None:
                        return normalized_nested, f"{holder}.{field}"
            return None, None

        def _prompt_ids_for_hash(input_ids_2d: torch.Tensor, attention_mask_2d: Optional[torch.Tensor], raw_prompt_value: Any) -> List[int]:
            prompt_ids: List[int] = []
            raw_ids: Optional[List[int]] = None

            if raw_prompt_value is not None:
                if isinstance(raw_prompt_value, torch.Tensor):
                    raw_ids = [int(x) for x in raw_prompt_value.detach().cpu().reshape(-1).tolist()]
                elif isinstance(raw_prompt_value, np.ndarray):
                    raw_ids = [int(x) for x in raw_prompt_value.reshape(-1).tolist()]
                elif isinstance(raw_prompt_value, (list, tuple)):
                    raw_ids = []
                    for token in raw_prompt_value:
                        try:
                            raw_ids.append(int(token))
                        except Exception:
                            raw_ids = None
                            break

            if raw_ids:
                return raw_ids

            if attention_mask_2d is not None:
                valid_len = int(attention_mask_2d[0].sum().item())
                valid_len = max(1, min(valid_len, int(input_ids_2d.size(1))))
                prompt_ids = [int(x) for x in input_ids_2d[0, -valid_len:].detach().cpu().tolist()]
            else:
                prompt_ids = [int(x) for x in input_ids_2d[0].detach().cpu().tolist()]
            return prompt_ids

        def _hash_prompt_ids(prompt_ids: Sequence[int]) -> str:
            payload = ",".join(str(int(x)) for x in prompt_ids).encode("utf-8")
            return hashlib.sha1(payload).hexdigest()[:16]

        def _build_cache_key(data_source: Optional[str], sample_id: Optional[str], prompt_hash: str) -> str:
            source_key = str(data_source if data_source is not None else "UNKNOWN").strip().lower() or "unknown"
            sample_part = sample_id if sample_id is not None else f"hash:{prompt_hash}"
            return f"{source_key}::{sample_part}"

        def _derive_sync_sampling_seed(prompt_hash: str, data_source: Optional[str]) -> int:
            payload = (
                f"{sync_sampling_base_seed}|"
                f"{str(data_source if data_source is not None else 'UNKNOWN').strip().lower()}|"
                f"{prompt_hash}"
            ).encode("utf-8")
            return int(hashlib.sha1(payload).hexdigest()[:16], 16)

        raw_cache_key_fields = getattr(self.config, "tarp_reasoning_cache_key_fields", "index,task_id,id,sample_id,uid")
        if isinstance(raw_cache_key_fields, (list, tuple)):
            cache_key_fields = [str(item).strip() for item in raw_cache_key_fields if str(item).strip()]
        else:
            cache_key_fields = [item.strip() for item in str(raw_cache_key_fields).split(",") if item.strip()]
        if not cache_key_fields:
            cache_key_fields = ["index", "task_id", "id", "sample_id", "uid"]

        cache_version = str(getattr(self.config, "tarp_reasoning_cache_version", "v1")).strip() or "v1"
        tokenizer_id = str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))
        model_path_for_cache = str(os.environ.get("BASE_MODEL", "") or os.environ.get("INITIAL_MODEL", "")).strip()
        if model_path_for_cache == "":
            model_path_for_cache = str(getattr(getattr(self.config, "model", None), "path", "")).strip()

        raw_cache_enabled = getattr(self.config, "tarp_reasoning_cache_enabled", None)
        if _is_none_like(raw_cache_enabled):
            reasoning_cache_enabled = bool(tarp_export_only)
        else:
            reasoning_cache_enabled = _parse_bool(raw_cache_enabled, default=False)

        reasoning_cache_path = None
        reasoning_cache_fp = None
        reasoning_cache_index: Dict[str, Dict[str, Any]] = {}
        cache_hit_count = 0
        cache_miss_count = 0
        cache_write_count = 0
        cache_invalid_count = 0

        def _load_reasoning_cache_index(path: str) -> tuple[Dict[str, Dict[str, Any]], int, int]:
            index: Dict[str, Dict[str, Any]] = {}
            total_lines = 0
            invalid_lines = 0
            if not os.path.exists(path):
                return index, total_lines, invalid_lines
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    for raw_line in fp:
                        total_lines += 1
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            invalid_lines += 1
                            continue
                        if not isinstance(record, dict):
                            invalid_lines += 1
                            continue
                        key = record.get("cache_key")
                        if not isinstance(key, str) or key == "":
                            invalid_lines += 1
                            continue
                        index[key] = record
            except Exception as exc:
                print(f"[TARP Cache] Failed to read cache index from {path}: {exc}", flush=True)
                return {}, total_lines, max(invalid_lines, 1)
            return index, total_lines, invalid_lines

        def _append_reasoning_cache_record(record: Dict[str, Any]) -> bool:
            nonlocal reasoning_cache_fp
            if reasoning_cache_fp is None:
                return False
            try:
                reasoning_cache_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                reasoning_cache_fp.flush()
                return True
            except Exception as exc:
                print(f"[TARP Cache] Failed to append cache record: {exc}", flush=True)
                return False

        def _validate_cache_meta(record: Dict[str, Any]) -> bool:
            meta = record.get("meta")
            if not isinstance(meta, dict):
                return False
            record_version = str(meta.get("cache_version", "")).strip()
            if record_version and record_version != cache_version:
                return False
            record_tokenizer = str(meta.get("tokenizer_id", "")).strip()
            if record_tokenizer and record_tokenizer != tokenizer_id:
                return False
            record_model_path = str(meta.get("model_path", "")).strip()
            if record_model_path and model_path_for_cache and record_model_path != model_path_for_cache:
                return False
            record_sampling_signature = str(meta.get("sampling_signature", "")).strip()
            if record_sampling_signature != sampling_signature:
                return False
            return True

        def _restore_cached_reasoning(
            record: Dict[str, Any],
            *,
            prompt_len: int,
            data_source: Optional[str],
            prompt_hash: str,
        ) -> Optional[Dict[str, Any]]:
            if not isinstance(record, dict):
                return None
            if str(record.get("prompt_hash", "")).strip() != str(prompt_hash):
                return None
            if not _validate_cache_meta(record):
                return None

            trajectories = record.get("trajectories")
            if not isinstance(trajectories, list) or len(trajectories) == 0:
                return None

            try:
                n_traj_record = int(record.get("n_traj_effective", len(trajectories)))
            except Exception:
                return None
            if n_traj_record != n_traj_effective or len(trajectories) != n_traj_record:
                return None

            all_full_ids: List[torch.Tensor] = []
            all_resp_str: List[str] = []
            all_scores: List[float] = []
            all_reason_lines: List[List[str]] = []
            for traj in trajectories:
                if not isinstance(traj, dict):
                    return None
                full_ids = traj.get("full_ids")
                if not isinstance(full_ids, list) or len(full_ids) <= prompt_len:
                    return None
                try:
                    full_ids_tensor = torch.tensor([int(x) for x in full_ids], dtype=torch.long, device="cpu")
                except Exception:
                    return None
                response = traj.get("response", None)
                if response is None:
                    response = tokenizer.decode(full_ids_tensor[prompt_len:], skip_special_tokens=True)
                if not isinstance(response, str):
                    response = str(response)
                score = traj.get("score", None)
                try:
                    score_value = float(score)
                except Exception:
                    return None
                reason_lines = traj.get("reason_lines")
                if not isinstance(reason_lines, list) or any(not isinstance(line, str) for line in reason_lines):
                    reason_lines = _reason_lines_from_response(response)
                all_full_ids.append(full_ids_tensor)
                all_resp_str.append(response)
                all_scores.append(score_value)
                all_reason_lines.append(reason_lines)

            try:
                correct_indices = sorted({int(idx) for idx in record.get("correct_indices", [])})
            except Exception:
                return None
            if any(idx < 0 or idx >= len(trajectories) for idx in correct_indices):
                return None

            anchor_idx_raw = record.get("anchor_index", None)
            if anchor_idx_raw is None:
                anchor_idx = None
            else:
                try:
                    anchor_idx = int(anchor_idx_raw)
                except Exception:
                    return None
                if anchor_idx < 0 or anchor_idx >= len(trajectories):
                    return None
                if correct_indices and anchor_idx not in correct_indices:
                    return None

            if not correct_indices:
                # Keep compatibility with historical records where is_correct was saved but correct_indices was omitted.
                derived_correct = []
                for idx, traj in enumerate(trajectories):
                    marker = traj.get("is_correct", None)
                    if marker is None:
                        marker = _is_fully_correct(all_scores[idx], data_source)
                    if bool(marker):
                        derived_correct.append(idx)
                correct_indices = derived_correct

            return {
                "all_full_ids": all_full_ids,
                "all_resp_str": all_resp_str,
                "all_scores": all_scores,
                "all_reason_lines": all_reason_lines,
                "correct_indices": correct_indices,
                "anchor_index": anchor_idx,
            }

        def _build_reasoning_cache_record(
            *,
            cache_key: str,
            data_source: Optional[str],
            sample_id: Optional[str],
            sample_id_field: Optional[str],
            prompt_hash: str,
            all_full_ids: Sequence[torch.Tensor],
            all_resp_str: Sequence[str],
            all_scores: Sequence[float],
            all_reason_lines: Sequence[Sequence[str]],
            correct_indices: Sequence[int],
            anchor_index: Optional[int],
        ) -> Dict[str, Any]:
            correct_set = {int(idx) for idx in correct_indices}
            trajectories: List[Dict[str, Any]] = []
            for idx, full_ids in enumerate(all_full_ids):
                full_ids_cpu = full_ids.detach().cpu() if isinstance(full_ids, torch.Tensor) else torch.tensor(full_ids, dtype=torch.long)
                ids_list = [int(x) for x in full_ids_cpu.tolist()]
                response_text = all_resp_str[idx] if idx < len(all_resp_str) else tokenizer.decode(full_ids_cpu[prompt_len:], skip_special_tokens=True)
                score_value = float(all_scores[idx]) if idx < len(all_scores) else 0.0
                reason_lines = list(all_reason_lines[idx]) if idx < len(all_reason_lines) else _reason_lines_from_response(response_text)
                trajectories.append(
                    {
                        "full_ids": ids_list,
                        "response": response_text,
                        "score": score_value,
                        "is_correct": bool(idx in correct_set),
                        "reason_lines": reason_lines,
                    }
                )

            return {
                "cache_key": cache_key,
                "data_source": data_source,
                "sample_id": sample_id,
                "sample_id_field": sample_id_field,
                "prompt_hash": prompt_hash,
                "n_traj_effective": int(n_traj_effective),
                "correct_indices": [int(idx) for idx in sorted(correct_set)],
                "anchor_index": (None if anchor_index is None else int(anchor_index)),
                "trajectories": trajectories,
                "meta": {
                    "tokenizer_id": tokenizer_id,
                    "model_path": model_path_for_cache,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "cache_version": cache_version,
                    "sampling_signature": sampling_signature,
                },
            }

        if reasoning_cache_enabled:
            configured_cache_path = getattr(self.config, "tarp_reasoning_cache_path", None)
            if _is_none_like(configured_cache_path):
                tarp_dir_cfg = getattr(self.config, "tarp_dir", None)
                tarp_tag_cfg = getattr(self.config, "tarp_tag", None)
                if not _is_none_like(tarp_dir_cfg) and not _is_none_like(tarp_tag_cfg):
                    reasoning_cache_path = os.path.join(
                        str(tarp_dir_cfg),
                        f"{str(tarp_tag_cfg)}__reasoning_cache__rank-{rank}-of-{world}.jsonl",
                    )
                else:
                    reasoning_cache_path = os.path.join(str(save_dir), f"TARP__reasoning_cache__rank-{rank}-of-{world}.jsonl")
            else:
                reasoning_cache_path = os.path.expanduser(str(configured_cache_path))

            cache_dir = os.path.dirname(reasoning_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            reasoning_cache_index, loaded_lines, loaded_invalid = _load_reasoning_cache_index(reasoning_cache_path)
            if loaded_lines > 0:
                print(
                    f"[TARP Cache] Indexed {len(reasoning_cache_index)} records from {reasoning_cache_path} "
                    f"(raw_lines={loaded_lines}, invalid_lines={loaded_invalid})",
                    flush=True,
                )
            try:
                reasoning_cache_fp = open(reasoning_cache_path, "a", encoding="utf-8")
            except Exception as exc:
                print(f"[TARP Cache] Failed to open cache file for append ({reasoning_cache_path}): {exc}", flush=True)
                reasoning_cache_fp = None

        valid_samples_count = 0
        attempted_samples_count = 0
        skip_reason_counts = {
            "no_correct_traj": 0,
            "insufficient_correct_traj": 0,
            "anchor_span_empty": 0,
            "no_anchor_or_aux_spans": 0,
            "sem_scores_none_with_anchor_spans": 0,
            "zero_token_weight": 0,
            "zero_loss": 0,
        }
        progress_log_interval = max(1, int(getattr(self.config, "tarp_progress_log_interval", 20)))
        skip_detail_log_limit = max(0, int(getattr(self.config, "tarp_skip_detail_log_limit", 200)))
        stage_log_interval = max(1, int(getattr(self.config, "tarp_stage_log_interval", 10)))
        skip_detail_logged = 0
        last_stage = "init"
        data_source_observed = []

        def _format_skip_breakdown():
            active = [f"{name}={count}" for name, count in skip_reason_counts.items() if count > 0]
            return ", ".join(active) if active else "none"

        def _log_progress(force: bool = False):
            if attempted_samples_count <= 0:
                return
            if not force and (attempted_samples_count % progress_log_interval != 0):
                return
            skipped_total = max(0, attempted_samples_count - valid_samples_count)
            success_rate = float(valid_samples_count) / float(max(attempted_samples_count, 1))
            print(
                f"[TARP Progress pid={pid}] attempts={attempted_samples_count} "
                f"valid={valid_samples_count}/{target_valid_samples} skipped={skipped_total} "
                f"success_rate={success_rate:.4f} last_stage={last_stage} "
                f"skip_breakdown={_format_skip_breakdown()} "
                f"cache_hit={cache_hit_count} cache_miss={cache_miss_count} "
                f"cache_write={cache_write_count} cache_invalid={cache_invalid_count}",
                flush=True,
            )

        def _log_stage(stage: str, batch_idx: int, sample_idx: int, extra: str = "", force: bool = False):
            nonlocal last_stage
            last_stage = stage
            if not force and attempted_samples_count > 3 and (attempted_samples_count % stage_log_interval != 0):
                return
            msg = (
                f"[TARP Stage pid={pid}] stage={stage} batch={batch_idx} sample={sample_idx} "
                f"attempt={attempted_samples_count} valid={valid_samples_count}/{target_valid_samples}"
            )
            if extra:
                msg = f"{msg} | {extra}"
            print(msg, flush=True)

        def _log_skip(reason: str, batch_idx: int, sample_idx: int, extra: str = ""):
            nonlocal skip_detail_logged
            if reason not in skip_reason_counts:
                skip_reason_counts[reason] = 0
            skip_reason_counts[reason] += 1
            if skip_detail_log_limit == 0:
                return
            if skip_detail_logged < skip_detail_log_limit:
                msg = (
                    f"[TARP Skip pid={pid}] reason={reason} batch={batch_idx} sample={sample_idx} "
                    f"attempt={attempted_samples_count} valid={valid_samples_count}/{target_valid_samples}"
                )
                if extra:
                    msg = f"{msg} | {extra}"
                print(msg, flush=True)
                skip_detail_logged += 1
            elif skip_detail_logged == skip_detail_log_limit:
                print(
                    f"[TARP Skip pid={pid}] detail log limit reached ({skip_detail_log_limit}); "
                    "switching to periodic progress logs only.",
                    flush=True,
                )
                skip_detail_logged += 1

        with dynamo_ctx, torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            print("length", len(dataloader))
            for batch_idx, batch_dict in enumerate(dataloader):
                print(
                    f"[TARP pid={pid}] Processing Batch {batch_idx} | Current Valid Samples: {valid_samples_count}/{target_valid_samples}",
                    flush=True,
                )
                if valid_samples_count >= target_valid_samples:
                    print(f"[TARP] Reached target {target_valid_samples}", flush=True)
                    break

                original_batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_keys = []
                if "raw_prompt_ids" in original_batch.non_tensor_batch:
                    non_tensor_keys.append("raw_prompt_ids")
                gen_batch = original_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_keys)
                batch_data = {**gen_batch.batch.to(device=device), **gen_batch.non_tensor_batch}
                batch_size = int(batch_data["input_ids"].size(0))

                for sample_idx in range(batch_size):
                    if valid_samples_count >= target_valid_samples:
                        break
                    attempted_samples_count += 1

                    input_ids = batch_data["input_ids"][sample_idx : sample_idx + 1]
                    attention_mask = batch_data["attention_mask"][sample_idx : sample_idx + 1]
                    raw_prompt_value = None
                    if "raw_prompt_ids" in batch_data:
                        raw_prompt_value = _sample_non_tensor_value(batch_data, "raw_prompt_ids", sample_idx)
                    prompt_ids_for_hash = _prompt_ids_for_hash(input_ids, attention_mask, raw_prompt_value)
                    prompt_hash = _hash_prompt_ids(prompt_ids_for_hash)

                    data_source = None
                    if "data_source" in original_batch.non_tensor_batch:
                        data_source = str(original_batch.non_tensor_batch["data_source"][sample_idx])
                        data_source_observed.append(data_source)

                    ground_truth = None
                    if "reward_model" in original_batch.non_tensor_batch:
                        reward_item = original_batch.non_tensor_batch["reward_model"][sample_idx]
                        if isinstance(reward_item, dict):
                            ground_truth = reward_item.get("ground_truth", None)

                    extra_info = None
                    if "extra_info" in original_batch.non_tensor_batch:
                        extra_info = original_batch.non_tensor_batch["extra_info"][sample_idx]

                    sample_id, sample_id_field = _extract_sample_id(original_batch.non_tensor_batch, sample_idx, cache_key_fields)
                    cache_key = _build_cache_key(data_source, sample_id, prompt_hash)

                    # Normalize per-sample prompt before generation / cached replay.
                    # In TARP export-only path we process one sample at a time; stripping left pads avoids
                    # occasional RoPE shape mismatches with remove-padding kernels.
                    if attention_mask is not None:
                        valid_prompt_len = int(attention_mask[0].sum().item())
                        valid_prompt_len = max(1, min(valid_prompt_len, int(input_ids.size(1))))
                        if valid_prompt_len < int(input_ids.size(1)):
                            input_ids = input_ids[:, -valid_prompt_len:].contiguous()
                            attention_mask = attention_mask[:, -valid_prompt_len:].contiguous()
                        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

                    prompt_len = int(input_ids.size(1))
                    prompt_len_base = prompt_len

                    all_full_ids = []
                    all_resp_str = []
                    all_scores = []
                    all_reason_lines = []
                    correct_indices = []
                    anchor_k: Optional[int] = None

                    cache_write_done = False
                    cache_payload_ready = False

                    def _maybe_write_cache(anchor_index: Optional[int]) -> None:
                        nonlocal cache_write_done, cache_payload_ready, cache_write_count
                        if cache_write_done or (not cache_payload_ready):
                            return
                        if not reasoning_cache_enabled:
                            return
                        record = _build_reasoning_cache_record(
                            cache_key=cache_key,
                            data_source=data_source,
                            sample_id=sample_id,
                            sample_id_field=sample_id_field,
                            prompt_hash=prompt_hash,
                            all_full_ids=all_full_ids,
                            all_resp_str=all_resp_str,
                            all_scores=all_scores,
                            all_reason_lines=all_reason_lines,
                            correct_indices=correct_indices,
                            anchor_index=anchor_index,
                        )
                        if _append_reasoning_cache_record(record):
                            reasoning_cache_index[cache_key] = record
                            cache_write_done = True
                            cache_write_count += 1

                    cache_hit_payload = None
                    if reasoning_cache_enabled:
                        record = reasoning_cache_index.get(cache_key)
                        if record is not None:
                            cache_hit_payload = _restore_cached_reasoning(
                                record,
                                prompt_len=prompt_len,
                                data_source=data_source,
                                prompt_hash=prompt_hash,
                            )
                            if cache_hit_payload is None:
                                cache_invalid_count += 1
                                cache_miss_count += 1
                                _log_stage("reasoning_cache:invalid", batch_idx, sample_idx, extra=f"cache_key={cache_key}")
                            else:
                                cache_hit_count += 1
                        else:
                            cache_miss_count += 1

                    if cache_hit_payload is not None:
                        all_full_ids = cache_hit_payload["all_full_ids"]
                        all_resp_str = cache_hit_payload["all_resp_str"]
                        all_scores = cache_hit_payload["all_scores"]
                        all_reason_lines = cache_hit_payload["all_reason_lines"]
                        correct_indices = cache_hit_payload["correct_indices"]
                        anchor_k = cache_hit_payload["anchor_index"]
                        _log_stage("reasoning_cache:hit", batch_idx, sample_idx, extra=f"cache_key={cache_key}")
                    else:
                        _log_stage("rollout_generate:start", batch_idx, sample_idx, extra=f"data_source={data_source}")
                        rollout_t0 = time.perf_counter()
                        sampling_seed = _derive_sync_sampling_seed(prompt_hash, data_source) if sync_sampling_across_ranks else None
                        with self._tarp_full_param_ctx(), torch.no_grad():
                            gen_config = {
                                "max_new_tokens": max_new_tokens,
                                "temperature": 0.0 if deterministic_export else 1.0,
                                "top_k": -1 if deterministic_export else 50,
                                "top_p": 1.0 if deterministic_export else 0.7,
                                "do_sample": (not deterministic_export),
                                "num_beams": 1,
                                "num_return_sequences": n_traj_effective,
                                "pad_token_id": pad_token_id,
                                "eos_token_id": eos_token_id,
                                "use_cache": gen_use_cache,
                                "return_dict_in_generate": True,
                                "logits_processor": logits_processor,
                            }
                            outputs = _generate_with_rope_fallback(
                                sample_input_ids=input_ids,
                                sample_attention_mask=attention_mask,
                                sample_gen_config=gen_config,
                                batch_idx=batch_idx,
                                sample_idx=sample_idx,
                                data_source=data_source,
                                sampling_seed=sampling_seed,
                            )
                            seqs = outputs.sequences.detach().cpu()
                            del outputs
                        rollout_dt = time.perf_counter() - rollout_t0
                        _log_stage(
                            "rollout_generate:done",
                            batch_idx,
                            sample_idx,
                            extra=f"dt={rollout_dt:.2f}s n_traj={n_traj_effective}",
                        )

                        score_t0 = time.perf_counter()
                        for traj_idx in range(n_traj_effective):
                            full_ids = seqs[traj_idx]
                            gen_ids = full_ids[prompt_len:]
                            resp_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
                            all_full_ids.append(full_ids)
                            all_resp_str.append(resp_str)
                            all_reason_lines.append(_reason_lines_from_response(resp_str))

                            score = self._score_response(reward_fn, data_source, resp_str, ground_truth, extra_info)
                            all_scores.append(score)
                            if _is_fully_correct(score, data_source):
                                correct_indices.append(traj_idx)
                        score_dt = time.perf_counter() - score_t0
                        _log_stage(
                            "reward_scoring:done",
                            batch_idx,
                            sample_idx,
                            extra=f"dt={score_dt:.2f}s correct={len(correct_indices)}/{n_traj_effective}",
                        )
                        cache_payload_ready = True

                    if not correct_indices:
                        if tarp_export_only and allow_weak_samples_export and all_scores:
                            best_idx = int(max(range(len(all_scores)), key=lambda idx: float(all_scores[idx])))
                            best_score = float(all_scores[best_idx])
                            if best_score >= weak_min_score_export:
                                correct_indices = [best_idx]
                                print(
                                    f"[TARP] Export fallback: no fully-correct traj; "
                                    f"use best traj idx={best_idx} score={best_score:.4f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[TARP] Export fallback skipped: best score {best_score:.4f} < {weak_min_score_export:.4f}",
                                    flush=True,
                                )
                        if not correct_indices:
                            _maybe_write_cache(anchor_index=None)
                            score_str = ",".join(f"{float(s):.3f}" for s in all_scores)
                            max_score = max((float(s) for s in all_scores), default=float("nan"))
                            _log_skip(
                                "no_correct_traj",
                                batch_idx,
                                sample_idx,
                                extra=f"data_source={data_source} max_score={max_score:.3f} scores=[{score_str}]",
                            )
                            _log_progress()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                    required_correct = max(min_correct_count, int(math.ceil(float(n_traj_effective) * min_correct_ratio)))
                    required_correct = min(required_correct, n_traj_effective)
                    if len(correct_indices) < required_correct:
                        _maybe_write_cache(anchor_index=None)
                        score_str = ",".join(f"{float(s):.3f}" for s in all_scores)
                        _log_skip(
                            "insufficient_correct_traj",
                            batch_idx,
                            sample_idx,
                            extra=f"correct={len(correct_indices)} required={required_correct} scores=[{score_str}]",
                        )
                        _log_progress()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if anchor_k is None:
                        anchor_k = _choose_anchor_index(correct_indices, all_reason_lines, all_scores)
                    _maybe_write_cache(anchor_index=anchor_k)
                    anchor_ids_cpu = _get_ids_and_span(all_full_ids[anchor_k], prompt_len)
                    if anchor_ids_cpu is None:
                        _log_skip(
                            "anchor_span_empty",
                            batch_idx,
                            sample_idx,
                            extra=f"anchor_idx={anchor_k} prompt_len={prompt_len}",
                        )
                        _log_progress()
                        continue

                    if anchor_ids_cpu.numel() > max_fisher_seq_len:
                        old_len = int(anchor_ids_cpu.numel())
                        if skip_overlong_fisher_seq:
                            _log_skip(
                                "anchor_seq_too_long",
                                batch_idx,
                                sample_idx,
                                extra=f"anchor_idx={anchor_k} seq_len={old_len} max={int(max_fisher_seq_len)} prompt_len={prompt_len}",
                            )
                            _log_progress()
                            continue
                        offset = old_len - int(max_fisher_seq_len)
                        anchor_ids_cpu = anchor_ids_cpu[offset:]
                        prompt_len = max(0, prompt_len - offset)

                    anchor_ids_gpu = anchor_ids_cpu.unsqueeze(0).to(device, non_blocking=True)
                    anchor_lines = all_reason_lines[anchor_k]
                    anchor_spans = locate_line_spans(tokenizer, anchor_ids_cpu.tolist(), anchor_lines, start_cursor=prompt_len)
                    aux_spans = _build_aux_spans(anchor_ids_cpu.tolist(), prompt_len)
                    if not anchor_spans and not aux_spans:
                        _log_skip(
                            "no_anchor_or_aux_spans",
                            batch_idx,
                            sample_idx,
                            extra=f"anchor_lines={len(anchor_lines)}",
                        )
                        _log_progress()
                        continue

                    other_lines_list = [all_reason_lines[idx] for idx in correct_indices if idx != anchor_k]

                    lexical_sem_scores = _text_sem_scores(anchor_lines, other_lines_list, agg=sem_agg)
                    embed_sem_scores = None
                    if sem_metric in {"embedding", "hybrid"} and anchor_lines:
                        embed_sem_scores = torch.full((len(anchor_lines),), sem_fallback, device=device, dtype=torch.float32)
                        anchor_embed_spans = [(s, e, txt) for (s, e, txt, _) in anchor_spans]
                        if anchor_embed_spans:
                            with self._tarp_full_param_ctx(), torch.no_grad():
                                anchor_embs = encode_sentence_embeds(
                                    self.actor_module,
                                    anchor_ids_gpu,
                                    anchor_embed_spans,
                                    exclude_ids=exclude_ids,
                                )
                                if anchor_embs is not None and anchor_embs.size(0) == len(anchor_embed_spans):
                                    other_embs_list = []
                                    for traj_idx in correct_indices:
                                        if traj_idx == anchor_k:
                                            continue

                                        other_ids_cpu = _get_ids_and_span(all_full_ids[traj_idx], prompt_len_base)
                                        if other_ids_cpu is None:
                                            other_embs_list.append(None)
                                            continue

                                        other_prompt_len = prompt_len_base
                                        if other_ids_cpu.numel() > max_fisher_seq_len:
                                            old_other_len = int(other_ids_cpu.numel())
                                            if skip_overlong_fisher_seq:
                                                other_embs_list.append(None)
                                                continue
                                            offset = old_other_len - int(max_fisher_seq_len)
                                            other_ids_cpu = other_ids_cpu[offset:]
                                            other_prompt_len = max(0, other_prompt_len - offset)

                                        other_lines = all_reason_lines[traj_idx]
                                        other_spans = locate_line_spans(tokenizer, other_ids_cpu.tolist(), other_lines, start_cursor=other_prompt_len)
                                        if not other_spans:
                                            other_embs_list.append(None)
                                            continue

                                        other_ids_gpu = other_ids_cpu.unsqueeze(0).to(device, non_blocking=True)
                                        other_embed_spans = [(s, e, txt) for (s, e, txt, _) in other_spans]
                                        other_embs = encode_sentence_embeds(
                                            self.actor_module,
                                            other_ids_gpu,
                                            other_embed_spans,
                                            exclude_ids=exclude_ids,
                                        )
                                        other_embs_list.append(other_embs)
                                        del other_ids_gpu

                                    has_valid_other_embs = any(
                                        embs is not None and embs.numel() > 0 for embs in other_embs_list
                                    )
                                    if has_valid_other_embs:
                                        span_embed_scores = semantic_consistency(anchor_embs, other_embs_list, agg=sem_agg)
                                    else:
                                        span_embed_scores = torch.full(
                                            (len(anchor_embed_spans),),
                                            sem_fallback,
                                            device=anchor_ids_gpu.device,
                                            dtype=torch.float32,
                                        )
                                    for span_idx, (_, _, _, line_idx) in enumerate(anchor_spans):
                                        if line_idx < embed_sem_scores.numel() and span_idx < span_embed_scores.numel():
                                            embed_sem_scores[line_idx] = span_embed_scores[span_idx]

                    sem_scores = _merge_semantic_scores(lexical_sem_scores, embed_sem_scores)
                    if sem_scores is None:
                        if anchor_spans:
                            _log_skip(
                                "sem_scores_none_with_anchor_spans",
                                batch_idx,
                                sample_idx,
                                extra=f"anchor_spans={len(anchor_spans)}",
                            )
                            _log_progress()
                            continue
                        sem_scores = torch.empty(0, dtype=torch.float32, device=device)
                    if len(correct_indices) < 2 and sem_scores.numel() > 0:
                        sem_scores = torch.full_like(sem_scores, sem_fallback)
                    current_dataset_key = get_tarp_dataset_key(self.config, data_source=data_source)
                    if current_dataset_key == "KK":
                        line_type_weights = {
                            "logic_step": kk_line_weight_logic_step,
                            "consistency_check": kk_line_weight_consistency_check,
                            "statement_focus": kk_line_weight_statement_focus,
                            "summary": kk_line_weight_summary,
                            "filler": kk_line_weight_filler,
                        }
                        tail_head_weight_current = kk_tail_head_weight
                        tail_tail_weight_current = kk_tail_tail_weight
                    else:
                        line_type_weights = {
                            "math_step": line_weight_math_step,
                            "search_control": line_weight_search_control,
                            "filler": line_weight_filler,
                        }
                        tail_head_weight_current = tail_head_weight
                        tail_tail_weight_current = tail_tail_weight
                    line_weights, line_weight_profile = compute_tarp_line_weight_profile(
                        anchor_lines,
                        sem_scores=sem_scores,
                        dataset_key=current_dataset_key,
                        apply_sem_margin=(len(correct_indices) >= 2),
                        sem_tau=sem_tau,
                        sem_power=sem_power,
                        use_line_type_weight=use_line_type_weight,
                        use_tail_bias=use_tail_bias,
                        line_type_weights=line_type_weights,
                        tail_head_weight=tail_head_weight_current,
                        tail_tail_weight=tail_tail_weight_current,
                        sem_fallback=sem_fallback,
                        device=device,
                    )
                    line_profile_by_idx = {int(item["line_idx"]): item for item in line_weight_profile}

                    if debug_print_limit > 0 and valid_samples_count < debug_print_limit:
                        question_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        print("correct_indices:", correct_indices, "len=", len(correct_indices))
                        print("other_lines_list lens:", [len(x) for x in other_lines_list])
                        print(f"\n[TARP Debug] Sample {valid_samples_count + 1}", flush=True)
                        print(f"Question : {question_str}")
                        print(f"GroundTruth: {str(ground_truth)}")
                        print(f"batch {batch_idx} {'Final W':<8} | {'Raw Score':<9} | {'Type':<14} | {'Reasoning Step (Sentence)'}")
                        print("-" * 80)
                        for (start_tok, end_tok, text_content, line_idx) in anchor_spans:
                            profile = line_profile_by_idx.get(int(line_idx), None)
                            if profile is None:
                                continue
                            clean_text = text_content.replace("\n", " ").strip()
                            if len(clean_text) > 60:
                                clean_text = clean_text[:57] + "..."
                            print(
                                f"{float(profile['final_weight']):.4f}   | {float(profile['sem_raw']):.4f}     | "
                                f"{str(profile['line_type']):<14} | {clean_text}"
                            )
                        for aux_span in aux_spans:
                            label = aux_span["label"]
                            print(f"{aux_span['weight']:.4f}   | fixed     | {'aux':<14} | [{label}]")
                        print("-" * 80 + "\n", flush=True)

                    _log_stage("fisher_backward:start", batch_idx, sample_idx)
                    backward_t0 = time.perf_counter()
                    self.actor_module.zero_grad(set_to_none=True)
                    out = self.actor_module(
                        input_ids=anchor_ids_gpu,
                        attention_mask=torch.ones_like(anchor_ids_gpu),
                        use_cache=False,
                    )
                    logits = out.logits[:, :-1, :]
                    labels = anchor_ids_gpu[:, 1:]
                    nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="none",
                    ).view_as(labels).float()
                    valid_start_idx = max(0, prompt_len - 1)
                    token_weight = build_tarp_token_weight_from_lines(
                        anchor_ids=anchor_ids_gpu,
                        nll=nll,
                        anchor_spans=anchor_spans,
                        line_weights=line_weights,
                        exclude_ids=exclude_ids,
                        aux_spans=aux_spans,
                        valid_start_idx=valid_start_idx,
                    )
                    if not (token_weight > 0).any().item():
                        _log_skip(
                            "zero_token_weight",
                            batch_idx,
                            sample_idx,
                            extra=f"anchor_spans={len(anchor_spans)} aux_spans={len(aux_spans)}",
                        )
                        _log_stage(
                            "fisher_backward:skip_zero_token_weight",
                            batch_idx,
                            sample_idx,
                            extra=f"dt={time.perf_counter() - backward_t0:.2f}s",
                        )
                        _log_progress()
                        del out, logits, labels, nll, token_weight, anchor_ids_gpu
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    valid_count = (token_weight > 0).float().sum().clamp(min=1.0)
                    loss = (nll * token_weight).sum() / valid_count
                    if float(loss.detach().item()) != 0.0:
                        loss.backward()
                        with self._tarp_full_param_ctx(with_grads=True):
                            tarp_module.accumulate_fisher_after_backward()
                        valid_samples_count += 1
                        print(f"[TARP] Accumulated sample {valid_samples_count}", flush=True)
                        _log_stage(
                            "fisher_backward:done",
                            batch_idx,
                            sample_idx,
                            extra=f"dt={time.perf_counter() - backward_t0:.2f}s loss={float(loss.detach().item()):.6e}",
                        )
                        _log_progress()
                    else:
                        _log_skip(
                            "zero_loss",
                            batch_idx,
                            sample_idx,
                            extra=f"loss={float(loss.detach().item()):.6e}",
                        )
                        _log_stage(
                            "fisher_backward:skip_zero_loss",
                            batch_idx,
                            sample_idx,
                            extra=f"dt={time.perf_counter() - backward_t0:.2f}s",
                        )
                        _log_progress()

                    self.actor_module.zero_grad(set_to_none=True)
                    del out, logits, labels, nll, token_weight, loss, anchor_ids_gpu
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        _log_progress(force=True)
        skipped_total = max(0, attempted_samples_count - valid_samples_count)
        print(
            f"[TARP Summary pid={pid}] attempts={attempted_samples_count} valid={valid_samples_count} "
            f"skipped={skipped_total} skip_breakdown={_format_skip_breakdown()} "
            f"cache_enabled={int(reasoning_cache_enabled)} cache_hit={cache_hit_count} "
            f"cache_miss={cache_miss_count} cache_write={cache_write_count} cache_invalid={cache_invalid_count} "
            f"cache_path={reasoning_cache_path if reasoning_cache_path else 'null'}",
            flush=True,
        )
        if reasoning_cache_fp is not None:
            try:
                reasoning_cache_fp.flush()
                reasoning_cache_fp.close()
            except Exception:
                pass
        print(f"[TARP] Finalizing PCA...", flush=True)
        tarp_module.finalize_fisher()

        part_id = resolve_tarp_part_id(self.config)
        primary_data_source = next((source for source in data_source_observed if source), None)
        dataset_key = get_tarp_dataset_key(self.config, data_source=primary_data_source) or infer_primary_dataset_key(data_source_observed) or "UNKNOWN"
        part_path, merged_path = resolve_tarp_save_paths(self.config, dataset_key, rank, world, valid_samples_count, part_id)
        state = tarp_module.state_dict()
        state["_meta"] = build_tarp_state_meta(dataset_key, primary_data_source, self.config, part_id, rank, world, valid_samples_count)
        state["_meta"]["pid"] = int(pid)
        state["_meta"]["experiment_name"] = str(getattr(self.config, "tarp_experiment_name", ""))

        online_tarp_enabled = is_online_tarp_enabled(self.config)
        if online_tarp_enabled:
            prior_candidates = resolve_tarp_load_candidates(self.config, rank, world, data_source=primary_data_source)
            prior_path = next((path for path in prior_candidates if os.path.exists(path)), None)
            if prior_path is not None:
                prior_state = torch.load(prior_path, map_location="cpu")
                validate_tarp_state_dict(
                    prior_state,
                    [meta.name for meta in tarp_module.mat_metas] + [meta.name for meta in tarp_module.vec_metas],
                    expected_config={
                        "rank_per_block": int(getattr(self.config, "tarp_rank_per_block", 32)),
                        "block_size": int(getattr(self.config, "tarp_block_size", 2048)),
                        "proj_dim": int(getattr(self.config, "tarp_proj_dim", 48)),
                    },
                )
                tarp_module.validate_state_dict_shapes(prior_state)
                state = consolidate_tarp_state_dict(
                    previous_state=prior_state,
                    current_state=state,
                    decay=get_tarp_online_decay(self.config),
                )
                state.setdefault("_meta", {})
                state["_meta"]["online_prior_path"] = prior_path
                if rank == 0:
                    print(f"[TARP] Online TARP consolidation loaded prior shard: {prior_path}", flush=True)
            elif rank == 0:
                print("[TARP] Online TARP consolidation found no prior shard; saving current task as the initial prior.", flush=True)
            state.setdefault("_meta", {})
            state["_meta"]["online_tarp"] = True
            state["_meta"].setdefault("online_history", [])
            if state["_meta"].get("experiment_name"):
                if not state["_meta"]["online_history"]:
                    state["_meta"]["online_history"] = [state["_meta"]["experiment_name"]]
                elif state["_meta"]["experiment_name"] not in state["_meta"]["online_history"]:
                    state["_meta"]["online_history"].append(state["_meta"]["experiment_name"])

        def _atomic_save(path: str):
            tmp_path = path + f".tmp{pid}"
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)

        _atomic_save(part_path)
        _atomic_save(merged_path)

        if is_dist:
            dist.barrier()
        if rank == 0:
            print(f"[TARP] Saved part shard: {part_path}", flush=True)
            print(f"[TARP] Saved merged shard: {merged_path}", flush=True)
        return
