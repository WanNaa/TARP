from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch

EWC_DEFAULT_ROOT = "/path/to/outputs/ewc"
EWC_DEFAULT_MODE = "DIAG_FISHER"
FSDP_WRAPPED_MODULE_PREFIX = "_fsdp_wrapped_module."


def is_none_like(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "nil"}


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_cfg(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _param_name_aliases(name: str) -> tuple[str, ...]:
    aliases = [name]
    canonical = name.replace(FSDP_WRAPPED_MODULE_PREFIX, "")
    aliases.append(canonical)
    aliases.append(FSDP_WRAPPED_MODULE_PREFIX + canonical)
    return tuple(dict.fromkeys(aliases))


def _canonical_param_name(name: str) -> str:
    return str(name).replace(FSDP_WRAPPED_MODULE_PREFIX, "")


def _canonical_source_key_map(source: Dict[str, torch.Tensor]) -> Dict[str, str]:
    canonical_map: Dict[str, str] = {}
    for key in source.keys():
        canonical_map.setdefault(_canonical_param_name(key), key)
    return canonical_map


def resolve_torch_dtype(value, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
    }
    if text in mapping:
        return mapping[text]
    raise ValueError(f"Unsupported torch dtype value: {value}")


def normalize_ewc_mode(mode) -> str:
    raw = "" if mode is None else str(mode).strip().upper()
    if raw in {"", "DIAG_FISHER", "DIAGFISHER", "DIAGONAL_FISHER"}:
        return "DIAG_FISHER"
    raise ValueError(f"Unsupported ewc_mode: {mode}")


def resolve_ewc_mode(config) -> str:
    return normalize_ewc_mode(_get_cfg(config, "ewc_mode", EWC_DEFAULT_MODE))


def is_ewc_export_only(config) -> bool:
    return _is_truthy(_get_cfg(config, "ewc_export_only", False))


def is_ewc_loss_enabled(config) -> bool:
    return _is_truthy(_get_cfg(config, "use_ewc_loss", False))


def infer_dataset_key(data_source: Optional[str]) -> Optional[str]:
    if data_source is None:
        return None
    source = str(data_source).strip().lower()
    if not source:
        return None
    if "countdown" in source:
        return "CD"
    if "kk" in source:
        return "KK"
    if source == "code" or "code-r1" in source:
        return "R1"
    return None


def infer_primary_dataset_key(data_sources: Iterable[object]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for source in data_sources:
        key = infer_dataset_key(None if source is None else str(source))
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def get_ewc_dataset_key(config, data_source: Optional[str] = None) -> Optional[str]:
    explicit = _get_cfg(config, "ewc_dataset_key", None)
    if not is_none_like(explicit):
        return str(explicit).strip().upper()
    return infer_dataset_key(data_source)


def get_ewc_storage_dir(config, dataset_key: Optional[str]) -> str:
    explicit_dir = _get_cfg(config, "ewc_dir", None)
    if not is_none_like(explicit_dir):
        return str(explicit_dir)
    root = str(_get_cfg(config, "ewc_root", EWC_DEFAULT_ROOT))
    key = (dataset_key or "UNKNOWN").upper()
    return os.path.join(root, key)


def build_ewc_artifact_stem(dataset_key: str, mode: str) -> str:
    compact_mode = resolve_ewc_mode({"ewc_mode": mode}).lower().replace("_", "")
    return f"EWC__{dataset_key.upper()}__{compact_mode}"


def build_ewc_part_filename(dataset_key: str, mode: str, sample_count: int, rank: int, world_size: int) -> str:
    stem = build_ewc_artifact_stem(dataset_key, mode)
    return f"{stem}__samples-{int(sample_count)}__rank-{int(rank)}-of-{int(world_size)}.pt"


def build_ewc_merged_filename(dataset_key: str, mode: str, rank: int, world_size: int) -> str:
    stem = build_ewc_artifact_stem(dataset_key, mode)
    return f"{stem}__merged__rank-{int(rank)}-of-{int(world_size)}.pt"


def resolve_ewc_save_paths(config, dataset_key: str, rank: int, world_size: int, sample_count: int) -> tuple[str, str]:
    storage_dir = get_ewc_storage_dir(config, dataset_key)
    os.makedirs(storage_dir, exist_ok=True)
    mode = resolve_ewc_mode(config)

    explicit_tag = _get_cfg(config, "ewc_tag", None)
    if is_none_like(explicit_tag):
        part_name = build_ewc_part_filename(dataset_key, mode, sample_count, rank, world_size)
        merged_name = build_ewc_merged_filename(dataset_key, mode, rank, world_size)
    else:
        stem = str(explicit_tag)
        part_name = f"{stem}__samples-{int(sample_count)}__rank-{int(rank)}-of-{int(world_size)}.pt"
        merged_name = f"{stem}__merged__rank-{int(rank)}-of-{int(world_size)}.pt"
    return os.path.join(storage_dir, part_name), os.path.join(storage_dir, merged_name)


def _candidate_rank_files(base: str, rank: int, world_size: int) -> list[str]:
    root, ext = os.path.splitext(base)
    suffix = f"__rank-{int(rank)}-of-{int(world_size)}"
    if suffix in base:
        return [base]
    candidates = []
    if ext == ".pt":
        candidates.extend(
            [
                f"{root}__merged{suffix}.pt",
                f"{root}{suffix}.pt",
            ]
        )
        if world_size != 1:
            candidates.extend(
                [
                    f"{root}__merged__rank-0-of-1.pt",
                    f"{root}__rank-0-of-1.pt",
                ]
            )
        candidates.append(base)
        return list(dict.fromkeys(candidates))

    candidates.extend(
        [
            f"{base}__merged{suffix}.pt",
            f"{base}{suffix}.pt",
        ]
    )
    if world_size != 1:
        candidates.extend(
            [
                f"{base}__merged__rank-0-of-1.pt",
                f"{base}__rank-0-of-1.pt",
            ]
        )
    candidates.append(base)
    return list(dict.fromkeys(candidates))


def resolve_ewc_load_candidates(config, rank: int, world_size: int, data_source: Optional[str] = None) -> list[str]:
    explicit_path = _get_cfg(config, "ewc_load_path", None)
    tag = _get_cfg(config, "ewc_tag", None)
    mode = resolve_ewc_mode(config)
    candidates: list[str] = []

    if not is_none_like(explicit_path):
        base_path = str(explicit_path)
        if os.path.isdir(base_path):
            if not is_none_like(tag):
                tagged = os.path.join(
                    base_path,
                    f"{str(tag)}__merged__rank-{int(rank)}-of-{int(world_size)}.pt",
                )
                candidates.append(tagged)
            candidates.extend(
                sorted(
                    glob.glob(os.path.join(base_path, f"*__merged__rank-{int(rank)}-of-{int(world_size)}.pt"))
                )
            )
            candidates.extend(
                sorted(
                    glob.glob(os.path.join(base_path, f"*__samples-*__rank-{int(rank)}-of-{int(world_size)}.pt"))
                )
            )
            if world_size != 1:
                if not is_none_like(tag):
                    candidates.extend(
                        [
                            os.path.join(base_path, f"{str(tag)}__merged__rank-0-of-1.pt"),
                        ]
                    )
                    candidates.extend(
                        sorted(
                            glob.glob(os.path.join(base_path, f"{str(tag)}__samples-*__rank-0-of-1.pt"))
                        )
                    )
                candidates.extend(sorted(glob.glob(os.path.join(base_path, "*__merged__rank-0-of-1.pt"))))
                candidates.extend(sorted(glob.glob(os.path.join(base_path, "*__samples-*__rank-0-of-1.pt"))))
        else:
            candidates.extend(_candidate_rank_files(base_path, rank, world_size))
        return list(dict.fromkeys(candidates))

    dataset_key = get_ewc_dataset_key(config, data_source=data_source)
    if dataset_key is None:
        return []

    storage_dir = get_ewc_storage_dir(config, dataset_key)
    if is_none_like(tag):
        merged_name = build_ewc_merged_filename(dataset_key, mode, rank, world_size)
        candidates.append(os.path.join(storage_dir, merged_name))
        candidates.extend(
            sorted(
                glob.glob(
                    os.path.join(
                        storage_dir,
                        f"{build_ewc_artifact_stem(dataset_key, mode)}__samples-*__rank-{int(rank)}-of-{int(world_size)}.pt",
                    )
                )
            )
        )
        if world_size != 1:
            candidates.append(os.path.join(storage_dir, build_ewc_merged_filename(dataset_key, mode, 0, 1)))
            candidates.extend(
                sorted(
                    glob.glob(
                        os.path.join(
                            storage_dir,
                            f"{build_ewc_artifact_stem(dataset_key, mode)}__samples-*__rank-0-of-1.pt",
                        )
                    )
                )
            )
    else:
        stem = str(tag)
        candidates.append(os.path.join(storage_dir, f"{stem}__merged__rank-{int(rank)}-of-{int(world_size)}.pt"))
        candidates.extend(
            sorted(
                glob.glob(
                    os.path.join(storage_dir, f"{stem}__samples-*__rank-{int(rank)}-of-{int(world_size)}.pt")
                )
            )
        )
        if world_size != 1:
            candidates.append(os.path.join(storage_dir, f"{stem}__merged__rank-0-of-1.pt"))
            candidates.extend(
                sorted(
                    glob.glob(os.path.join(storage_dir, f"{stem}__samples-*__rank-0-of-1.pt"))
                )
            )
    return list(dict.fromkeys(candidates))


def build_ewc_state_meta(dataset_key: str, data_source: Optional[str], config, rank: int, world_size: int, valid_samples_used: int) -> dict:
    return {
        "dataset_key": dataset_key.upper(),
        "dataset": None if data_source is None else str(data_source),
        "mode": resolve_ewc_mode(config),
        "rank": int(rank),
        "world_size": int(world_size),
        "valid_samples_used": int(valid_samples_used),
        "sample_count": int(valid_samples_used),
        "ewc_lambda": float(_get_cfg(config, "ewc_lambda", 0.0)),
        "source_checkpoint": _get_cfg(config, "ewc_source_model_path", None),
        "experiment_name": _get_cfg(config, "ewc_experiment_name", ""),
        "target_valid_samples": int(_get_cfg(config, "ewc_target_valid_samples", 0)),
    }


def validate_ewc_state_dict(state_dict: dict, expected_param_names: Iterable[str], expected_mode: Optional[str] = None) -> None:
    if not isinstance(state_dict, dict):
        raise TypeError("EWC state_dict must be a dict.")
    if "theta_star" not in state_dict or "fisher_diag" not in state_dict:
        raise KeyError("EWC state_dict must contain 'theta_star' and 'fisher_diag'.")

    theta_star = state_dict["theta_star"]
    fisher_diag = state_dict["fisher_diag"]
    if not isinstance(theta_star, dict) or not isinstance(fisher_diag, dict):
        raise TypeError("EWC theta_star and fisher_diag must both be dicts.")
    theta_key_map = _canonical_source_key_map(theta_star)
    fisher_key_map = _canonical_source_key_map(fisher_diag)

    if expected_mode is not None:
        meta = state_dict.get("_meta", {})
        actual_mode = normalize_ewc_mode(meta.get("mode", expected_mode))
        wanted_mode = normalize_ewc_mode(expected_mode)
        if actual_mode != wanted_mode:
            raise ValueError(f"EWC mode mismatch: expected {wanted_mode}, found {actual_mode}")

    missing: list[str] = []
    for name in expected_param_names:
        canonical = _canonical_param_name(name)
        if canonical not in theta_key_map or canonical not in fisher_key_map:
            missing.append(name)
    if missing:
        raise ValueError(f"EWC state missing parameter entries for: {missing[:8]}")


def consolidate_ewc_state_dict(previous_state: dict, current_state: dict, decay: float = 1.0) -> dict:
    if decay < 0:
        raise ValueError(f"EWC online decay must be >= 0, got {decay}")

    prev_theta = previous_state.get("theta_star", {})
    prev_fisher = previous_state.get("fisher_diag", {})
    curr_theta = current_state.get("theta_star", {})
    curr_fisher = current_state.get("fisher_diag", {})
    if not isinstance(prev_theta, dict) or not isinstance(prev_fisher, dict):
        raise TypeError("previous_state must contain dict theta_star and fisher_diag")
    if not isinstance(curr_theta, dict) or not isinstance(curr_fisher, dict):
        raise TypeError("current_state must contain dict theta_star and fisher_diag")

    prev_theta_key_map = _canonical_source_key_map(prev_theta)
    prev_fisher_key_map = _canonical_source_key_map(prev_fisher)
    curr_theta_key_map = _canonical_source_key_map(curr_theta)
    curr_fisher_key_map = _canonical_source_key_map(curr_fisher)

    merged_theta: Dict[str, torch.Tensor] = {}
    merged_fisher: Dict[str, torch.Tensor] = {}

    for curr_name, curr_theta_tensor in curr_theta.items():
        canonical = _canonical_param_name(curr_name)
        curr_fisher_key = curr_fisher_key_map.get(canonical)
        if curr_fisher_key is None:
            raise KeyError(f"current_state missing fisher tensor for {curr_name}")

        curr_fisher_tensor = curr_fisher[curr_fisher_key]
        merged_theta[curr_name] = curr_theta_tensor.detach().cpu().clone()
        fisher_tensor = curr_fisher_tensor.detach().cpu().float().clone()

        prev_theta_key = prev_theta_key_map.get(canonical)
        prev_fisher_key = prev_fisher_key_map.get(canonical)
        if prev_theta_key is not None and prev_fisher_key is not None:
            prev_theta_tensor = prev_theta[prev_theta_key]
            prev_fisher_tensor = prev_fisher[prev_fisher_key]
            if tuple(prev_theta_tensor.shape) != tuple(curr_theta_tensor.shape):
                raise ValueError(
                    f"EWC online consolidation theta shape mismatch for {curr_name}: "
                    f"previous={tuple(prev_theta_tensor.shape)} current={tuple(curr_theta_tensor.shape)}"
                )
            if tuple(prev_fisher_tensor.shape) != tuple(curr_fisher_tensor.shape):
                raise ValueError(
                    f"EWC online consolidation fisher shape mismatch for {curr_name}: "
                    f"previous={tuple(prev_fisher_tensor.shape)} current={tuple(curr_fisher_tensor.shape)}"
                )
            fisher_tensor.add_(prev_fisher_tensor.detach().cpu().float(), alpha=float(decay))

        merged_fisher[curr_name] = fisher_tensor.to(dtype=curr_fisher_tensor.dtype)

    result = dict(current_state)
    result["theta_star"] = merged_theta
    result["fisher_diag"] = merged_fisher

    prev_count = float(previous_state.get("sample_count", 0) or 0.0)
    curr_count = float(current_state.get("sample_count", 0) or 0.0)
    result["sample_count"] = curr_count + decay * prev_count

    meta = dict(current_state.get("_meta", {}))
    if meta:
        prev_meta = previous_state.get("_meta", {}) if isinstance(previous_state.get("_meta", {}), dict) else {}
        history = list(prev_meta.get("online_history", []))
        if not history and prev_meta.get("experiment_name"):
            history = [prev_meta.get("experiment_name")]
        current_experiment = meta.get("experiment_name")
        if current_experiment:
            if current_experiment in history:
                raise RuntimeError(
                    f"EWC online consolidation would double-count experiment {current_experiment!r}. "
                    "Use a fresh ewc_tag/ewc_dir when rebuilding the same stage."
                )
            history.append(current_experiment)
        meta["online_ewc"] = True
        meta["online_decay"] = float(decay)
        meta["online_consolidated_from_dataset_key"] = prev_meta.get("dataset_key")
        meta["online_consolidated_from_valid_samples"] = prev_meta.get("valid_samples_used")
        meta["online_history"] = history
        meta["online_stage_count"] = len(history)
        result["_meta"] = meta

    return result


@dataclass
class EWCParamRef:
    name: str
    param: torch.nn.Parameter


class DiagonalFisherEWC:
    def __init__(
        self,
        model: torch.nn.Module,
        param_names: Optional[Sequence[str]] = None,
        accum_dtype: torch.dtype = torch.float32,
        store_dtype: torch.dtype = torch.bfloat16,
    ):
        named_params = [(name, param) for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]
        if param_names is not None:
            allowed = set(param_names)
            named_params = [(name, param) for name, param in named_params if name in allowed]
        if not named_params:
            raise ValueError("No trainable parameters selected for DiagonalFisherEWC.")

        self.model = model
        self.accum_dtype = accum_dtype
        self.store_dtype = resolve_torch_dtype(store_dtype)
        self.param_refs = [EWCParamRef(name=name, param=param) for name, param in named_params]
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.fisher_diag_cpu: Dict[str, torch.Tensor] = {
            ref.name: torch.zeros_like(ref.param.detach().cpu(), dtype=self.accum_dtype) for ref in self.param_refs
        }
        self.sample_count: int = 0

        self._cached_device: Optional[torch.device] = None
        self.theta_star_gpu: Dict[str, torch.Tensor] = {}
        self.fisher_diag_gpu: Dict[str, torch.Tensor] = {}

    @property
    def param_names(self) -> list[str]:
        return [ref.name for ref in self.param_refs]

    def _current_param_map(self) -> Dict[str, torch.nn.Parameter]:
        param_map: Dict[str, torch.nn.Parameter] = {}
        for name, param in self.model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            for alias in _param_name_aliases(name):
                param_map.setdefault(alias, param)
        return param_map

    @torch.no_grad()
    def snapshot_theta_star(self):
        self.theta_star = {
            ref.name: ref.param.detach().cpu().clone().to(dtype=self.store_dtype)
            for ref in self.param_refs
        }
        self._cached_device = None
        self.theta_star_gpu.clear()
        self.fisher_diag_gpu.clear()

    @torch.no_grad()
    def accumulate_fisher_after_backward(self):
        self.sample_count += 1
        for ref in self.param_refs:
            if ref.param.grad is None:
                continue
            grad_sq = ref.param.grad.detach().float().cpu().pow(2)
            self.fisher_diag_cpu[ref.name].add_(grad_sq.to(dtype=self.accum_dtype))

    @torch.no_grad()
    def finalize_fisher(self):
        if self.sample_count <= 0:
            raise RuntimeError("No samples accumulated for EWC Fisher export.")
        for name, fisher in list(self.fisher_diag_cpu.items()):
            self.fisher_diag_cpu[name] = (fisher / float(self.sample_count)).to(dtype=self.store_dtype)
        self._cached_device = None
        self.theta_star_gpu.clear()
        self.fisher_diag_gpu.clear()

    @torch.no_grad()
    def cache_to_device(self, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        dev = torch.device(device)
        cache_dtype = resolve_torch_dtype(dtype, default=self.store_dtype)
        self._cached_device = dev
        self.theta_star_gpu = {
            name: tensor.to(device=dev, dtype=cache_dtype, non_blocking=True)
            for name, tensor in self.theta_star.items()
        }
        self.fisher_diag_gpu = {
            name: tensor.to(device=dev, dtype=cache_dtype, non_blocking=True)
            for name, tensor in self.fisher_diag_cpu.items()
        }

    def penalty(self, coef: float = 1.0) -> torch.Tensor:
        total = None
        param_map = self._current_param_map()
        for name in self.param_names:
            param = param_map[name]
            if self._cached_device is not None:
                theta = self.theta_star_gpu[name]
                fisher = self.fisher_diag_gpu[name]
            else:
                theta = self.theta_star[name].to(device=param.device)
                fisher = self.fisher_diag_cpu[name].to(device=param.device)
            delta = param - theta.to(device=param.device, dtype=param.dtype)
            term = 0.5 * (fisher.to(device=param.device, dtype=delta.dtype) * delta.pow(2)).sum()
            total = term if total is None else (total + term)
        if total is None:
            device = next(iter(param_map.values())).device
            return torch.zeros((), device=device, dtype=torch.float32)
        return total * float(coef)

    def state_magnitude_summary(self) -> dict:
        means = []
        maxes = []
        layer_stats = {}
        for name, fisher in self.fisher_diag_cpu.items():
            fisher_f = fisher.float()
            mean_val = float(fisher_f.mean().item()) if fisher_f.numel() > 0 else 0.0
            max_val = float(fisher_f.max().item()) if fisher_f.numel() > 0 else 0.0
            means.append(mean_val)
            maxes.append(max_val)
            layer_stats[name] = {"mean": mean_val, "max": max_val}
        return {
            "fisher_global_mean": float(sum(means) / len(means)) if means else 0.0,
            "fisher_global_max": float(max(maxes)) if maxes else 0.0,
            "fisher_layer_stats": layer_stats,
            "fisher_count": int(sum(int(t.numel()) for t in self.fisher_diag_cpu.values())),
        }

    def state_dict(self) -> dict:
        return {
            "theta_star": self.theta_star,
            "fisher_diag": self.fisher_diag_cpu,
            "sample_count": int(self.sample_count),
            "param_shapes": {name: list(tensor.shape) for name, tensor in self.theta_star.items()},
            "store_dtype": str(self.store_dtype),
        }

    def load_state_dict(self, state_dict: dict):
        theta_star_src = state_dict["theta_star"]
        fisher_diag_src = state_dict["fisher_diag"]
        theta_key_map = _canonical_source_key_map(theta_star_src)
        fisher_key_map = _canonical_source_key_map(fisher_diag_src)
        current_names = self.param_names
        remapped_theta: Dict[str, torch.Tensor] = {}
        remapped_fisher: Dict[str, torch.Tensor] = {}

        for name in current_names:
            canonical = _canonical_param_name(name)
            matched_theta = theta_key_map.get(canonical)
            matched_fisher = fisher_key_map.get(canonical)
            if matched_theta is None or matched_fisher is None:
                raise KeyError(f"Failed to find EWC tensors for parameter {name}")
            remapped_theta[name] = theta_star_src[matched_theta].detach().cpu().clone()
            remapped_fisher[name] = fisher_diag_src[matched_fisher].detach().cpu().clone()

        for ref in self.param_refs:
            theta = remapped_theta[ref.name]
            fisher = remapped_fisher[ref.name]
            if tuple(theta.shape) != tuple(ref.param.shape):
                raise ValueError(
                    f"EWC theta_star shape mismatch for {ref.name}: "
                    f"state={tuple(theta.shape)} current={tuple(ref.param.shape)}"
                )
            if tuple(fisher.shape) != tuple(ref.param.shape):
                raise ValueError(
                    f"EWC fisher_diag shape mismatch for {ref.name}: "
                    f"state={tuple(fisher.shape)} current={tuple(ref.param.shape)}"
                )

        self.theta_star = {
            name: tensor.to(dtype=self.store_dtype)
            for name, tensor in remapped_theta.items()
        }
        self.fisher_diag_cpu = {
            name: tensor.to(dtype=self.store_dtype)
            for name, tensor in remapped_fisher.items()
        }
        self.sample_count = int(state_dict.get("sample_count", 0))
        self._cached_device = None
        self.theta_star_gpu.clear()
        self.fisher_diag_gpu.clear()
