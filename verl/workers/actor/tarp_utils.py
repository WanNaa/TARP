import torch
import torch.nn as nn
import os
from glob import glob
import re
import gc
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence, Iterable, Any

# ==================================================================
# Part 1: Block Exact Subspace TARP Implementation
# ==================================================================

TARP_DEFAULT_ROOT = "/path/to/outputs/anchor"
TARP_DEFAULT_PARAM_VIEW = "orig"
TARP_DEFAULT_TRAIN_MODE = "baseline"
FSDP_WRAPPED_MODULE_PREFIX = "_fsdp_wrapped_module."
TARP_DEFAULT_LINE_TYPE_WEIGHTS = {
    "math_step": 1.0,
    "search_control": 0.3,
    "filler": 0.05,
}
TARP_KK_LINE_TYPE_WEIGHTS = {
    "logic_step": 1.0,
    "consistency_check": 0.8,
    "statement_focus": 0.55,
    "summary": 0.3,
    "filler": 0.1,
}


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_none_like(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null"}


def compact_tarp_tag_value(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("'\"")
    text = re.sub(r"[ /:,]+", "_", text)
    text = text.lower().replace(".", "p").replace("+", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def infer_model_size_from_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    basename = os.path.basename(str(path))
    prefixed = re.search(r"([A-Za-z]+[0-9]+-[0-9]+(?:\.[0-9]+)?B)", basename, flags=re.IGNORECASE)
    if prefixed is not None:
        return prefixed.group(1).upper()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?B)", basename, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper()


def build_anchor_suite_name(model_size: str, stage_sequence: Sequence[str], cap_value, lam_value) -> str:
    normalized = extend_stage_sequence(stage_sequence, None)
    stage_expr = "+".join(normalized) if normalized else "UNKNOWN"
    return (
        f"{str(model_size).upper()}-{stage_expr}-"
        f"cap{compact_tarp_tag_value(cap_value)}-"
        f"lam{compact_tarp_tag_value(lam_value)}"
    )


def build_anchor_storage_dir(root: str, model_size: str, stage_sequence: Sequence[str], cap_value, lam_value) -> str:
    return os.path.join(
        str(root),
        build_anchor_suite_name(
            model_size=model_size,
            stage_sequence=stage_sequence,
            cap_value=cap_value,
            lam_value=lam_value,
        ),
    )


def is_anchor_root(root: Optional[str]) -> bool:
    if root is None:
        return False
    normalized = os.path.basename(os.path.normpath(str(root)))
    return normalized == "Anchor"


def infer_stage_sequence_from_path(path: Optional[str]) -> list[str]:
    if path is None:
        return []

    basename = os.path.basename(str(path))
    tokens = re.split(r"[^A-Za-z0-9]+", basename)
    stages: list[str] = []
    for token in tokens:
        normalized = token.strip().upper()
        if not normalized:
            continue
        stage = None
        if re.fullmatch(r"CD\d*", normalized):
            stage = "CD"
        elif re.fullmatch(r"KK\d*", normalized):
            stage = "KK"
        elif re.fullmatch(r"R1\d*", normalized):
            stage = "R1"
        if stage is not None and (not stages or stages[-1] != stage):
            stages.append(stage)
    return stages


def extend_stage_sequence(stage_sequence: Sequence[str], stage: Optional[str]) -> list[str]:
    normalized: list[str] = []
    for item in stage_sequence:
        candidate = _normalize_tarp_dataset_key(item)
        if candidate is None:
            continue
        if not normalized or normalized[-1] != candidate:
            normalized.append(candidate)

    current_stage = _normalize_tarp_dataset_key(stage)
    if current_stage is not None and (not normalized or normalized[-1] != current_stage):
        normalized.append(current_stage)
    return normalized


def build_online_tarp_tag(stage_sequence: Sequence[str], cap_value, lam_value) -> str:
    normalized = extend_stage_sequence(stage_sequence, None)
    stage_prefix = "-".join(normalized) if normalized else "UNKNOWN"
    return (
        f"{stage_prefix}_TARP_"
        f"cap{compact_tarp_tag_value(cap_value)}_"
        f"lam{compact_tarp_tag_value(lam_value)}"
    )


def _param_name_aliases(name: str) -> tuple[str, ...]:
    aliases = [name]
    canonical = name.replace(FSDP_WRAPPED_MODULE_PREFIX, "")
    aliases.append(canonical)
    aliases.append(FSDP_WRAPPED_MODULE_PREFIX + canonical)
    return tuple(dict.fromkeys(aliases))


def normalize_tarp_train_mode(mode) -> str:
    raw = "" if mode is None else str(mode).strip().lower()
    if raw in {"", "0", "false", "none", "null", "off", "baseline"}:
        return "baseline"
    if raw in {"1", "true", "on", "tarp", "continual"}:
        return "continual"
    raise ValueError(f"Unsupported tarp_train_mode: {mode}")


def resolve_tarp_train_mode(config) -> str:
    if hasattr(config, "get"):
        explicit_mode = config.get("tarp_train_mode", None)
    else:
        explicit_mode = getattr(config, "tarp_train_mode", None)
    if explicit_mode is None:
        explicit_mode = TARP_DEFAULT_TRAIN_MODE
    return normalize_tarp_train_mode(explicit_mode)


def is_online_tarp_enabled(config) -> bool:
    if hasattr(config, "get"):
        raw = config.get("online_tarp", False)
    else:
        raw = getattr(config, "online_tarp", False)
    return _is_truthy(raw)


def get_tarp_online_decay(config) -> float:
    if hasattr(config, "get"):
        raw = config.get("tarp_online_decay", 1.0)
    else:
        raw = getattr(config, "tarp_online_decay", 1.0)
    decay = float(raw)
    if decay < 0:
        raise ValueError(f"tarp_online_decay must be >= 0, got {decay}")
    return decay


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


def _normalize_tarp_dataset_key(dataset_key: Optional[str]) -> Optional[str]:
    if dataset_key is None:
        return None
    normalized = str(dataset_key).strip().upper()
    return normalized or None


def infer_primary_dataset_key(data_sources: Iterable[object]) -> Optional[str]:
    keys = [infer_dataset_key(source) for source in data_sources]
    keys = [key for key in keys if key is not None]
    if not keys:
        return None
    return Counter(keys).most_common(1)[0][0]


def get_tarp_param_view(config) -> str:
    return str(getattr(config, "tarp_param_view", TARP_DEFAULT_PARAM_VIEW)).strip().lower()


def get_tarp_dataset_key(config, data_source: Optional[str] = None) -> Optional[str]:
    config_key = getattr(config, "tarp_dataset_key", None)
    if config_key:
        return str(config_key).strip().upper()
    return infer_dataset_key(data_source)


def get_tarp_load_dataset_key(config, data_source: Optional[str] = None) -> Optional[str]:
    load_key = getattr(config, "tarp_load_dataset_key", None)
    if load_key is not None:
        normalized = str(load_key).strip()
        if normalized.lower() in {"", "none", "null"}:
            return None
        return normalized.upper()
    return get_tarp_dataset_key(config, data_source=data_source)


def get_tarp_model_size(config) -> Optional[str]:
    explicit = getattr(config, "tarp_model_size", None)
    if explicit:
        return str(explicit).strip().upper()
    env_value = os.environ.get("TARP_MODEL_SIZE")
    if env_value:
        return str(env_value).strip().upper()
    return None


def get_tarp_storage_dir(config, dataset_key: Optional[str]) -> str:
    root = str(getattr(config, "tarp_root", TARP_DEFAULT_ROOT))
    key = (dataset_key or "UNKNOWN").upper()
    if is_anchor_root(root):
        model_size = get_tarp_model_size(config)
        if model_size:
            cap_value = getattr(config, "tarp_to_pg_ratio_cap", 0)
            lam_value = getattr(config, "tarp_lambda", 0.0)
            return build_anchor_storage_dir(
                root=root,
                model_size=model_size,
                stage_sequence=[key],
                cap_value=cap_value,
                lam_value=lam_value,
            )
    return os.path.join(root, key)


def build_tarp_artifact_stem(dataset_key: str, param_view: str) -> str:
    return f"TARP__{dataset_key.upper()}__{param_view.lower()}"


def build_tarp_part_filename(dataset_key: str, param_view: str, sample_count: int, part_id: int, rank: int, world_size: int) -> str:
    stem = build_tarp_artifact_stem(dataset_key, param_view)
    return f"{stem}__samples-{int(sample_count)}__part-{int(part_id)}__rank-{int(rank)}-of-{int(world_size)}.pt"


def build_tarp_merged_filename(dataset_key: str, param_view: str, rank: int, world_size: int) -> str:
    stem = build_tarp_artifact_stem(dataset_key, param_view)
    return f"{stem}__merged__rank-{int(rank)}-of-{int(world_size)}.pt"


def resolve_tarp_part_id(config) -> int:
    try:
        return int(getattr(config, "tarp_part_id", 0))
    except Exception:
        return int(os.environ.get("TARP_PART_ID", "0"))


def resolve_tarp_load_candidates(config, rank: int, world_size: int, data_source: Optional[str] = None) -> list[str]:
    candidates: list[str] = []

    def _append_unique(path: str) -> None:
        if path not in candidates:
            candidates.append(path)

    def _extend_glob(pattern: str) -> None:
        for match in reversed(sorted(glob(pattern))):
            _append_unique(match)

    explicit_dir = getattr(config, "tarp_load_dir", None)
    if is_none_like(explicit_dir):
        explicit_dir = getattr(config, "tarp_dir", None)
    explicit_tag = getattr(config, "tarp_load_tag", None)
    if is_none_like(explicit_tag):
        explicit_tag = getattr(config, "tarp_tag", None)
    dataset_key = get_tarp_load_dataset_key(config, data_source=data_source)
    param_view = get_tarp_param_view(config)

    if not is_none_like(explicit_dir):
        storage_dir = str(explicit_dir)
        stem = (
            str(explicit_tag).strip()
            if not is_none_like(explicit_tag)
            else (build_tarp_artifact_stem(dataset_key, param_view) if dataset_key else None)
        )
        if stem:
            _append_unique(os.path.join(storage_dir, f"{stem}__merged__rank-{rank}-of-{world_size}.pt"))
            _extend_glob(os.path.join(storage_dir, f"{stem}__merged__rank-{rank}-of-*.pt"))
            _extend_glob(os.path.join(storage_dir, f"{stem}__merged__rank-*-of-*.pt"))
            part_id = resolve_tarp_part_id(config)
            part_patterns = [
                os.path.join(storage_dir, f"{stem}__samples-*__part-{part_id}__rank-{rank}-of-{world_size}.pt"),
                os.path.join(storage_dir, f"{stem}__samples-*__part-{part_id}__rank-{rank}-of-*.pt"),
                os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-{rank}-of-{world_size}.pt"),
                os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-{rank}-of-*.pt"),
                os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-*-of-*.pt"),
            ]
            for pattern in part_patterns:
                _extend_glob(pattern)

        if not is_none_like(explicit_tag):
            _append_unique(os.path.join(storage_dir, f"{explicit_tag}_rank{rank}-of-{world_size}.pt"))
            _append_unique(os.path.join(storage_dir, f"{explicit_tag}_rank-{rank}-of-{world_size}.pt"))

    if dataset_key is None:
        return candidates

    root = str(getattr(config, "tarp_root", TARP_DEFAULT_ROOT))
    stem = build_tarp_artifact_stem(dataset_key, param_view)

    storage_dirs: list[str] = []
    if is_anchor_root(root):
        model_size = get_tarp_model_size(config)
        if model_size:
            anchor_pattern = os.path.join(root, f"{model_size}-{dataset_key.upper()}-*")
            exact_stem_dirs: list[str] = []
            fallback_dirs: list[str] = []
            for candidate_dir in sorted(glob(anchor_pattern)):
                if not os.path.isdir(candidate_dir):
                    continue
                if glob(os.path.join(candidate_dir, f"{stem}__merged__rank-*-of-*.pt")):
                    exact_stem_dirs.append(candidate_dir)
                else:
                    fallback_dirs.append(candidate_dir)
            storage_dirs.extend(exact_stem_dirs + fallback_dirs)

    if not storage_dirs:
        storage_dirs.append(get_tarp_storage_dir(config, dataset_key))

    part_id = resolve_tarp_part_id(config)
    for storage_dir in storage_dirs:
        _append_unique(os.path.join(storage_dir, build_tarp_merged_filename(dataset_key, param_view, rank, world_size)))
        _extend_glob(os.path.join(storage_dir, f"{stem}__merged__rank-{rank}-of-*.pt"))
        _extend_glob(os.path.join(storage_dir, f"{stem}__merged__rank-*-of-*.pt"))
        if is_anchor_root(root):
            _extend_glob(os.path.join(storage_dir, f"*__merged__rank-{rank}-of-*.pt"))
            _extend_glob(os.path.join(storage_dir, f"*__merged__rank-*-of-*.pt"))

        part_patterns = [
            os.path.join(storage_dir, f"{stem}__samples-*__part-{part_id}__rank-{rank}-of-{world_size}.pt"),
            os.path.join(storage_dir, f"{stem}__samples-*__part-{part_id}__rank-{rank}-of-*.pt"),
            os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-{rank}-of-{world_size}.pt"),
            os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-{rank}-of-*.pt"),
            os.path.join(storage_dir, f"{stem}__samples-*__part-*__rank-*-of-*.pt"),
        ]
        if is_anchor_root(root):
            part_patterns.extend(
                [
                    os.path.join(storage_dir, f"*__samples-*__part-{part_id}__rank-{rank}-of-*.pt"),
                    os.path.join(storage_dir, f"*__samples-*__part-*__rank-{rank}-of-*.pt"),
                ]
            )
        for pattern in part_patterns:
            _extend_glob(pattern)

    return candidates


def resolve_tarp_save_paths(config, dataset_key: str, rank: int, world_size: int, sample_count: int, part_id: int) -> tuple[str, str]:
    param_view = get_tarp_param_view(config)
    explicit_dir = getattr(config, "tarp_dir", None)
    explicit_tag = getattr(config, "tarp_tag", None)
    storage_dir = str(explicit_dir) if not is_none_like(explicit_dir) else get_tarp_storage_dir(config, dataset_key)
    os.makedirs(storage_dir, exist_ok=True)
    if not is_none_like(explicit_tag):
        stem = str(explicit_tag).strip()
        part_name = f"{stem}__samples-{int(sample_count)}__part-{int(part_id)}__rank-{int(rank)}-of-{int(world_size)}.pt"
        merged_name = f"{stem}__merged__rank-{int(rank)}-of-{int(world_size)}.pt"
    else:
        part_name = build_tarp_part_filename(dataset_key, param_view, sample_count, part_id, rank, world_size)
        merged_name = build_tarp_merged_filename(dataset_key, param_view, rank, world_size)
    part_path = os.path.join(storage_dir, part_name)
    merged_path = os.path.join(storage_dir, merged_name)
    return part_path, merged_path


def build_tarp_state_meta(dataset_key: str, data_source: Optional[str], config, part_id: int, rank: int, world_size: int, valid_samples_used: int) -> dict:
    return {
        "dataset_key": dataset_key.upper(),
        "data_source": None if data_source is None else str(data_source),
        "tarp_mode": str(getattr(config, "tarp_mode", "TARP")),
        "param_view": get_tarp_param_view(config),
        "part_id": int(part_id),
        "rank": int(rank),
        "world_size": int(world_size),
        "valid_samples_used": int(valid_samples_used),
    }


def validate_tarp_state_dict(state_dict: dict, expected_param_names: Iterable[str], expected_config: Optional[dict] = None) -> None:
    meta = state_dict.get("_meta")
    if not isinstance(meta, dict):
        raise RuntimeError("Legacy TARP artifact detected: missing `_meta`. Orig-view loading requires newly generated artifacts.")
    if meta.get("param_view") != TARP_DEFAULT_PARAM_VIEW:
        raise RuntimeError(f"Unsupported TARP param_view={meta.get('param_view')!r}; expected `{TARP_DEFAULT_PARAM_VIEW}`.")

    theta_star = state_dict.get("theta_star", {})
    if not isinstance(theta_star, dict):
        raise RuntimeError("Malformed TARP artifact: `theta_star` is missing or not a dict.")

    expected = set(expected_param_names)
    found = set(theta_star.keys())
    if expected != found:
        missing = sorted(expected - found)[:8]
        extra = sorted(found - expected)[:8]
        raise RuntimeError(
            "TARP parameter mismatch for orig-view loading. "
            f"missing={missing} extra={extra} expected_count={len(expected)} found_count={len(found)}"
        )

    if expected_config:
        saved_cfg = state_dict.get("config", {})
        if not isinstance(saved_cfg, dict):
            raise RuntimeError("Malformed TARP artifact: missing `config` dictionary.")
        mismatches = {}
        for key, expected_value in expected_config.items():
            if expected_value is None:
                continue
            actual_value = saved_cfg.get(key)
            if actual_value != expected_value:
                mismatches[key] = {"expected": expected_value, "actual": actual_value}
        if mismatches:
            raise RuntimeError(f"TARP config mismatch for orig-view loading: {mismatches}")


def _stable_top_psd_eigensystem(
    cov: torch.Tensor,
    rank: int,
    *,
    vec_dtype: torch.dtype,
    val_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Online TARP consolidation can produce nearly singular PSD blocks. Keep the
    # merge numerically stable without changing the underlying objective.
    cov64 = cov.to(dtype=torch.float64)
    cov64 = 0.5 * (cov64 + cov64.t())
    dim = cov64.shape[0]
    target_rank = min(rank if rank > 0 else dim, dim)
    eye = torch.eye(dim, dtype=cov64.dtype, device=cov64.device)

    last_exc = None
    diag_scale = float(cov64.diagonal().abs().mean().item()) if dim > 0 else 0.0
    diag_scale = max(diag_scale, 1.0)
    for rel_jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try_cov = cov64
        if rel_jitter > 0.0:
            try_cov = cov64 + eye * (diag_scale * rel_jitter)
        try:
            e, v = torch.linalg.eigh(try_cov)
            top_e = e[-target_rank:].flip(0).clamp_min(1e-6).to(dtype=val_dtype)
            top_v = v[:, -target_rank:].flip(1).to(dtype=vec_dtype)
            return top_e, top_v
        except torch._C._LinAlgError as exc:
            last_exc = exc

    # SVD is more robust for ill-conditioned PSD blocks and yields the same
    # principal subspace after symmetrization.
    try:
        u, s, _ = torch.linalg.svd(cov64, full_matrices=False)
        top_e = s[:target_rank].clamp_min(1e-6).to(dtype=val_dtype)
        top_v = u[:, :target_rank].to(dtype=vec_dtype)
        return top_e, top_v
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise


def consolidate_tarp_state_dict(previous_state: dict, current_state: dict, decay: float = 1.0) -> dict:
    decay = float(decay)
    if decay < 0:
        raise ValueError(f"decay must be >= 0, got {decay}")

    if not previous_state:
        return current_state

    current_theta = current_state.get("theta_star", {})
    previous_theta = previous_state.get("theta_star", {})
    if set(previous_theta.keys()) != set(current_theta.keys()):
        raise RuntimeError(
            "TARP online consolidation parameter mismatch: "
            f"prev={len(previous_theta)} current={len(current_theta)}"
        )

    current_R = current_state.get("R_cpu", {})
    previous_R = previous_state.get("R_cpu", {})
    for name, curr_R in current_R.items():
        prev_R = previous_R.get(name)
        if prev_R is None:
            raise RuntimeError(f"TARP online consolidation missing projection matrix for {name}")
        if prev_R.shape != curr_R.shape or not torch.allclose(prev_R.float(), curr_R.float(), atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"TARP online consolidation projection mismatch for {name}")

    result = dict(current_state)
    result["theta_star"] = {k: v.cpu() for k, v in current_theta.items()}
    result["R_cpu"] = {k: v.cpu() for k, v in current_R.items()}

    prev_cfg = dict(previous_state.get("config", {}))
    curr_cfg = dict(current_state.get("config", {}))
    merged_cfg = dict(prev_cfg)
    merged_cfg.update(curr_cfg)
    result["config"] = merged_cfg

    current_fisher = current_state.get("fisher_diag_cpu", {})
    previous_fisher = previous_state.get("fisher_diag_cpu", {})
    merged_fisher = {}
    for name, curr_fd in current_fisher.items():
        prev_fd = previous_fisher.get(name)
        merged = curr_fd.float()
        if prev_fd is not None:
            if prev_fd.shape != curr_fd.shape:
                raise RuntimeError(f"TARP online consolidation fisher shape mismatch for {name}")
            merged = merged + decay * prev_fd.float()
        merged_fisher[name] = merged.clamp_min(0.0).to(dtype=curr_fd.dtype).cpu()
    result["fisher_diag_cpu"] = merged_fisher

    rank_per_block = int(merged_cfg.get("rank_per_block", 0))
    current_subspaces = current_state.get("subspaces_cpu", {})
    previous_subspaces = previous_state.get("subspaces_cpu", {})
    merged_subspaces = {}
    for name in sorted(set(previous_subspaces.keys()) | set(current_subspaces.keys())):
        curr_blocks = current_subspaces.get(name)
        prev_blocks = previous_subspaces.get(name)

        # With very small smoke runs it is possible to finish a stage with zero
        # valid TARP samples. In that case we still want to carry the prior
        # curvature forward while re-centering theta_star on the current model.
        if not curr_blocks:
            if prev_blocks is not None:
                merged_subspaces[name] = [(U.cpu(), lam.cpu()) for (U, lam) in prev_blocks]
            continue
        if prev_blocks is None:
            merged_subspaces[name] = [(U.cpu(), lam.cpu()) for (U, lam) in curr_blocks]
            continue
        if len(prev_blocks) != len(curr_blocks):
            raise RuntimeError(f"TARP online consolidation block count mismatch for {name}")

        merged_blocks = []
        for (prev_U, prev_lam), (curr_U, curr_lam) in zip(prev_blocks, curr_blocks):
            if prev_U.shape[0] != curr_U.shape[0]:
                raise RuntimeError(f"TARP online consolidation block shape mismatch for {name}")

            cov = (curr_U.float() * curr_lam.float().unsqueeze(0)) @ curr_U.float().t()
            cov = cov + decay * ((prev_U.float() * prev_lam.float().unsqueeze(0)) @ prev_U.float().t())
            r = min(rank_per_block if rank_per_block > 0 else curr_U.shape[1], curr_U.shape[1])
            top_e, top_v = _stable_top_psd_eigensystem(
                cov,
                r,
                vec_dtype=curr_U.dtype,
                val_dtype=curr_lam.dtype,
            )
            merged_blocks.append((top_v.cpu(), top_e.cpu()))
        merged_subspaces[name] = merged_blocks
    result["subspaces_cpu"] = merged_subspaces

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
                    f"TARP online consolidation would double-count experiment {current_experiment!r}. "
                    "Use a fresh tarp_tag/tarp_dir when rebuilding the same stage."
                )
            history.append(current_experiment)
        meta["online_tarp"] = True
        meta["online_decay"] = decay
        meta["online_consolidated_from_dataset_key"] = prev_meta.get("dataset_key")
        meta["online_consolidated_from_valid_samples"] = prev_meta.get("valid_samples_used")
        meta["online_history"] = history
        meta["online_stage_count"] = len(history)
        result["_meta"] = meta

    return result

@dataclass
class MatBlockMeta:
    name: str
    param: torch.nn.Parameter
    orig_shape: Tuple[int, ...]
    out_dim: int
    in_dim: int
    block_size: int
    proj_dim: int

@dataclass
class VecMeta:
    name: str
    param: torch.nn.Parameter
    orig_shape: Tuple[int, ...]
    dim: int

class BlockExactSubspaceTARP:
    """
    Practical exact subspace TARP (block-wise):
      - Matrix params: block-wise covariance over output dimension.
      - Input dimension compressed by multiple Rademacher projections (proj_dim = d).
      - Vector params (bias, LayerNorm): diagonal Fisher.
      - CPU accumulation + optional GPU caching for training-time penalty.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        param_names: Optional[List[str]] = None,
        rank_per_block: int = 16,
        block_size: int = 1024,
        proj_dim: int = 8,
        seed: int = 42,
        cov_dtype: torch.dtype = torch.float32,
        store_dtype: torch.dtype = torch.float16,
        normalize_proj_columns: bool = True,
        average_over_proj_dim: bool = True,
    ):
        self.model = model
        self.rank_per_block = int(rank_per_block)
        self.block_size = int(block_size)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.cov_dtype = cov_dtype
        self.store_dtype = store_dtype
        self.normalize_proj_columns = normalize_proj_columns
        self.average_over_proj_dim = average_over_proj_dim

        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if param_names is not None:
            allow = set(param_names)
            named_params = [(n, p) for n, p in named_params if n in allow]

        if not named_params:
            raise ValueError("No trainable parameters selected.")

        self.mat_metas: List[MatBlockMeta] = []
        self.vec_metas: List[VecMeta] = []

        # CPU storage
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.sample_count: int = 0
        self.R_cpu: Dict[str, torch.Tensor] = {}
        self.cov_acc: Dict[str, List[torch.Tensor]] = {}
        self.subspaces_cpu: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.fisher_diag_cpu: Dict[str, torch.Tensor] = {}

        # GPU caches
        self._cached_device: Optional[torch.device] = None
        self.R_gpu: Dict[str, torch.Tensor] = {}
        self.subspaces_gpu: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.fisher_diag_gpu: Dict[str, torch.Tensor] = {}
        self.theta_star_gpu: Dict[str, torch.Tensor] = {}
        try:
            first_param = next(model.parameters())
            self._eig_device = first_param.device if first_param.is_cuda else torch.device("cpu")
        except StopIteration:
            self._eig_device = torch.device("cpu")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        for name, p in named_params:
            shape = tuple(p.shape)
            if len(shape) >= 2:
                out_dim = shape[0]
                in_dim = int(torch.tensor(shape[1:]).prod().item())

                # Rademacher Projection
                R = torch.empty((in_dim, self.proj_dim), dtype=torch.float32, device="cpu")
                R.bernoulli_(0.5, generator=gen).mul_(2.0).sub_(1.0)
                R = R / (in_dim ** 0.5)
                self.R_cpu[name] = R

                n_blocks = (out_dim + self.block_size - 1) // self.block_size
                self.cov_acc[name] = []
                for bi in range(n_blocks):
                    b0 = bi * self.block_size
                    b1 = min(out_dim, (bi + 1) * self.block_size)
                    b = b1 - b0
                    self.cov_acc[name].append(torch.zeros((b, b), dtype=self.cov_dtype, device="cpu"))

                self.mat_metas.append(MatBlockMeta(name, p, shape, out_dim, in_dim, self.block_size, self.proj_dim))
            else:
                dim = p.numel()
                self.vec_metas.append(VecMeta(name, p, shape, dim))
                self.fisher_diag_cpu[name] = torch.zeros((dim,), dtype=torch.float32, device="cpu")

    def _current_named_params(self) -> Dict[str, torch.nn.Parameter]:
        param_map: Dict[str, torch.nn.Parameter] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            for alias in _param_name_aliases(name):
                param_map.setdefault(alias, param)
        return param_map

    def _current_param(self, name: str, param_map: Optional[Dict[str, torch.nn.Parameter]] = None) -> torch.nn.Parameter:
        current_map = self._current_named_params() if param_map is None else param_map
        if name not in current_map:
            raise RuntimeError(f"TARP runtime parameter {name!r} is missing from current model.named_parameters().")
        return current_map[name]

    @staticmethod
    def _shape_tuple(param: torch.nn.Parameter) -> Tuple[int, ...]:
        return tuple(param.shape)

    def validate_state_dict_shapes(self, state_dict: dict) -> None:
        theta_star = state_dict.get("theta_star", {})
        fisher_diag = state_dict.get("fisher_diag_cpu", {})
        subspaces = state_dict.get("subspaces_cpu", {})
        proj = state_dict.get("R_cpu", {})

        expected_mat = {m.name: m for m in self.mat_metas}
        expected_vec = {v.name: v for v in self.vec_metas}
        saved_mat = set(subspaces.keys())
        saved_vec = set(fisher_diag.keys())

        if saved_mat != set(expected_mat.keys()):
            missing = sorted(set(expected_mat.keys()) - saved_mat)[:8]
            extra = sorted(saved_mat - set(expected_mat.keys()))[:8]
            raise RuntimeError(
                "TARP matrix parameter partition mismatch for orig-view loading. "
                f"missing={missing} extra={extra} expected_count={len(expected_mat)} found_count={len(saved_mat)}"
            )
        if saved_vec != set(expected_vec.keys()):
            missing = sorted(set(expected_vec.keys()) - saved_vec)[:8]
            extra = sorted(saved_vec - set(expected_vec.keys()))[:8]
            raise RuntimeError(
                "TARP vector parameter partition mismatch for orig-view loading. "
                f"missing={missing} extra={extra} expected_count={len(expected_vec)} found_count={len(saved_vec)}"
            )

        for name, meta in expected_mat.items():
            theta = theta_star.get(name)
            if theta is None:
                raise RuntimeError(f"TARP artifact missing theta_star for matrix param {name}")
            if tuple(theta.shape) != meta.orig_shape:
                raise RuntimeError(
                    f"TARP theta_star shape mismatch for matrix param {name}: "
                    f"saved={tuple(theta.shape)} expected={meta.orig_shape}"
                )

            R = proj.get(name)
            expected_R_shape = (meta.in_dim, self.proj_dim)
            if R is None or tuple(R.shape) != expected_R_shape:
                raise RuntimeError(
                    f"TARP projection shape mismatch for {name}: "
                    f"saved={None if R is None else tuple(R.shape)} expected={expected_R_shape}"
                )

            blocks = subspaces.get(name)
            expected_blocks = (meta.out_dim + meta.block_size - 1) // meta.block_size
            if blocks is None or len(blocks) != expected_blocks:
                raise RuntimeError(
                    f"TARP block count mismatch for {name}: "
                    f"saved={0 if blocks is None else len(blocks)} expected={expected_blocks}"
                )

            for bi, (U, lam) in enumerate(blocks):
                b0 = bi * meta.block_size
                b1 = min(meta.out_dim, (bi + 1) * meta.block_size)
                block_rows = b1 - b0
                rank = min(self.rank_per_block, block_rows)
                if U.dim() != 2 or tuple(U.shape) != (block_rows, lam.numel()):
                    raise RuntimeError(
                        f"TARP subspace shape mismatch for {name} block {bi}: "
                        f"saved_U={tuple(U.shape)} saved_lam={tuple(lam.shape)} expected_rows={block_rows}"
                    )
                if lam.dim() != 1 or lam.numel() > rank:
                    raise RuntimeError(
                        f"TARP eigenvalue shape mismatch for {name} block {bi}: "
                        f"saved={tuple(lam.shape)} expected_rank<={rank}"
                    )

        for name, meta in expected_vec.items():
            theta = theta_star.get(name)
            if theta is None:
                raise RuntimeError(f"TARP artifact missing theta_star for vector param {name}")
            if tuple(theta.shape) != meta.orig_shape:
                raise RuntimeError(
                    f"TARP theta_star shape mismatch for vector param {name}: "
                    f"saved={tuple(theta.shape)} expected={meta.orig_shape}"
                )

            fd = fisher_diag.get(name)
            expected_fd_shape = (meta.dim,)
            if fd is None or tuple(fd.shape) != expected_fd_shape:
                raise RuntimeError(
                    f"TARP diagonal fisher shape mismatch for {name}: "
                    f"saved={None if fd is None else tuple(fd.shape)} expected={expected_fd_shape}"
                )

    @torch.no_grad()
    def snapshot_theta_star(self):
        param_map = self._current_named_params()
        self.theta_star = {}
        for m in self.mat_metas:
            p = self._current_param(m.name, param_map=param_map)
            if self._shape_tuple(p) != m.orig_shape:
                raise RuntimeError(
                    f"TARP snapshot expected full matrix param view for {m.name}, "
                    f"got shape={self._shape_tuple(p)} expected={m.orig_shape}"
                )
            self.theta_star[m.name] = p.detach().cpu().clone().to(dtype=self.store_dtype)
        for v in self.vec_metas:
            p = self._current_param(v.name, param_map=param_map)
            if self._shape_tuple(p) != v.orig_shape:
                raise RuntimeError(
                    f"TARP snapshot expected full vector param view for {v.name}, "
                    f"got shape={self._shape_tuple(p)} expected={v.orig_shape}"
                )
            self.theta_star[v.name] = p.detach().cpu().clone().to(dtype=self.store_dtype)
        # Clear GPU cache
        self._cached_device = None; self.R_gpu.clear(); self.subspaces_gpu.clear(); self.fisher_diag_gpu.clear(); self.theta_star_gpu.clear()

    @torch.no_grad()
    def accumulate_fisher_after_backward(self):
        self.sample_count += 1
        param_map = self._current_named_params()
        # Vectors
        for v in self.vec_metas:
            p = self._current_param(v.name, param_map=param_map)
            if self._shape_tuple(p) != v.orig_shape:
                raise RuntimeError(
                    f"TARP fisher accumulation expected full vector param view for {v.name}, "
                    f"got shape={self._shape_tuple(p)} expected={v.orig_shape}"
                )
            if p.grad is None:
                continue
            g = p.grad.detach().reshape(-1).float().cpu()
            self.fisher_diag_cpu[v.name] += g.pow(2)

        # Matrices
        alpha_proj = (1.0 / float(self.proj_dim)) if self.average_over_proj_dim else 1.0
        for m in self.mat_metas:
            p = self._current_param(m.name, param_map=param_map)
            if self._shape_tuple(p) != m.orig_shape:
                raise RuntimeError(
                    f"TARP fisher accumulation expected full matrix param view for {m.name}, "
                    f"got shape={self._shape_tuple(p)} expected={m.orig_shape}"
                )
            if p.grad is None:
                continue
            g = p.grad.detach().reshape(m.out_dim, m.in_dim)
            R = self.R_cpu[m.name].to(device=g.device, dtype=g.dtype, non_blocking=True)
            n_blocks = len(self.cov_acc[m.name])
            if self.normalize_proj_columns:
                col_sq_sum = torch.zeros((1, self.proj_dim), dtype=torch.float32, device="cpu")
                for bi in range(n_blocks):
                    b0 = bi * m.block_size
                    b1 = min(m.out_dim, (bi + 1) * m.block_size)
                    proj_block = torch.matmul(g[b0:b1, :], R).float().cpu()
                    col_sq_sum.add_(proj_block.pow(2).sum(dim=0, keepdim=True))
                col_norm = col_sq_sum.sqrt_().clamp_min_(1e-6)
            else:
                col_norm = None

            for bi in range(n_blocks):
                b0 = bi * m.block_size
                b1 = min(m.out_dim, (bi + 1) * m.block_size)
                block = torch.matmul(g[b0:b1, :], R).float().cpu()
                if col_norm is not None:
                    block = block / col_norm
                self.cov_acc[m.name][bi].addmm_(block, block.t(), beta=1.0, alpha=alpha_proj)

    # @torch.no_grad()
    # def finalize_fisher(self):
    #     if self.sample_count <= 0: return
    #     print(f"[BlockExactCTS] Finalizing with {self.sample_count} samples...")

    #     # Normalize Vectors
    #     for name in list(self.fisher_diag_cpu.keys()):
    #         self.fisher_diag_cpu[name] = (self.fisher_diag_cpu[name] / float(self.sample_count)).to(dtype=self.store_dtype)

    #     # PCA for Matrices
    #     self.subspaces_cpu = {}
    #     for m in self.mat_metas:
    #         self.subspaces_cpu[m.name] = []
    #         blocks = self.cov_acc.get(m.name, [])
    #         for cov in blocks:
    #             cov = cov / float(self.sample_count)
    #             try:
    #                 e, v = torch.linalg.eigh(cov)
    #                 r = min(self.rank_per_block, v.shape[1])
    #                 top_e = e[-r:].flip(0).clamp_min(1e-6).to(dtype=self.store_dtype)
    #                 top_v = v[:, -r:].flip(1).to(dtype=self.store_dtype)
    #                 self.subspaces_cpu[m.name].append((top_v, top_e))
    #             except Exception as e:
    #                 print(f"PCA failed for {m.name}: {e}")
    #         self.cov_acc[m.name] = [] # Free RAM

    #     self.cov_acc = {}
    #     gc.collect()

    # @torch.no_grad()
    # def cache_to_device(self, device: Union[str, torch.device], dtype: torch.dtype = torch.float16):
    #     dev = torch.device(device)
    #     self._cached_device = dev
    #     self.R_gpu = {k: v.to(dev, dtype, non_blocking=True) for k, v in self.R_cpu.items()}
    #     self.fisher_diag_gpu = {k: v.to(dev, dtype, non_blocking=True) for k, v in self.fisher_diag_cpu.items()}
    #     self.theta_star_gpu = {k: v.to(dev, dtype, non_blocking=True) for k, v in self.theta_star.items()}
    #     self.subspaces_gpu = {}
    #     for k, v_list in self.subspaces_cpu.items():
    #         self.subspaces_gpu[k] = [(U.to(dev, dtype, non_blocking=True), lam.to(dev, dtype, non_blocking=True)) for U, lam in v_list]

    # def penalty(self, model: Optional[torch.nn.Module] = None, coef: float = 1.0) -> torch.Tensor:
    #     coef = float(coef)
    #     total = None

    #     # vector diagonal penalty
    #     for v in self.vec_metas:
    #         p = v.param
    #         if self._cached_device:
    #             ts = self.theta_star_gpu[v.name]
    #             fd = self.fisher_diag_gpu[v.name]
    #         else:
    #             ts = self.theta_star[v.name].to(p.device)
    #             fd = self.fisher_diag_cpu[v.name].to(p.device)

    #         import torch
    #         import torch.distributed as dist

    #         rank = dist.get_rank() if dist.is_initialized() else 0

    #         # --- 先算 delta ---
    #         delta = p.reshape(-1) - ts.to(dtype=p.dtype)

    #         # --- 只在 rank1、只对 flat_param 打一次 debug ---
    #         if rank == 1 and (v.name.endswith("_flat_param") or v.name == "_fsdp_wrapped_module._flat_param"):
    #             if not hasattr(self, "_dbg_fp16_once"):
    #                 self._dbg_fp16_once = set()
    #             if v.name not in self._dbg_fp16_once:
    #                 self._dbg_fp16_once.add(v.name)

    #                 # fp16 路径（和你现在 cache_to_device 默认一致）
    #                 prod16 = fd.to(dtype=torch.float16) * delta.to(dtype=torch.float16).pow(2)
    #                 s16 = prod16.sum()

    #                 # fp32 参考路径（用来判断是不是 fp16 sum 溢出）
    #                 prod32 = fd.to(dtype=torch.float32) * delta.to(dtype=torch.float32).pow(2)
    #                 s32 = prod32.sum()

    #                 print(
    #                     f"[TARP DBG] rank{rank} {v.name} | "
    #                     f"sum_fp16={s16.item()} finite16={torch.isfinite(s16).item()} | "
    #                     f"sum_fp32={s32.item()} finite32={torch.isfinite(s32).item()} | "
    #                     f"max_fd={fd.float().max().item()} max_delta2={(delta.float().pow(2)).max().item()} max_prod32={prod32.max().item()}",
    #                     flush=True
    #                 )

    #         delta = (p.reshape(-1) - ts.to(dtype=p.dtype))
    #         term = 0.5 * (fd.to(dtype=delta.dtype) * delta.pow(2)).sum()
    #         total = term if total is None else (total + term)

    #     # matrix block subspace penalty
    #     alpha_proj = (1.0 / float(self.proj_dim)) if self.average_over_proj_dim else 1.0

    #     for m in self.mat_metas:
    #         name = m.name
    #         # Check if this layer has subspace computed
    #         source_subspaces = self.subspaces_gpu if self._cached_device else self.subspaces_cpu
    #         if name not in source_subspaces: continue

    #         p = m.param
    #         if self._cached_device:
    #             ts = self.theta_star_gpu[name]
    #             R = self.R_gpu[name]
    #             subspace_blocks = self.subspaces_gpu[name]
    #         else:
    #             ts = self.theta_star[name].to(p.device)
    #             R = self.R_cpu[name].to(p.device)
    #             subspace_blocks = self.subspaces_cpu[name]

    #         delta = (p - ts.to(dtype=p.dtype)).reshape(m.out_dim, m.in_dim)
    #         delta_proj = torch.matmul(delta, R.to(dtype=delta.dtype))  # (out_dim, d)

    #         for bi, (U, lam) in enumerate(subspace_blocks):
    #             b0 = bi * m.block_size
    #             b1 = min(m.out_dim, (bi + 1) * m.block_size)

    #             # Z: (b, d) - d columns of projected drift
    #             Z = delta_proj[b0:b1, :]
    #             U_ = U.to(dtype=Z.dtype)
    #             lam_ = lam.to(dtype=Z.dtype)

    #             # Project each column k into subspace: P_k = U^T Z_k
    #             # P: (r, d)
    #             P = torch.matmul(U_.t(), Z)

    #             # Penalty: 0.5 * sum_{k=1..d} [ (P_k)^T Lambda P_k ]
    #             #        = 0.5 * sum_{k, j} lam_j * (P_{jk})^2
    #             term = 0.5 * (lam_.unsqueeze(1) * P.pow(2)).sum() * alpha_proj
    #             total = term if total is None else (total + term)

    #     if total is None:
    #         dev = next(self.mat_metas[0].param).device if self.mat_metas else torch.device("cpu")
    #         return torch.zeros((), device=dev)

    #     return total * coef

    @torch.no_grad()
    def finalize_fisher(self):
        if self.sample_count <= 0:
            return
        print(f"[BlockExactCTS] Finalizing with {self.sample_count} samples...")

        def _top_block_eig(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            cov32 = cov.float()
            cov32 = 0.5 * (cov32 + cov32.t())
            eig_device = self._eig_device if getattr(self, "_eig_device", torch.device("cpu")).type != "cpu" else cov32.device
            cov_work = cov32.to(device=eig_device, non_blocking=True) if eig_device != cov32.device else cov32
            r = min(self.rank_per_block, cov_work.shape[1])
            if r <= 0:
                raise RuntimeError("TARP block rank resolved to zero during finalize_fisher.")

            # CPU full eigh on every 2048x2048 block is very slow for 3B models.
            # On GPU, full eigh is typically faster and more stable than lobpcg for these block sizes.
            if cov_work.device.type == "cpu" and cov_work.shape[0] >= 512 and r < cov_work.shape[0]:
                try:
                    X = torch.randn((cov_work.shape[0], r), dtype=cov_work.dtype, device=cov_work.device)
                    eigvals, eigvecs = torch.lobpcg(cov_work, k=r, X=X, niter=40, tol=1e-4, largest=True)
                    order = torch.argsort(eigvals, descending=True)
                    return eigvecs[:, order].to("cpu"), eigvals[order].to("cpu")
                except Exception as ex:
                    print(f"[BlockExactCTS] lobpcg fallback to eigh (block={tuple(cov_work.shape)}): {ex}", flush=True)

            try:
                eigvals, eigvecs = torch.linalg.eigh(cov_work)
                return eigvecs[:, -r:].flip(1).to("cpu"), eigvals[-r:].flip(0).to("cpu")
            except Exception as ex:
                # Single-sample smoke runs can yield highly rank-deficient blocks that occasionally
                # make GPU eigh unhappy. Retry on CPU with a small diagonal jitter before giving up.
                print(f"[BlockExactCTS] eigh fallback to CPU+jitter (block={tuple(cov_work.shape)}): {ex}", flush=True)
                cov_cpu = cov32.cpu()
                cov_cpu = 0.5 * (cov_cpu + cov_cpu.t())
                jitter = max(1e-6, float(cov_cpu.diagonal().abs().max().item()) * 1e-6)
                cov_cpu.diagonal().add_(jitter)
                try:
                    eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
                    return eigvecs[:, -r:].flip(1), eigvals[-r:].flip(0)
                except Exception as cpu_ex:
                    # Preserve block structure even if decomposition is numerically unstable.
                    print(
                        f"[BlockExactCTS] CPU+jitter eigh still failed (block={tuple(cov_cpu.shape)}): {cpu_ex}. "
                        "Using identity fallback.",
                        flush=True,
                    )
                    eye = torch.eye(cov_cpu.shape[0], dtype=torch.float32)
                    fallback_vecs = eye[:, :r].contiguous()
                    fallback_vals = torch.full((r,), 1e-6, dtype=torch.float32)
                    return fallback_vecs, fallback_vals

        # -------- helper: sanitize + safe cast --------
        def _sanitize_and_cast(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
            if not torch.is_floating_point(x):
                return x
            x32 = x.float()
            # 把 NaN/Inf 变成有限数，避免保存后污染文件
            finfo32 = torch.finfo(torch.float32)
            x32 = torch.nan_to_num(x32, nan=0.0, posinf=finfo32.max, neginf=0.0)

            # 如果你仍然坚持存 fp16，则必须 clamp，否则一定会 inf
            if out_dtype == torch.float16:
                x32 = x32.clamp_max(torch.finfo(torch.float16).max)

            return x32.to(dtype=out_dtype)

        # 1) Normalize Vectors (diag fisher)
        for name in list(self.fisher_diag_cpu.keys()):
            fd = self.fisher_diag_cpu[name] / float(self.sample_count)
            fd = fd.clamp_min(0.0)
            self.fisher_diag_cpu[name] = _sanitize_and_cast(fd, self.store_dtype)

        # 2) PCA for Matrices
        self.subspaces_cpu = {}
        for m in self.mat_metas:
            self.subspaces_cpu[m.name] = []
            blocks = self.cov_acc.get(m.name, [])
            for cov in blocks:
                cov = cov / float(self.sample_count)
                try:
                    top_v, top_e = _top_block_eig(cov)
                    top_e = top_e.clamp_min(1e-6)

                    self.subspaces_cpu[m.name].append(
                        (_sanitize_and_cast(top_v, self.store_dtype),
                         _sanitize_and_cast(top_e, self.store_dtype))
                    )
                except Exception as ex:
                    print(f"PCA failed for {m.name}: {ex}")

            self.cov_acc[m.name] = []

        self.cov_acc = {}
        gc.collect()

        # 如果之前缓存过 GPU，finalize 后让缓存失效，避免 dtype/数值脏数据继续用
        self._cached_device = None
        self.R_gpu = {}
        self.subspaces_gpu = {}
        self.fisher_diag_gpu = {}
        self.theta_star_gpu = {}

    @torch.no_grad()
    def cache_to_device(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None):
        """
        关键点：
        - 不要默认 float16
        - dtype=None 时用 self.store_dtype（你在 estimate_fisher_remote 里会设成 bf16）
        - cache 前做 nan/inf 修复，避免历史脏文件直接把训练弄崩
        """
        dev = torch.device(device)
        self._cached_device = dev
        dtype = self.store_dtype if dtype is None else dtype

        def _to_dev(x: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(x):
                return x.to(device=dev, non_blocking=True)
            x32 = x.float()
            finfo32 = torch.finfo(torch.float32)
            x32 = torch.nan_to_num(x32, nan=0.0, posinf=finfo32.max, neginf=0.0)
            if dtype == torch.float16:
                x32 = x32.clamp_max(torch.finfo(torch.float16).max)
            return x32.to(device=dev, dtype=dtype, non_blocking=True)

        self.R_gpu = {k: _to_dev(v) for k, v in self.R_cpu.items()}
        self.fisher_diag_gpu = {k: _to_dev(v) for k, v in self.fisher_diag_cpu.items()}
        self.theta_star_gpu = {k: _to_dev(v) for k, v in self.theta_star.items()}

        self.subspaces_gpu = {}
        for k, v_list in self.subspaces_cpu.items():
            self.subspaces_gpu[k] = [(_to_dev(U), _to_dev(lam)) for (U, lam) in v_list]

    def _penalty_impl(self, model: Optional[torch.nn.Module] = None, coef: float = 1.0, collect_breakdown: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        关键点：
        - 不要把 fd/lam 强行 cast 回 fp16 再乘
        - 用 float32 做乘法与求和，避免溢出
        """
        coef = float(coef)
        total = None
        vector_total = None
        matrix_total = None
        param_map = self._current_named_params() if model is None else {name: param for name, param in model.named_parameters() if param.requires_grad}

        vector_layer_terms: Dict[str, float] = {}
        matrix_layer_terms: Dict[str, float] = {}
        layer_delta_stats: Dict[str, Dict[str, float]] = {}

        # vector diagonal penalty (fp32 accumulate)
        for v in self.vec_metas:
            p = self._current_param(v.name, param_map=param_map)
            if self._shape_tuple(p) != v.orig_shape:
                raise RuntimeError(
                    f"TARP penalty expected full vector param view for {v.name}, "
                    f"got shape={self._shape_tuple(p)} expected={v.orig_shape}"
                )
            if self._cached_device:
                ts = self.theta_star_gpu[v.name]
                fd = self.fisher_diag_gpu[v.name]
            else:
                ts = self.theta_star[v.name].to(p.device)
                fd = self.fisher_diag_cpu[v.name].to(p.device)

            delta = (p.reshape(-1) - ts.to(dtype=p.dtype))
            term = 0.5 * (fd.float() * delta.float().pow(2)).sum()
            total = term if total is None else (total + term)
            if collect_breakdown:
                vector_total = term if vector_total is None else (vector_total + term)
                vector_layer_terms[v.name] = float(term.detach().float().item())
                delta_det = delta.detach().float()
                layer_delta_stats[v.name] = {
                    "group": "vector",
                    "l2": float(torch.norm(delta_det, p=2).item()),
                    "mean_abs": float(delta_det.abs().mean().item()),
                    "max_abs": float(delta_det.abs().max().item()),
                }

        # matrix block subspace penalty (fp32)
        alpha_proj = (1.0 / float(self.proj_dim)) if self.average_over_proj_dim else 1.0

        for m in self.mat_metas:
            name = m.name
            source_subspaces = self.subspaces_gpu if self._cached_device else self.subspaces_cpu
            if name not in source_subspaces:
                continue

            p = self._current_param(name, param_map=param_map)
            if self._shape_tuple(p) != m.orig_shape:
                raise RuntimeError(
                    f"TARP penalty expected full matrix param view for {name}, "
                    f"got shape={self._shape_tuple(p)} expected={m.orig_shape}"
                )
            if self._cached_device:
                ts = self.theta_star_gpu[name]
                R = self.R_gpu[name]
                subspace_blocks = self.subspaces_gpu[name]
            else:
                ts = self.theta_star[name].to(p.device)
                R = self.R_cpu[name].to(p.device)
                subspace_blocks = self.subspaces_cpu[name]

            layer_total = None
            if collect_breakdown:
                delta_sq_sum = 0.0
                delta_abs_sum = 0.0
                delta_max_abs = 0.0
                delta_count = 0

            p_matrix = p.reshape(m.out_dim, m.in_dim)
            ts_matrix = ts.reshape(m.out_dim, m.in_dim)
            R_ = R.to(device=p.device, dtype=p.dtype)

            for bi, (U, lam) in enumerate(subspace_blocks):
                b0 = bi * m.block_size
                b1 = min(m.out_dim, (bi + 1) * m.block_size)

                delta_block = p_matrix[b0:b1, :] - ts_matrix[b0:b1, :].to(dtype=p.dtype)
                Z = torch.matmul(delta_block, R_).float()
                U_ = U.to(device=Z.device, dtype=torch.float32)
                lam_ = lam.to(device=Z.device, dtype=torch.float32)

                P = torch.matmul(U_.t(), Z)  # (r, d)
                term = 0.5 * (lam_.unsqueeze(1) * P.pow(2)).sum() * alpha_proj
                total = term if total is None else (total + term)
                if collect_breakdown:
                    layer_total = term if layer_total is None else (layer_total + term)
                    delta_det = delta_block.detach().float()
                    delta_sq_sum += float(delta_det.pow(2).sum().item())
                    delta_abs_sum += float(delta_det.abs().sum().item())
                    delta_max_abs = max(delta_max_abs, float(delta_det.abs().max().item()))
                    delta_count += int(delta_det.numel())

            if collect_breakdown and layer_total is not None:
                matrix_total = layer_total if matrix_total is None else (matrix_total + layer_total)
                matrix_layer_terms[name] = float(layer_total.detach().float().item())
                layer_delta_stats[name] = {
                    "group": "matrix",
                    "l2": float(delta_sq_sum ** 0.5),
                    "mean_abs": float(delta_abs_sum / max(delta_count, 1)),
                    "max_abs": float(delta_max_abs),
                }

        if total is None:
            if param_map:
                dev = next(iter(param_map.values())).device
            else:
                dev = torch.device("cpu")
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            if not collect_breakdown:
                return zero
            breakdown = {
                "vector_total": 0.0,
                "matrix_total": 0.0,
                "total": 0.0,
                "vector_layer_terms": vector_layer_terms,
                "matrix_layer_terms": matrix_layer_terms,
                "delta_stats": layer_delta_stats,
            }
            return zero, breakdown

        total = total * coef
        if not collect_breakdown:
            return total

        vector_total_val = float((vector_total * coef).detach().float().item()) if vector_total is not None else 0.0
        matrix_total_val = float((matrix_total * coef).detach().float().item()) if matrix_total is not None else 0.0
        breakdown = {
            "vector_total": vector_total_val,
            "matrix_total": matrix_total_val,
            "total": float(total.detach().float().item()),
            "vector_layer_terms": vector_layer_terms,
            "matrix_layer_terms": matrix_layer_terms,
            "delta_stats": layer_delta_stats,
        }
        return total, breakdown

    def penalty(self, model: Optional[torch.nn.Module] = None, coef: float = 1.0) -> torch.Tensor:
        return self._penalty_impl(model=model, coef=coef, collect_breakdown=False)

    def penalty_with_breakdown(self, model: Optional[torch.nn.Module] = None, coef: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = self._penalty_impl(model=model, coef=coef, collect_breakdown=True)
        if isinstance(out, tuple):
            return out
        return out, {}

    @torch.no_grad()
    def state_magnitude_summary(self) -> Dict[str, Any]:
        fisher_stats: Dict[str, Dict[str, float]] = {}
        fisher_sum = 0.0
        fisher_count = 0
        fisher_max = 0.0
        for name, fd in self.fisher_diag_cpu.items():
            x = fd.detach().float()
            if x.numel() == 0:
                continue
            layer_mean = float(x.mean().item())
            layer_max = float(x.max().item())
            fisher_stats[name] = {"mean": layer_mean, "max": layer_max, "numel": int(x.numel())}
            fisher_sum += float(x.sum().item())
            fisher_count += int(x.numel())
            fisher_max = max(fisher_max, layer_max)

        lam_stats: Dict[str, Dict[str, float]] = {}
        lam_sum = 0.0
        lam_count = 0
        lam_max = 0.0
        for name, blocks in self.subspaces_cpu.items():
            if not blocks:
                continue
            layer_sum = 0.0
            layer_count = 0
            layer_max = 0.0
            for _, lam in blocks:
                lam_x = lam.detach().float()
                if lam_x.numel() == 0:
                    continue
                layer_sum += float(lam_x.sum().item())
                layer_count += int(lam_x.numel())
                layer_max = max(layer_max, float(lam_x.max().item()))
            if layer_count == 0:
                continue
            layer_mean = layer_sum / float(layer_count)
            lam_stats[name] = {"mean": layer_mean, "max": layer_max, "numel": layer_count}
            lam_sum += layer_sum
            lam_count += layer_count
            lam_max = max(lam_max, layer_max)

        return {
            "fisher_global_mean": (fisher_sum / float(fisher_count)) if fisher_count > 0 else 0.0,
            "fisher_global_max": fisher_max,
            "fisher_count": fisher_count,
            "fisher_layer_stats": fisher_stats,
            "lam_global_mean": (lam_sum / float(lam_count)) if lam_count > 0 else 0.0,
            "lam_global_max": lam_max,
            "lam_count": lam_count,
            "lam_layer_stats": lam_stats,
        }


    def state_dict(self):
        return {
            "subspaces_cpu": self.subspaces_cpu,
            "fisher_diag_cpu": self.fisher_diag_cpu,
            "theta_star": self.theta_star,
            "R_cpu": self.R_cpu,
            "sample_count": self.sample_count,
            "config": {
                "rank_per_block": self.rank_per_block,
                "block_size": self.block_size,
                "proj_dim": self.proj_dim,
                "normalize_proj_columns": self.normalize_proj_columns,
                "average_over_proj_dim": self.average_over_proj_dim,
            }
        }

    def load_state_dict(self, state_dict):
        self.validate_state_dict_shapes(state_dict)
        saved_config = state_dict.get("config", {})
        if isinstance(saved_config, dict):
            if "normalize_proj_columns" in saved_config:
                self.normalize_proj_columns = bool(saved_config["normalize_proj_columns"])
            if "average_over_proj_dim" in saved_config:
                self.average_over_proj_dim = bool(saved_config["average_over_proj_dim"])
        self.subspaces_cpu = state_dict["subspaces_cpu"]
        self.fisher_diag_cpu = state_dict["fisher_diag_cpu"]
        self.theta_star = state_dict["theta_star"]
        self.R_cpu = state_dict["R_cpu"]
        self.sample_count = state_dict["sample_count"]

# ==================================================================
# Part 2: TARP Helper Functions
# ==================================================================

def extract_between(text: Optional[str], left: Optional[str], right: Optional[str]) -> str:
    if text is None:
        return ""
    if right and right in text:
        text = text.split(right, 1)[0]
    if left and left in text:
        text = text.split(left, 1)[1]
    return text

def split_reason_sentences(text: str):
    think = extract_between(text, "<think>", "</think>")
    lines = [ln.strip() for ln in think.split("\n") if ln.strip()]
    if len(lines) >= 2:
        return lines
    parts = re.split(r"(?<=[\.\?\!])\s+", think.strip())
    return [p.strip() for p in parts if p.strip()] or ([think.strip()] if think.strip() else [])

def _tarp_norm_text(line: str) -> str:
    text = (line or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tarp_count_numbers(text: str) -> int:
    return len(re.findall(r"\d+(?:\.\d+)?", text))


def _tarp_has_binary_expr(text: str) -> bool:
    # 例如: 69 - 68, 12 * 5, 36 / 5
    return re.search(r"\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?", text) is not None


def _tarp_has_equation_chain(text: str) -> bool:
    # 例如:
    # 69 + 68 = 137
    # 12 * 5 - 36 = 60 - 36 = 24
    # (48 - 25) * 3 = 23 * 3 = 69
    if "=" not in text:
        return False
    n_nums = _tarp_count_numbers(text)
    has_op = re.search(r"[\+\-\*/]", text) is not None
    return n_nums >= 2 and has_op


def _tarp_is_plain_problem_restatement(low: str) -> bool:
    patterns = [
        r"\bwe need to use\b",
        r"\buse the numbers\b",
        r"\bexactly once\b",
        r"\bbasic arithmetic operations\b",
        r"\bdifferent combinations\b",
        r"\bto get\b",
    ]
    return any(re.search(pat, low) for pat in patterns)


def _tarp_is_search_control(low: str) -> bool:
    patterns = [
        r"\blet'?s try\b",
        r"\blets try\b",
        r"\blet'?s calculate\b",
        r"\btry another\b",
        r"\banother approach\b",
        r"\banother combination\b",
        r"\bthis is not\b",
        r"\bthis is also not\b",
        r"\bthis is incorrect\b",
        r"\btoo high\b",
        r"\btoo low\b",
        r"\bnot an integer\b",
        r"\bnone of these worked\b",
        r"\bnone of this worked\b",
        r"\btry this\b",
        r"\bcheck\b",
        r"\bbacktrack\b",
        r"\balternative\b",
        r"\bswitch approach\b",
    ]
    return any(re.search(pat, low) for pat in patterns)


def _tarp_is_math_step(text: str) -> bool:
    """
    更保守的 Countdown math-step 判定。
    保留真正的候选表达式 / 中间计算 / 等式链，
    避免把题目复述、失败反馈、控制句误抬成 math_step。
    """
    if not text:
        return False

    low = text.lower()
    n_nums = _tarp_count_numbers(text)
    has_op = re.search(r"[\+\-\*/]", text) is not None
    has_paren = ("(" in text) or (")" in text)

    # 明确排除题目复述 / 控制流 / 结果反馈
    if _tarp_is_plain_problem_restatement(low):
        return False
    if _tarp_is_search_control(low):
        return False

    # 1) 最可靠：等式链
    if _tarp_has_equation_chain(text):
        return True

    # 2) 短二元表达式
    if _tarp_has_binary_expr(text):
        return True

    # 3) 带括号的候选表达式
    if n_nums >= 2 and has_op and has_paren:
        return True

    # 4) 不带等号但明显像完整候选式
    # 例如: (93 - 56) + (24 - 11)
    if n_nums >= 3 and has_op:
        op_count = text.count("+") + text.count("-") + text.count("*") + text.count("/")
        natural_lang_cues = re.search(
            r"\b(we|need|numbers|exactly|using|operations|approach|calculate|works|incorrect|high|low)\b",
            low,
        )
        if op_count >= 2 and (has_paren or natural_lang_cues is None):
            return True

    return False


def _tarp_is_kk_summary(text: str, low: str) -> bool:
    patterns = [
        r"^in summary\b",
        r"^overall\b",
        r"^therefore,?\s+the identities\b",
        r"^thus,?\s+the identities\b",
        r"^so,?\s+the identities\b",
        r"^the identities are\b",
        r"^final answer\b",
        r"^\s*[-*]\s*[a-z][a-z]+(?:\s*[:\-]\s*|\s+is\s+(?:a\s+)?)?(?:knight|knave)\b",
        r"^\s*\(?\d+\)?[\.\:]?\s*[a-z][a-z]+(?:\s*[:\-]\s*|\s+is\s+(?:a\s+)?)?(?:knight|knave)\b",
    ]
    return any(re.search(pat, low) for pat in patterns)


def _tarp_is_kk_statement_focus(low: str) -> bool:
    patterns = [
        r"^\s*\d+\.\s*[a-z][a-z]+:\s*[\"']",
        r"^(?:first|next|now|finally|then),?\s+(?:let'?s|let us|we)\s+(?:consider|look at|check|examine)\b",
        r"^(?:let'?s|let us|we)\s+(?:consider|look at|check|examine)\b",
        r"^(?:first|next|now|finally|then),?\s+[a-z][a-z]+(?:'s)?\s+statement\b",
        r"^[a-z][a-z]+(?:'s)?\s+statement\b",
        r"^as\s+[a-z][a-z]+\s+(?:put it|said|claimed|remarked|expressed)\b",
        r"^according to\s+[a-z][a-z]+\b",
    ]
    return any(re.search(pat, low) for pat in patterns)


def _tarp_is_kk_consistency_check(low: str) -> bool:
    patterns = [
        r"\bcontradiction\b",
        r"\bcontradicts?\b",
        r"\bconsistent\b",
        r"\binconsistent\b",
        r"\bcannot both\b",
        r"\bcan(?:not|'t)\s+be\s+both\b",
        r"\bmust be\b",
        r"\bcannot be\b",
        r"\bcan(?:not|'t)\s+be\b",
        r"\bthis (?:means|implies|shows)\b",
        r"\bwhich (?:means|implies|shows)\b",
        r"\bthis would make\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
    ]
    return any(re.search(pat, low) for pat in patterns)


def _tarp_is_kk_logic_step(low: str) -> bool:
    has_logic_cue = re.search(
        r"\b(if\b.*\bthen|since|because|suppose|assume|otherwise|would mean|follows that|implies|therefore|thus|hence)\b",
        low,
    )
    has_role_cue = re.search(
        r"\b(knight|knave|lying|lie|lies|truth|true|false|statement)\b",
        low,
    )
    return has_logic_cue is not None and has_role_cue is not None


def _classify_kk_tarp_line_type(text: str, low: str) -> str:
    if _tarp_is_kk_summary(text, low):
        return "summary"
    if _tarp_is_kk_consistency_check(low):
        return "consistency_check"
    if _tarp_is_kk_statement_focus(low):
        return "statement_focus"
    if _tarp_is_kk_logic_step(low):
        return "logic_step"
    return "filler"


def default_tarp_line_type_weights(dataset_key: Optional[str] = None) -> Dict[str, float]:
    normalized_dataset_key = _normalize_tarp_dataset_key(dataset_key)
    if normalized_dataset_key == "KK":
        return dict(TARP_KK_LINE_TYPE_WEIGHTS)
    return dict(TARP_DEFAULT_LINE_TYPE_WEIGHTS)


def classify_tarp_line_type(line: str, dataset_key: Optional[str] = None) -> str:
    text = _tarp_norm_text(line)
    if not text:
        return "filler"

    low = text.lower()
    normalized_dataset_key = _normalize_tarp_dataset_key(dataset_key)

    if normalized_dataset_key == "KK":
        return _classify_kk_tarp_line_type(text, low)

    # 优先判 search_control，避免
    # "This is not 24. Let's try another combination"
    # 因为带数字被误判成 math_step
    if _tarp_is_search_control(low):
        return "search_control"

    if _tarp_is_math_step(text):
        return "math_step"

    if _tarp_is_plain_problem_restatement(low):
        return "filler"

    if re.search(r"\b(try|attempt|approach|combine|combination|search|retry|calculate)\b", low):
        return "search_control"

    return "filler"

def compute_tarp_line_weight_profile(
    lines: Sequence[str],
    sem_scores: Optional[torch.Tensor],
    *,
    dataset_key: Optional[str] = None,
    apply_sem_margin: bool = True,
    sem_tau: float = 0.2,
    sem_power: float = 1.0,
    use_line_type_weight: bool = True,
    use_tail_bias: bool = True,
    line_type_weights: Optional[Dict[str, float]] = None,
    tail_head_weight: float = 0.8,
    tail_tail_weight: float = 1.2,
    sem_fallback: float = 0.2,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Union[int, float, str]]]]:
    n_lines = len(lines)
    dev = device
    if dev is None:
        if sem_scores is not None and sem_scores.numel() > 0:
            dev = sem_scores.device
        else:
            dev = torch.device("cpu")

    if n_lines <= 0:
        return torch.empty(0, dtype=torch.float32, device=dev), []

    if line_type_weights is None:
        line_type_weights = default_tarp_line_type_weights(dataset_key)

    sem_fallback = max(0.0, min(1.0, float(sem_fallback)))
    sem_tau = max(0.0, min(1.0, float(sem_tau)))
    sem_power = max(0.0, float(sem_power))

    sem_tensor: Optional[torch.Tensor] = None
    if sem_scores is not None and sem_scores.numel() > 0:
        sem_tensor = sem_scores.detach().float().reshape(-1).to(device=dev)

    final_weights: List[float] = []
    profile: List[Dict[str, Union[int, float, str]]] = []

    for idx, line in enumerate(lines):
        sem_raw = sem_fallback
        if sem_tensor is not None and idx < sem_tensor.numel():
            sem_raw = float(sem_tensor[idx].item())
        sem_raw = max(0.0, min(1.0, sem_raw))

        sem_scaled = sem_raw
        if apply_sem_margin:
            sem_scaled = 0.0 if sem_scaled <= sem_tau else (sem_scaled - sem_tau) / (1.0 - sem_tau + 1e-9)
        sem_scaled = max(0.0, min(1.0, sem_scaled)) ** sem_power

        line_type = classify_tarp_line_type(line, dataset_key=dataset_key)
        type_weight = float(line_type_weights.get(line_type, 1.0)) if use_line_type_weight else 1.0

        # 连续位置权重，替代硬二分
        if use_tail_bias:
            if n_lines <= 1:
                pos_alpha = 1.0
            else:
                pos_alpha = idx / float(n_lines - 1)
            tail_weight = float(tail_head_weight) * (1.0 - pos_alpha) + float(tail_tail_weight) * pos_alpha
        else:
            tail_weight = 1.0

        final_weight = sem_scaled * type_weight * tail_weight
        final_weights.append(float(final_weight))
        profile.append(
            {
                "line_idx": int(idx),
                "line_type": str(line_type),
                "line_text": str(line),
                "sem_raw": float(sem_raw),
                "sem_scaled": float(sem_scaled),
                "line_type_weight": float(type_weight),
                "tail_weight": float(tail_weight),
                "final_weight": float(final_weight),
            }
        )

    weights = torch.tensor(final_weights, dtype=torch.float32, device=dev)
    return weights, profile

def build_tarp_token_weight_from_lines(
    *,
    anchor_ids: torch.Tensor,
    nll: torch.Tensor,
    anchor_spans: Sequence[Tuple[int, int, str, int]],
    line_weights: torch.Tensor,
    exclude_ids: Optional[torch.Tensor] = None,
    aux_spans: Optional[Sequence[Dict[str, Union[int, float, str, bool]]]] = None,
    valid_start_idx: int = 0,
) -> torch.Tensor:
    token_weight = torch.zeros_like(nll, dtype=torch.float32)
    tm1 = int(nll.size(1))

    def _accumulate_span(start_tok: int, end_tok: int, weight_value: float, allow_excluded: bool):
        s_logit = min(max(0, int(start_tok) - 1), tm1)
        e_logit = min(max(0, int(end_tok) - 1), tm1)
        if e_logit <= s_logit:
            return
        scalar_weight = float(weight_value)
        if scalar_weight <= 0.0:
            return

        length_slice = e_logit - s_logit
        sent_ids = anchor_ids[:, int(start_tok) : int(end_tok)]
        if allow_excluded or exclude_ids is None or exclude_ids.numel() == 0:
            valid_mask = torch.ones_like(sent_ids, dtype=torch.float32)
        else:
            valid_mask = (~isin_mask(sent_ids, exclude_ids)).float()

        if valid_mask.size(1) < length_slice:
            pad = torch.ones(
                (valid_mask.size(0), length_slice - valid_mask.size(1)),
                dtype=torch.float32,
                device=valid_mask.device,
            )
            valid_mask = torch.cat([valid_mask, pad], dim=1)

        token_weight[:, s_logit:e_logit] += scalar_weight * valid_mask[:, :length_slice]

    for (start_tok, end_tok, _, line_idx) in anchor_spans:
        if line_idx < 0 or line_idx >= line_weights.numel():
            continue
        _accumulate_span(start_tok, end_tok, float(line_weights[line_idx].item()), allow_excluded=False)

    for aux in (aux_spans or []):
        _accumulate_span(
            int(aux["start_tok"]),
            int(aux["end_tok"]),
            float(aux["weight"]),
            allow_excluded=bool(aux.get("allow_excluded", False)),
        )

    token_weight[:, : max(0, int(valid_start_idx))] = 0.0
    return token_weight

def locate_line_spans(tok, token_ids_list, lines):
    spans = []
    cursor = 0
    for ln in lines:
        ln_ids = tok(ln, add_special_tokens=False)["input_ids"]
        if not ln_ids: continue
        start = -1
        max_i = len(token_ids_list) - len(ln_ids)
        for i in range(max(cursor, 0), max_i + 1):
            if token_ids_list[i:i + len(ln_ids)] == ln_ids:
                start = i; break
        if start == -1:
            ln2 = re.sub(r"\s+", " ", ln).strip()
            ln_ids2 = tok(ln2, add_special_tokens=False)["input_ids"]
            if ln_ids2:
                max_i2 = len(token_ids_list) - len(ln_ids2)
                for i in range(max(cursor, 0), max_i2 + 1):
                    if token_ids_list[i:i + len(ln_ids2)] == ln_ids2:
                        start = i; ln_ids = ln_ids2; break
        if start == -1: continue
        end = start + len(ln_ids)
        spans.append((start, end, ln))
        cursor = end
    return spans

def isin_mask(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    try: return torch.isin(x, values)
    except: return (x.unsqueeze(-1) == values.view(1, 1, -1)).any(dim=-1)

def make_exclude_ids(tok, device):
    exclude_strings = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>", "<|", "|>", "\\", "\\(", "\\)", "\\[", "\\]"]
    ids = []
    for s in exclude_strings:
        try: ids.extend(tok.encode(s, add_special_tokens=False))
        except: pass
    ids = list(set(ids))
    return torch.tensor(ids, device=device, dtype=torch.long) if ids else None

def find_subseq(haystack, needle):
    if needle is None or len(needle) == 0: return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i + n] == needle: return i + n
    return None

def trim_end_by_pad(seq_1d: torch.Tensor, pad_id: int):
    if pad_id is None: return int(seq_1d.numel())
    if (seq_1d == pad_id).any():
        idx = (seq_1d == pad_id).nonzero(as_tuple=True)[0]
        if idx.numel() > 0: return int(idx[0].item())
    return int(seq_1d.numel())

@torch.no_grad()
def encode_sentence_embeds(actor_module, input_ids: torch.Tensor, spans, exclude_ids=None):
    device = input_ids.device
    attn = torch.ones_like(input_ids, device=device)
    # Ensure module is in eval mode or handle accordingly
    out = actor_module(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1].float()
    embs = []
    for (s, e, _) in spans:
        if e <= s: continue
        ids_seg = input_ids[:, s:e]
        mask = (~isin_mask(ids_seg, exclude_ids)).float() if (exclude_ids is not None and exclude_ids.numel() > 0) else torch.ones_like(ids_seg, dtype=torch.float32)
        vec = hs[:, s:e, :]
        denom = mask.sum().clamp_min(1e-6)
        pooled = (vec * mask.unsqueeze(-1)).sum(dim=1) / denom
        embs.append(pooled.squeeze(0))
    if not embs: return None
    embs = torch.stack(embs, dim=0)
    return torch.nn.functional.normalize(embs, p=2, dim=-1)

def semantic_consistency(anchor_embs: torch.Tensor, other_embs_list, agg="mean"):
    Sa = anchor_embs.size(0)
    sims = []
    for embs in other_embs_list:
        if embs is None or embs.numel() == 0: continue
        sim = anchor_embs @ embs.t()
        best = sim.max(dim=1).values.clamp(-1.0, 1.0)
        best01 = (best + 1.0) * 0.5
        sims.append(best01)
    if not sims: return torch.full((Sa,), 0.35, device=anchor_embs.device, dtype=torch.float32)
    stack = torch.stack(sims, dim=0)
    if agg == "min": sem = stack.min(dim=0).values
    elif agg == "max": sem = stack.max(dim=0).values
    else: sem = stack.mean(dim=0)
    return sem.clamp(0.0, 1.0)
