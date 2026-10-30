from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple


def has_fsdp_sharded_checkpoint(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    try:
        find_world_size(local_dir)
    except FileNotFoundError:
        return False
    return True


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    ensure_runtime_imports()
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    skip_ext = {".pt", ".bin", ".safetensors"}
    os.makedirs(dst_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if any(filename.endswith(ext) for ext in skip_ext):
            continue
        if "model_world_size" in filename:
            continue

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    print(f"Copied tokenizer/config files from {src_dir} to {dst_dir}")


def find_world_size(local_dir: str) -> int:
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(
        f"No checkpoint file matching model_world_size_*_rank_0.pt found under {local_dir}"
    )


def load_shards(local_dir: str, world_size: int) -> List[Dict[str, torch.Tensor]]:
    ensure_runtime_imports()
    model_state_dicts: List[Dict[str, torch.Tensor]] = [{} for _ in range(world_size)]

    def load_shard(rank: int) -> None:
        path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dicts[rank] = torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = []
        for rank in range(world_size):
            futures.append(executor.submit(load_shard, rank))
        for future in futures:
            future.result()

    return model_state_dicts


def merge_state_dicts(
    model_state_dicts: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    ensure_runtime_imports()
    all_keys = set(model_state_dicts[0].keys())
    for state_dict in model_state_dicts[1:]:
        if set(state_dict.keys()) != all_keys:
            raise ValueError("Checkpoint shards do not share the same state_dict keys")

    merged_state_dict: Dict[str, torch.Tensor] = {}
    param_placements: Dict[str, Tuple[Placement, ...]] = {}

    for key in sorted(all_keys):
        merged_state_dict[key] = []
        for state_dict in model_state_dicts:
            tensor = state_dict[key]
            if isinstance(tensor, DTensor):
                merged_state_dict[key].append(tensor._local_tensor.to(torch.bfloat16))
                placements = tuple(tensor.placements)
                if key not in param_placements:
                    param_placements[key] = placements
                elif param_placements[key] != placements:
                    raise ValueError(f"Inconsistent placements found for key {key}")
            elif isinstance(tensor, torch.Tensor):
                merged_state_dict[key] = tensor.to(torch.bfloat16)
                break
            else:
                merged_state_dict[key] = tensor
                break

    for key, value in list(merged_state_dict.items()):
        if not isinstance(value, list):
            continue
        placements = param_placements[key]
        if len(placements) != 1:
            raise NotImplementedError(f"Only 1D placement is supported, got {placements}")
        merged_state_dict[key] = merge_by_placement(value, placements[0])

    return merged_state_dict


def resolve_auto_model(config: AutoConfig):
    ensure_runtime_imports()
    architecture = config.architectures[0]
    if "ForTokenClassification" in architecture:
        return AutoModelForTokenClassification
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in architecture:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture: {architecture}")


def merge_checkpoint(local_dir: str, merge_path: str, overwrite: bool = False) -> None:
    ensure_runtime_imports()
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {local_dir}")

    if os.path.exists(merge_path):
        if overwrite:
            shutil.rmtree(merge_path)
        else:
            raise FileExistsError(
                f"Merge output already exists: {merge_path}. Use --overwrite to replace it."
            )

    os.makedirs(merge_path, exist_ok=True)

    world_size = find_world_size(local_dir)
    print(f"Merging checkpoint from {local_dir}")
    print(f"Detected world_size={world_size}")

    model_state_dicts = load_shards(local_dir, world_size)
    first_state_dict = model_state_dicts[0]
    pivot_key = sorted(first_state_dict.keys())[0]
    pivot_tensor = first_state_dict[pivot_key]
    if not isinstance(pivot_tensor, DTensor):
        raise TypeError(f"Expected DTensor weights, got {type(pivot_tensor)} for key {pivot_key}")

    device_mesh = pivot_tensor.device_mesh
    print(f"Device mesh: {device_mesh.mesh}, mesh_dim_names={device_mesh.mesh_dim_names}")

    merged_state_dict = merge_state_dicts(model_state_dicts)

    config = AutoConfig.from_pretrained(local_dir)
    auto_model = resolve_auto_model(config)

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.save_pretrained(merge_path, state_dict=merged_state_dict)

    copy_non_weight_files(local_dir, merge_path)
    print(f"Merged model weights saved to: {merge_path}")


def ensure_runtime_imports() -> None:
    global torch
    global DTensor
    global Placement
    global AutoConfig
    global AutoModelForCausalLM
    global AutoModelForTokenClassification
    global AutoModelForVision2Seq

    if "torch" in globals():
        return

    import torch  # type: ignore[no-redef]
    from torch.distributed._tensor import DTensor, Placement  # type: ignore[no-redef]
    from transformers import (  # type: ignore[no-redef]
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForTokenClassification,
        AutoModelForVision2Seq,
    )
