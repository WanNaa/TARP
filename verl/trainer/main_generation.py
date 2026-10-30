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
Generate responses given a dataset of prompts
"""

import os
from typing import Any

import hydra
import numpy as np
import ray
import torch

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.chat_prompt import normalize_prompt_entry, render_prompt_for_generation
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _tokenize_prompt_entry(tokenizer, prompt_entry: Any, prompt_length: int, truncation: str):
    prompt_entry = normalize_prompt_entry(prompt_entry)
    rendered_prompt, _ = render_prompt_for_generation(tokenizer, prompt_entry)
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=rendered_prompt,
        tokenizer=tokenizer,
        max_length=prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )

    position_ids = compute_position_id_with_mask(attention_mask)
    return input_ids[0], attention_mask[0], position_ids[0]


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        init_kwargs = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            "num_cpus": config.ray_init.num_cpus,
        }
        temp_dir = os.environ.get("RAY_TMPDIR") or config.ray_init.get("temp_dir", None)
        if temp_dir:
            init_kwargs["_temp_dir"] = temp_dir
        ray.init(**init_kwargs)

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [normalize_prompt_entry(chat) for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        tokenized_prompts = [
            _tokenize_prompt_entry(
                tokenizer=tokenizer,
                prompt_entry=prompt_entry,
                prompt_length=config.rollout.prompt_length,
                truncation=config.data.get("truncation", "error"),
            )
            for prompt_entry in batch_chat_lst
        ]
        input_ids = torch.stack([item[0] for item in tokenized_prompts], dim=0)
        attention_mask = torch.stack([item[1] for item in tokenized_prompts], dim=0)
        position_ids = torch.stack([item[2] for item in tokenized_prompts], dim=0)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    # write to a new file according to suffix
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    output_path = config.data.output_path
    suffix = os.path.splitext(output_path)[1].lower()

    if suffix in [".parquet", ".pq"]:
        dataset.to_parquet(output_path)
    elif suffix == ".jsonl":
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        dataset.to_json(output_path, orient="records", lines=False, force_ascii=False)
    elif suffix in [".txt", ".tsv"]:
        dataset.to_csv(output_path, sep="\t", index=False)
    elif suffix == ".csv":
        dataset.to_csv(output_path, index=False)
    else:
        # default to parquet
        dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()
