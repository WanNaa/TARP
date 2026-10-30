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

from omegaconf import DictConfig, ListConfig
import os
from typing import List, Optional, Union
import copy
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.chat_prompt import render_prompt_for_generation
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def _rendered_prompt_token_count(tokenizer: PreTrainedTokenizer, prompt_entry) -> int:
    rendered_prompt, _ = render_prompt_for_generation(tokenizer, prompt_entry)
    return len(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"])


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.parquet_files  = copy.deepcopy(data_files)
        self.original_parquet_files  = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get('use_shm', False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_overlong_prompts", config.get("filter_prompts", True))
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        if self.filter_prompts:
            # filter out too long prompts
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            print("prompt_key", prompt_key)
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: _rendered_prompt_token_count(tokenizer, doc[prompt_key]) <= self.max_prompt_length,
                    axis=1,
                )
            ]
            print(f'filter dataset len: {len(self.dataframe)}')
            # Avoid dumping the full dataframe (can be hundreds of thousands of rows).
            print("dataframe head", self.dataframe.head(4))
        else:
            print("Skipping overlong-prompt filtering (filter_overlong_prompts=False).")

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    # def __getitem__(self, item):
    #     """
    #     Note that we also return the raw_input_ids so that it can be combined with other chat template
    #     """
    #     row_dict = self.dataframe.iloc[item].to_dict()

    #     chat = row_dict.pop(self.prompt_key)

    #     prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    #     input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
    #                                                                      tokenizer=self.tokenizer,
    #                                                                      max_length=self.max_prompt_length,
    #                                                                      pad_token_id=self.tokenizer.pad_token_id,
    #                                                                      left_pad=True,
    #                                                                      truncation=self.truncation)

    #     position_ids = compute_position_id_with_mask(attention_mask)

    #     row_dict['input_ids'] = input_ids[0]
    #     row_dict['attention_mask'] = attention_mask[0]
    #     row_dict['position_ids'] = position_ids[0]

    #     # encode prompts without chat template
    #     if self.return_raw_chat:
    #         row_dict['raw_prompt'] = chat.tolist()

    #     # add index for each prompt
    #     index = row_dict.get("extra_info", {}).get("index", 0)
    #     row_dict["index"] = index

    #     return row_dict
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        
        chat = row_dict.pop(self.prompt_key)
        prompt_with_chat_template, chat_messages = render_prompt_for_generation(self.tokenizer, chat)

        if self.image_key in row_dict:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = copy.deepcopy(chat_messages)
        
        # add index for each prompt
        # index = row_dict.get("extra_info", {}).get("index", 0)
        # row_dict["index"] = index
        return row_dict

    def __getstate__(self):
        # if not self.serialize_dataset:
        #     state = self.__dict__.copy()

        #     if 'dataframe' in state:
        #         del state['dataframe']
        #     return state
        return self.__dict__.copy()
