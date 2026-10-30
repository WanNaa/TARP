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

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.chat_prompt import append_or_merge_assistant_message, render_chat_messages, render_prompt_for_generation
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class DPODataset(Dataset):
    """In-memory offline DPO dataset with prompt/chosen/rejected triples."""

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        chosen_key = config.get("chosen_key", "chosen")
        rejected_key = config.get("rejected_key", "rejected")
        max_length = config.get("max_length", 2048)
        truncation = config.get("truncation", "left")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.max_length = max_length

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframes.append(pd.read_parquet(parquet_file))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _pad_or_truncate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int):
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            effective_prompt_length = prompt_length
            effective_sequence_length = sequence_length
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                dropped_tokens = sequence_length - self.max_length
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                effective_prompt_length = max(0, prompt_length - dropped_tokens)
                effective_sequence_length = self.max_length
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                effective_prompt_length = min(prompt_length, self.max_length)
                effective_sequence_length = self.max_length
            else:
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
        else:
            effective_prompt_length = prompt_length
            effective_sequence_length = sequence_length

        return input_ids, attention_mask, effective_prompt_length, effective_sequence_length

    def _tokenize_example(self, prompt_entry, response_text):
        rendered_prompt, prompt_messages = render_prompt_for_generation(self.tokenizer, prompt_entry)
        full_messages = append_or_merge_assistant_message(prompt_messages, response_text)
        rendered_full = render_chat_messages(self.tokenizer, full_messages, add_generation_prompt=False)

        prompt_ids = self.tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_output = self.tokenizer(rendered_full, return_tensors="pt", add_special_tokens=False)
        input_ids = full_output["input_ids"][0]
        attention_mask = full_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        sequence_length = input_ids.shape[0]

        input_ids, attention_mask, prompt_length, sequence_length = self._pad_or_truncate(input_ids, attention_mask, prompt_length)
        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(sequence_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, item):
        prompt = self.prompts[item]
        chosen_response = self.chosen_responses[item]
        rejected_response = self.rejected_responses[item]

        chosen = self._tokenize_example(prompt, chosen_response)
        rejected = self._tokenize_example(prompt, rejected_response)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_position_ids": chosen["position_ids"],
            "chosen_loss_mask": chosen["loss_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_position_ids": rejected["position_ids"],
            "rejected_loss_mask": rejected["loss_mask"],
        }
