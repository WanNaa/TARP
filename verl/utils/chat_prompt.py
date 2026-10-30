from __future__ import annotations

import re
from typing import Any

import numpy as np

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"


def normalize_prompt_entry(prompt_entry: Any) -> Any:
    if isinstance(prompt_entry, np.ndarray):
        return prompt_entry.tolist()
    if isinstance(prompt_entry, tuple):
        return list(prompt_entry)
    return prompt_entry


def _is_qwen_serialized_chat(text: str) -> bool:
    return QWEN_IM_START in text and "\n" in text


def parse_qwen_serialized_chat(prompt: str) -> list[dict[str, str]]:
    starts = [match.start() for match in re.finditer(re.escape(QWEN_IM_START), prompt)]
    if not starts:
        raise ValueError("Expected a Qwen-style serialized chat prompt.")

    starts.append(len(prompt))
    messages: list[dict[str, str]] = []

    for idx in range(len(starts) - 1):
        chunk = prompt[starts[idx] + len(QWEN_IM_START) : starts[idx + 1]]
        role, separator, content = chunk.partition("\n")
        if not separator:
            continue

        role = role.strip()
        if not role:
            continue

        if content.endswith(QWEN_IM_END + "\n"):
            content = content[: -len(QWEN_IM_END + "\n")]
        elif content.endswith(QWEN_IM_END):
            content = content[: -len(QWEN_IM_END)]

        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("Failed to parse any chat messages from serialized prompt.")

    return messages


def prompt_entry_to_messages(prompt_entry: Any) -> list[dict[str, str]]:
    prompt_entry = normalize_prompt_entry(prompt_entry)

    if isinstance(prompt_entry, str):
        if _is_qwen_serialized_chat(prompt_entry):
            return parse_qwen_serialized_chat(prompt_entry)
        return [{"role": "user", "content": prompt_entry}]

    if isinstance(prompt_entry, list):
        if (
            len(prompt_entry) == 1
            and isinstance(prompt_entry[0], dict)
            and isinstance(prompt_entry[0].get("content"), str)
            and _is_qwen_serialized_chat(prompt_entry[0]["content"])
        ):
            return parse_qwen_serialized_chat(prompt_entry[0]["content"])

        return [dict(message) for message in prompt_entry]

    raise TypeError(f"Unsupported prompt entry type: {type(prompt_entry)!r}")


def render_chat_messages(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    )


def render_prompt_for_generation(tokenizer, prompt_entry: Any) -> tuple[str, list[dict[str, str]]]:
    messages = prompt_entry_to_messages(prompt_entry)
    continue_final_message = bool(messages and messages[-1].get("role") == "assistant")
    rendered_prompt = render_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
    )
    return rendered_prompt, messages


def append_or_merge_assistant_message(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    merged_messages = [dict(message) for message in messages]
    if merged_messages and merged_messages[-1].get("role") == "assistant":
        merged_messages[-1]["content"] = merged_messages[-1].get("content", "") + content
    else:
        merged_messages.append({"role": "assistant", "content": content})
    return merged_messages
