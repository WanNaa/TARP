from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from verl.utils.checkpoint.merge_utils import merge_checkpoint


def parse_steps(raw_steps: Sequence[str]) -> List[int]:
    steps: List[int] = []
    for item in raw_steps:
        for chunk in item.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            steps.append(int(chunk))
    if not steps:
        raise ValueError("No valid steps were provided")
    return steps


def build_jobs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.local_dir and args.merge_path:
        return [(args.local_dir, args.merge_path)]

    if args.input_pattern and args.output_pattern and args.steps:
        jobs = []
        for step in parse_steps(args.steps):
            jobs.append(
                (
                    args.input_pattern.format(step=step),
                    args.output_pattern.format(step=step),
                )
            )
        return jobs

    raise ValueError(
        "Provide either --local-dir/--merge-path for a single merge, "
        "or --input-pattern/--output-pattern/--steps for batch merging."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FSDP actor checkpoints into a standard Hugging Face model directory."
    )
    parser.add_argument("--local-dir", help="Checkpoint actor directory, e.g. .../global_step_200/actor")
    parser.add_argument("--merge-path", help="Output directory for the merged Hugging Face model")
    parser.add_argument(
        "--input-pattern",
        help="Input checkpoint pattern containing {step}, e.g. /path/global_step_{step}/actor",
    )
    parser.add_argument(
        "--output-pattern",
        help="Merged output pattern containing {step}, e.g. /path/3B-CD{step}",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Checkpoint steps for batch merging, supports space or comma separation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the merged output directory if it already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for checkpoint_dir, output_dir in build_jobs(arguments):
        print(f"\n================ Processing {checkpoint_dir} ================\n")
        merge_checkpoint(checkpoint_dir, output_dir, overwrite=arguments.overwrite)
