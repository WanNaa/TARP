# Anonymous Submission Code

This repository contains the anonymized code package for a paper submission on multi-domain reasoning with reinforcement learning.

## Repository Layout

- `verl/`: core training and rollout framework
- `data/`: only the three default main-experiment datasets used by the pipeline
- `examples/grpo_trainer/math_and_code.sh`: the training launcher used by the pipeline
- `scripts/run_pipeline.sh`: the main experiment entrypoint
- `scripts/anchor_fisher_paths.sh` and `scripts/runtime_layout.sh`: helper scripts required by the pipeline
- `merge_model.py`: merge utility invoked by the pipeline

## Quick Start

The dependency set is pinned to the validated `data_rl` environment (Python 3.10).

Recommended setup:

```bash
conda env create -f environment.yml
conda activate data_rl
pip install -e .
```

If you prefer not to use `environment.yml`, install directly from:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

The main experiment helpers live under `scripts/`.

Example:

```bash
bash scripts/run_pipeline.sh
```

Many scripts use placeholder defaults such as `/path/to/models` and `/path/to/outputs`.
Please replace them with your local paths or override them via environment variables before running.

The default pipeline uses:

- `data/countdown/train_ins.parquet` and `data/countdown/test_ins.parquet`
- `data/kk/train_ins.parquet` and `data/kk/test_ins.parquet`
- `data/code-r1-12k/train_ins.parquet` and `data/code-r1-12k/test_ins.parquet`

## Notes

- This package is an anonymized submission copy.
- This package was further reduced to a minimal main-experiment bundle.
- Non-essential analysis scripts, auxiliary datasets, preprocessing sources, docs, tests, and runtime artifacts were removed.
- Large runtime artifacts such as logs, checkpoints, wandb outputs, and generated result folders are intentionally excluded from this package.
