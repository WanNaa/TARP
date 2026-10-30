# TARP: Trace-Anchored Regularization Prior for Retaining Stepwise Reasoning

Official implementation of **TARP**, a trace-centric regularizer for sequential reinforcement learning. TARP selects reliable spans from successful reasoning traces, compresses their local gradient sensitivity into a compact quadratic prior, and uses that prior to regularize later RL stages.

The released code supports the baseline and TARP pipelines reported for:

- Qwen2.5-3B-Instruct on CountDown -> KK-Logic -> Code-R1-12k;
- Qwen2.5-7B-Instruct on Code-R1-12k -> KK-Logic -> CountDown; and
- Llama-3.2-3B-Instruct on the short CountDown -> KK-Logic -> Code-R1-12k chain.

## Release scope

This repository contains:

- the TARP implementation in `verl/workers/actor/tarp_subspace.py`, `tarp_utils.py`, and the modified actor/trainer code;
- the staged baseline and TARP launcher, `scripts/run_pipeline.sh`;
- the GRPO launcher used by the pipeline, `examples/grpo_trainer/math_and_code.sh`;
- the exact processed train/test Parquet files used by the released main pipeline; and
- pinned Python dependencies in `environment.yml` and `requirements.txt`.

The release intentionally excludes pretrained base-model weights, trained checkpoints, generated rollouts, runtime logs, Weights & Biases files, and evaluation-only human/LLM-judge records. These artifacts are large, contain service-specific outputs, or are not required to run the released training pipeline. The paper describes the auxiliary evaluation protocols and rubrics. The public bundle is versioned by Git; please record the commit hash when reporting a reproduction.

## Repository layout

```text
data/                              Processed CountDown, KK-Logic, and Code-R1 data
examples/grpo_trainer/             GRPO training launcher used by the pipeline
scripts/run_pipeline.sh            Baseline/TARP sequential-training entrypoint
scripts/anchor_fisher_paths.sh     TARP prior naming and path helpers
scripts/runtime_layout.sh          Runtime, Ray, log, and Hydra directory layout
verl/workers/actor/tarp_*.py       TARP prior construction and regularization
verl/trainer/config/               Hydra training configuration
merge_model.py                     FSDP checkpoint merge utility
environment.yml                    Conda environment specification
requirements.txt                   Pinned Python packages
```

## Environment

The released environment uses Python 3.10 and pins the principal training packages, including PyTorch 2.6.0, Transformers 4.57.3, Ray 2.48.0, vLLM 0.8.3, and xFormers 0.0.29.post2. The reported TARP runs used Linux and four NVIDIA A100 80 GB GPUs with FSDP and mixed precision. A CUDA driver/toolkit compatible with the pinned PyTorch, vLLM, FlashAttention, and xFormers builds is required.

```bash
git clone https://github.com/WanNaa/TARP.git
cd TARP

conda env create -f environment.yml
conda activate data_rl
pip install -e .
```

Alternatively:

```bash
conda create -n data_rl python=3.10
conda activate data_rl
pip install -r requirements.txt
pip install -e .
```

## Data

The six processed files used by the main pipeline are already included:

| Task | Training file | Test file |
|---|---|---|
| CountDown | `data/countdown/train_ins.parquet` | `data/countdown/test_ins.parquet` |
| KK-Logic | `data/kk/train_ins.parquet` | `data/kk/test_ins.parquet` |
| Code-R1-12k | `data/code-r1-12k/train_ins.parquet` | `data/code-r1-12k/test_ins.parquet` |

Data provenance follows [TinyZero](https://github.com/Jiayi-Pan/TinyZero), [On Memorization of Large Language Models in Logical Reasoning](https://arxiv.org/abs/2410.23123), and [Code-R1](https://github.com/ganler/code-r1). The processed Code-R1 training split is a filtered mixture dominated by `likaixin/TACO-verified`, with a smaller contribution from `LeetCodeDataset`. These processed files are provided for research reproduction; the original sources retain their own licenses and terms of use.

SHA-256 checksums:

```text
f4df7302a6bffa7cbc3b671bc805a0c47a2fdb1e78d818cea5881f895fe912d9  data/code-r1-12k/train_ins.parquet
9cddc7a8dc054fcbc4a68367fd753de59acd9d1d2a575243c291707d3ea908f9  data/code-r1-12k/test_ins.parquet
18788811da8939e4dae1e3120b8cd0b3b9ceda118bfb97ed1f07b97b4f97dd27  data/countdown/train_ins.parquet
a82d2a11dcf09055015ebfdd10a6ee86e76912ef5dfa5e32d84febbcdc27406f  data/countdown/test_ins.parquet
7645828ded881a26d4e12de1f661e1bc6830360a1da8fc057969fceb1a83ed9a  data/kk/train_ins.parquet
12e30dc63fc9d2e3d37cad96302b3f69c5805c39758ad5b6cb31bec34021426f  data/kk/test_ins.parquet
```

## Base models

Download the required model from its official provider and set `INITIAL_MODEL` to the local Hugging Face-format directory:

- `Qwen/Qwen2.5-3B-Instruct`;
- `Qwen/Qwen2.5-7B-Instruct`; or
- `meta-llama/Llama-3.2-3B-Instruct` (subject to the provider's access terms).

The repository does not redistribute model weights. The base models retain their original licenses and terms of use.

## Pipeline presets

```bash
bash scripts/run_pipeline.sh --help
```

The entrypoint provides six presets:

| Preset | Method | Default stage order | Default steps |
|---|---|---|---|
| `qwen_3b_baseline` | GRPO | CD -> KK -> R1 | 200 / 300 / 300 |
| `qwen_3b_tarp` | GRPO + TARP | CD -> KK -> R1 | 200 / 300 / 300 |
| `qwen_7b_baseline` | GRPO | R1 -> KK -> CD | 300 / 300 / 200 |
| `qwen_7b_tarp` | GRPO + TARP | R1 -> KK -> CD | 300 / 300 / 200 |
| `llama_3b_baseline` | GRPO | CD -> KK -> R1 | configurable; use 100 / 100 / 100 below |
| `llama_3b_tarp` | GRPO + TARP | CD -> KK -> R1 | configurable; use 100 / 100 / 100 below |

CD, KK, and R1 denote CountDown, KK-Logic, and Code-R1-12k, respectively. Unless overridden, the launcher uses train/mini-batch sizes 256/64, four rollouts per prompt, maximum prompt/response lengths 512/1024, and KL coefficient 0.001. The Qwen2.5-3B presets use learning rate `1e-6`; the Qwen2.5-7B and Llama-3.2-3B presets use `5e-6`. TARP uses retained rank 32 and projection dimension 48 by default.

## Reproducing the main pipelines

Set absolute paths before every run. Use distinct output roots for baseline and TARP so that checkpoints and priors cannot be reused across methods accidentally. Keep `TARP_RUNTIME_ROOT` reasonably short because Ray uses Unix-domain socket paths.

The following common setup assumes four visible GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONDA_ENV_NAME=data_rl
export LOGGER="['console']"
export START_FRESH=1
```

For each command below, replace `/absolute/path/...` with local paths. `MODEL_ROOT` stores distributed checkpoints, `MERGE_BASE_ROOT` stores merged Hugging Face models, `TARP_ANCHOR_ROOT` stores TARP priors, and `TARP_RUNTIME_ROOT` stores logs, Hydra configurations, and Ray temporary files.

### Qwen2.5-3B-Instruct

```bash
export INITIAL_MODEL=/absolute/path/to/Qwen2.5-3B-Instruct

# Baseline
export MODEL_ROOT=/absolute/path/to/runs/qwen3b_baseline/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/qwen3b_baseline/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/qwen3b_baseline/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/qwen3b_baseline
bash scripts/run_pipeline.sh qwen_3b_baseline

# TARP: use fresh, method-specific output directories
export MODEL_ROOT=/absolute/path/to/runs/qwen3b_tarp/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/qwen3b_tarp/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/qwen3b_tarp/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/qwen3b_tarp
bash scripts/run_pipeline.sh qwen_3b_tarp
```

The TARP preset supplies the released lexical-consensus export settings: 32 valid samples, four trajectories, minimum correct ratio 0.25, and maximum Fisher generation/sequence lengths 512/512. The continual-stage regularization weight defaults to 0.01.

### Qwen2.5-7B-Instruct

```bash
export INITIAL_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export R1_PPO_MICRO_BATCH_SIZE=8
export KK_PPO_MICRO_BATCH_SIZE=2
export CD_PPO_MICRO_BATCH_SIZE=2

# Baseline
export MODEL_ROOT=/absolute/path/to/runs/qwen7b_baseline/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/qwen7b_baseline/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/qwen7b_baseline/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/qwen7b_baseline
bash scripts/run_pipeline.sh qwen_7b_baseline

# TARP
export MODEL_ROOT=/absolute/path/to/runs/qwen7b_tarp/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/qwen7b_tarp/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/qwen7b_tarp/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/qwen7b_tarp
export KK_TARP_LAMBDA=0.05
export CD_TARP_LAMBDA=0.05
bash scripts/run_pipeline.sh qwen_7b_tarp \
  actor_rollout_ref.actor.tarp_target_valid_samples=128 \
  actor_rollout_ref.actor.tarp_n_traj=4 \
  actor_rollout_ref.actor.tarp_sem_metric=hybrid \
  actor_rollout_ref.actor.tarp_min_correct_ratio=0.5 \
  actor_rollout_ref.actor.tarp_max_new_tokens=1024 \
  actor_rollout_ref.actor.tarp_max_fisher_seq_len=1024
```

### Llama-3.2-3B-Instruct short chain

```bash
export INITIAL_MODEL=/absolute/path/to/Llama-3.2-3B-Instruct
export CD_STEPS=100
export KK_STEPS=100
export R1_STEPS=100
export PPO_MICRO_BATCH_SIZE=16

# Baseline
export MODEL_ROOT=/absolute/path/to/runs/llama3b_baseline/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/llama3b_baseline/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/llama3b_baseline/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/llama3b_baseline
bash scripts/run_pipeline.sh llama_3b_baseline

# TARP
export MODEL_ROOT=/absolute/path/to/runs/llama3b_tarp/checkpoints
export MERGE_BASE_ROOT=/absolute/path/to/runs/llama3b_tarp/merged
export TARP_ANCHOR_ROOT=/absolute/path/to/runs/llama3b_tarp/priors
export TARP_RUNTIME_ROOT=/absolute/path/to/runtime/llama3b_tarp
bash scripts/run_pipeline.sh llama_3b_tarp
```

## Configuration and outputs

Any additional argument after the preset is forwarded to Hydra. Environment variables provide the most commonly used stage-level overrides:

- `CD_STEPS`, `KK_STEPS`, and `R1_STEPS`;
- `CD_LR`, `KK_LR`, and `R1_LR`;
- `CD_PPO_MICRO_BATCH_SIZE`, `KK_PPO_MICRO_BATCH_SIZE`, and `R1_PPO_MICRO_BATCH_SIZE`;
- `CD_TARP_LAMBDA`, `KK_TARP_LAMBDA`, and `R1_TARP_LAMBDA`;
- `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `ROLLOUT_N`, and sequence-length settings; and
- `LOGGER="['wandb']"` together with the normal W&B environment variables when online tracking is desired.

At launch, the script prints the resolved stage order, model path, GPU count, output locations, TARP strategy, and per-stage checkpoint target. Full console logs and Hydra-resolved configurations are written below `TARP_RUNTIME_ROOT`; distributed checkpoints are written below `MODEL_ROOT`; and merged Hugging Face models are written below `MERGE_BASE_ROOT`.

`START_FRESH=1` disables automatic resume and checkpoint reuse. Omit it only when intentionally resuming an interrupted run. The launcher refuses to silently mix fresh online TARP priors with an existing prior directory.

## Reproducibility notes and limitations

- GRPO rollout sampling and distributed GPU kernels can remain nondeterministic across hardware and software stacks even with fixed configuration values. Report means and standard deviations over independent training runs when reproducing multi-seed results.
- The release supplies the exact processed Parquet inputs used by the public pipeline. It does not include raw-data preprocessing sources or alternative auxiliary datasets.
- The repository reproduces the staged training paths and built-in reward validation. Paper-only plotting, route-drift judge calls, and the blinded human audit are documented in the paper but are not automated by this release.
- TARP currently assumes explicit textual reasoning traces and uses newline-oriented segmentation with a sentence-level fallback. It is not designed as a semantic classifier of whether an individual trace change is beneficial or harmful.

## License and attribution

This repository is released under the Apache License 2.0; see `LICENSE`. It builds on the [verl](https://github.com/volcengine/verl) reinforcement-learning framework. Third-party models and datasets are not relicensed by this repository and remain subject to their original licenses and terms of use.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{li2026tarp,
  title     = {TARP: Trace-Anchored Regularization Prior for Retaining Stepwise Reasoning},
  author    = {Li, Wei and Feng, Tao and Yuan, Hangjie and Lu, Aojun and Sun, Yanan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
