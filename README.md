# Sharpness-Aware Pretraining

This is the codebase used for our paper:

> **_Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting_**
> Ishaan Watts, Catherine Li, Sachin Goyal, Jacob Mitchell Springer, Aditi Raghunathan &nbsp;·&nbsp; ICML 2026 &nbsp;·&nbsp; [arXiv:2605.02105](https://arxiv.org/abs/2605.02105)

## Abstract

> Pretraining optimizers are tuned to produce the strongest possible base model, on the assumption
> that a stronger starting point yields a stronger model after subsequent changes like post-training
> and quantization. This overlooks the geometry of the base model which controls how much of the base
> model's capabilities survive subsequent parameter updates. We study three pretraining optimization
> approaches that bias optimization toward flatter minima: Sharpness-Aware Minimization (SAM), large
> learning rates, and shortened learning rate annealing periods. Across model sizes ranging from 20M
> to 150M parameters, we find that these interventions consistently improve downstream performance
> after post-training on five common datasets with up to 80% less forgetting. These principles hold at
> scale: a short SAM mid-training phase applied to an existing OLMo-2-1B checkpoint reduces forgetting
> by 31% after MetaMath post-training and by 40% after 4-bit quantization.

It builds on [OLMo](https://github.com/allenai/OLMo): the `olmo/` core trainer is extended
(e.g. SAM optimization, EWC), and `launch/` defines the experiment artifacts
(pretrain, midtrain, anneal, SFT, perturb, HF-convert, eval) as composable artifacts
submitted via the [`experiments`](https://github.com/jakespringer/experiments) launcher.

## Installation

We use two conda environments: `forgetting` for training/launching and `olmes` for downstream evaluation.

First install [PyTorch](https://pytorch.org) for your platform, then:

```bash
# 1. Main training / launching environment
conda create -n forgetting python=3.10 -y
conda activate forgetting

git clone https://github.com/WattsIshaan/sharpness-aware-pretraining.git
cd sharpness-aware-pretraining
pip install -e .[all]            # core + dev + train dependencies

# 2. Experiment launcher (Slurm + GCS orchestration used by launch/)
pip install -e .[launch]
# (equivalently: pip install git+https://github.com/jakespringer/experiments.git)
```

Downstream task evaluation runs through [OLMES](https://github.com/allenai/olmes) in its own
environment (the eval jobs `conda activate olmes` automatically):

```bash
conda create -n olmes python=3.10 -y
conda activate olmes
pip install git+https://github.com/allenai/olmes.git
```

Data and checkpoints are read from / written to Google Cloud Storage, so install and authenticate the
[gcloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud auth login`). Paths and bucket names are
configured in `launch/globals.py`.

## Running experiments

Each `launch/run_*.py` script builds a set of artifacts and submits them. Pass `launch` to submit jobs
or `printlines` for a dry run that prints the planned commands.

```bash
conda activate forgetting

# Submit pipelines
python launch/run_adamw.py launch        # AdamW small-scale pretraining + sft + eval
python launch/run_sam.py launch          # SAM small-scale pretraining + sft + eval
python launch/run_midtrain.py launch     # 1B midtraining + sft + eval

# Dry run (print the planned jobs without submitting)
python launch/run_adamw.py drylaunch
python launch/run_adamw.py printlines          # one line per job (individual bash scripts)

# Run a batch of artifacts locally (no Slurm)
python -m experiments.scripts.batch_local
```

### Launching an individual stage

Each pipeline is split into named **stages** via the `executor.stage('<name>', artifacts)` calls at the
bottom of each `launch/run_*.py`. Pass one or more stage names to `launch` (or `drylaunch`) to run only
those; omit them to run everything.

```bash
# Run only the pretraining stage
python launch/run_adamw.py launch pretrain

# Run a few specific stages
python launch/run_adamw.py launch sft sft_eval

# Everything except a stage; preview first; force a re-run of finished artifacts
python launch/run_adamw.py launch --exclude ewc_sft ewc_sft_eval
python launch/run_adamw.py drylaunch sft
python launch/run_adamw.py launch pretrain --rerun
```

Available stage names are defined per runner, for example:

- `run_adamw.py` / `run_sam.py`: `pretrain`, `pretrain_eval`, `hf`, `hf_eval`, `quant_eval`, `perturb`, `perturb_eval`, `sft`, `sft_eval`, `ewc_sft`, `ewc_sft_eval`.
- `run_midtrain.py`: `midtrain`, `midtrain_eval`, `hf`, `hf_eval_olmo`, `q_eval_olmo`, `sft`, `sft_eval`, `sft_hf`, `sft_hf_eval`.

Other useful subcommands: `relaunch` (cancel + resubmit), `cancel`, `print` (commands to pipe to bash),
`history`, and `export`.

## Where artifacts are stored

All artifacts are written to **Google Cloud Storage**, under the project's bucket root (`GS_PATH`, set in
the `experiments` project config and read in `launch/globals.py`). The project is selected by the
`Project.init(...)` call at the top of each runner (e.g. `60m-experiments`, `1b-experiments`), so paths
look like:

```
gs://<bucket>/outputs/<project>/<ArtifactType>/<run_name>/...
# e.g. gs://cmu-gpucloud-iwatts/outputs/60m-experiments/PretrainedModel/<run_name>/final-unsharded
```

Each artifact type lives in its own subfolder (the `relpath` property on each class in
`launch/artifacts.py`):

| Artifact | GCS subfolder |
| --- | --- |
| `PretrainedModel` | `PretrainedModel/<run_name>` |
| `MidtrainedModel` | `MidtrainedModel/<run_name>` |
| `AnnealedModel` / `AnnealedModel2` | `AnnealedModel/<run_name>` / `AnnealedModel2/<run_name>` |
| `SFTModel` | `SFTModel/<sft_dataset>/<run_name>` |
| `EWC_SFT` | `EWC_SFT/<sft_dataset>/<run_name>` |
| `PerturbedModel` | `PerturbedModel/<run_name>` |
| `HFModel` | `HFModel/<run_name>` |
| `ModelEvaluation` | `ModelEvaluation/<run_name>-eval.json` |
| `ModelEvaluationDownstream` | `ModelEvaluationDownstream/<run_name>-downstream-eval.jsonl` |

Trained model checkpoints are saved at `<relpath>/final-unsharded`. A stage is **skipped automatically**
if its output already exists in GCS (the `exists` check), so re-running a pipeline only fills in what is
missing. Jobs stage data/checkpoints to a scratch dir (`LOCAL_OUTPUT_PATH`) during execution and sync
results back to GCS on completion. Evaluation results can be pulled down with `gsutil`, e.g.:

```bash
gsutil cp -r gs://<bucket>/outputs/<project>/ModelEvaluation results/
```
