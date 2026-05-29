"""SAM pretraining and evaluation runner."""

from experiments import Project # type: ignore
Project.init('60m-experiments')

from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation, HFModel
from launch.sft_small import build_sft_models, build_sft_model_evaluations
from launch.ewc import build_ewc_sft_models, build_ewc_sft_model_evaluations
from launch.perturb import build_perturbed_models, build_perturbed_model_evaluations
from launch.quantize import build_quantized_model_evaluations
from launch.anneal import build_annealed_models, build_annealed_models2

# Build pretrained models
pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sam'],
            # -1 automatically chooses the final LR for the given token budget (for cosine schedules).
            'pt_lr': [-1],
            'train_tokens': [12, 24, 48, 96, 192],
            'weight_decay': [0.1],
            'batch_size': [256],
            # When using WSD, switch scheduler_name to the 'wsd' schedule.
            'scheduler_name': ['cosine_with_warmup'],
            # 'scheduler_name': ['constant_with_warmup'],
            'scheduler_alpha_f': [0.1],
            'model_size': ['60m'],
            'sam_rho': [0.05],
        }
    )

# ===========================================================================
# COSINE: operate directly on the pretrained (cosine-scheduled) models
# ===========================================================================
model_evaluations = pretrained_models.map(lambda model: ModelEvaluation(model=model))

hf_models = pretrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

quantized_models_evaluations = build_quantized_model_evaluations(hf_models)

perturbed_models = build_perturbed_models(pretrained_models)
perturbed_model_evaluations = build_perturbed_model_evaluations(perturbed_models)

sft_models = build_sft_models(pretrained_models)
sft_model_evaluations = build_sft_model_evaluations(sft_models)

# EWC SFT from base pretrained checkpoints (Elastic Weight Consolidation; ``launch/ewc.py``).
ewc_sft_models = build_ewc_sft_models(pretrained_models)
ewc_sft_model_evaluations = build_ewc_sft_model_evaluations(ewc_sft_models)

# ===========================================================================
# WSD (anneal): anneal the pretrained models, then operate on the annealed models
# ===========================================================================
# annealed_models = build_annealed_models(pretrained_models)
# annealed_model_evaluations = annealed_models.map(lambda model: ModelEvaluation(model=model))

# annealed_hf_models = annealed_models.map(lambda model: HFModel(pretrained_model=model))
# annealed_hf_model_evaluations = annealed_hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# annealed_quantized_models_evaluations = build_quantized_model_evaluations(annealed_hf_models)

# perturbed_annealed_models = build_perturbed_models(annealed_models)
# perturbed_annealed_evaluations = build_perturbed_model_evaluations(perturbed_annealed_models)

# annealed_sft_models = build_sft_models(annealed_models)
# annealed_sft_model_evaluations = build_sft_model_evaluations(annealed_sft_models)

# ===========================================================================
# WSD (anneal2): anneal with different percent annealing, then operate on those models
# ===========================================================================
# annealed_models2 = build_annealed_models2(pretrained_models)
# annealed_model_evaluations2 = annealed_models2.map(lambda model: ModelEvaluation(model=model))

# annealed_hf_models2 = annealed_models2.map(lambda model: HFModel(pretrained_model=model))
# annealed_hf_model_evaluations2 = annealed_hf_models2.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# annealed_quantized_models_evaluations2 = build_quantized_model_evaluations(annealed_hf_models2)

# perturbed_annealed_models2 = build_perturbed_models(annealed_models2)
# perturbed_annealed_evaluations2 = build_perturbed_model_evaluations(perturbed_annealed_models2)

# annealed_sft_models2 = build_sft_models(annealed_models2)
# annealed_sft_model_evaluations2 = build_sft_model_evaluations(annealed_sft_models2)

# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)

# ===========================================================================
# COSINE stages
# ===========================================================================
executor.stage('pretrain', pretrained_models)
executor.stage('pretrain_eval', model_evaluations)
executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)
executor.stage('quant_eval', quantized_models_evaluations)
executor.stage('perturb', perturbed_models)
executor.stage('perturb_eval', perturbed_model_evaluations)
executor.stage('sft', sft_models)
executor.stage('sft_eval', sft_model_evaluations)
executor.stage('ewc_sft', ewc_sft_models)
executor.stage('ewc_sft_eval', ewc_sft_model_evaluations)

# ===========================================================================
# WSD (anneal) stages
# ===========================================================================
# executor.stage('anneal', annealed_models)
# executor.stage('anneal_eval', annealed_model_evaluations)
# executor.stage('anneal_hf', annealed_hf_models)
# executor.stage('anneal_hf_eval', annealed_hf_model_evaluations)
# executor.stage('anneal_quant_eval', annealed_quantized_models_evaluations)
# executor.stage('anneal_perturb', perturbed_annealed_models)
# executor.stage('anneal_perturb_eval', perturbed_annealed_evaluations)
# executor.stage('anneal_sft', annealed_sft_models)
# executor.stage('anneal_sft_eval', annealed_sft_model_evaluations)

# ===========================================================================
# WSD (anneal2) stages
# ===========================================================================
# executor.stage('anneal2', annealed_models2)
# executor.stage('anneal2_eval', annealed_model_evaluations2)
# executor.stage('anneal2_hf', annealed_hf_models2)
# executor.stage('anneal2_hf_eval', annealed_hf_model_evaluations2)
# executor.stage('anneal2_quant_eval', annealed_quantized_models_evaluations2)
# executor.stage('anneal2_perturb', perturbed_annealed_models2)
# executor.stage('anneal2_perturb_eval', perturbed_annealed_evaluations2)
# executor.stage('anneal2_sft', annealed_sft_models2)
# executor.stage('anneal2_sft_eval', annealed_sft_model_evaluations2)


if __name__ == '__main__':
    executor.auto_cli()
