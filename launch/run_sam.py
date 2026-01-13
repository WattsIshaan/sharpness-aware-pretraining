"""SAM pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('60m-experiments2')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation, HFModel
from launch.cpt import build_cpt_models, build_cpt_model_evaluations
from launch.quantize import build_quantized_model_evaluations
from launch.anneal import build_annealed_models, build_anneal_model_evaluations


# Build SAM-only pretrained models
pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sam'],
            'pt_lr': [-1],
            'train_tokens': [192],#12, 24, 
            # 'train_tokens': [4, 8, 16, 32, 64],
            # 'train_tokens': [15, 30, 60, 120],
            # 'train_tokens': [120],
            'weight_decay': [0.1],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'sam_rho': [5e-2],
            'scheduler_t_warmup': [2000],
            'scheduler_alpha_f': [0.1],
            'sam_base_optimizer': ['adamw'],
            'model_size': ['60m']
        }
    )

# Evaluations for SAM models
model_evaluations = pretrained_models.map(lambda model: ModelEvaluation(model=model))

# Convert to HF Models
hf_models = pretrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# Quantize HF Models
quantized_models_evaluations = build_quantized_model_evaluations(hf_models)


# Anneal Pretrained models
# sam_annealed_models = build_annealed_models(sam_pretrained_models)
# sam_annealed_model_evaluations = sam_annealed_models.map(lambda model: ModelEvaluation(model=model))

# CPT stages for SAM models
cpt_models = build_cpt_models(pretrained_models)
cpt_model_evaluations = build_cpt_model_evaluations(cpt_models)

# CPT stages for Annealed SAM models
# annealed_cpt_models = build_cpt_models(annealed_models)
# annealed_cpt_model_evaluations = build_cpt_model_evaluations(annealed_cpt_models)


# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)


# Stage SAM pretraining and evaluations
executor.stage('pretrain', pretrained_models)
executor.stage('pretrain_eval', model_evaluations)

executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)

executor.stage('quant_eval', quantized_models_evaluations)

# Stage Annealed SAM models and evaluations
# executor.stage('anneal', annealed_models)
# executor.stage('anneal_eval', annealed_model_evaluations)

# Stage CPT and its evaluations
executor.stage('cpt', cpt_models)
executor.stage('cpt_eval', cpt_model_evaluations)

# Stage Annealed CPT and its evaluations
# executor.stage('anneal_cpt', annealed_cpt_models)
# executor.stage('anneal_cpt_eval', annealed_cpt_model_evaluations)


if __name__ == '__main__':
    executor.auto_cli()


