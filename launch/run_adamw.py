"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('150m-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation, HFModel
from launch.cpt import build_cpt_models, build_cpt_model_evaluations
from launch.perturb import build_perturbed_models, build_perturbed_model_evaluations
from launch.quantize import build_quantized_model_evaluations
from launch.anneal import build_annealed_models, build_anneal_model_evaluations

# Build SGD-only pretrained models
pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['adamw'],
            'pt_lr': [3e-4],
            'train_tokens': [121],#12, 24,  
            # 'train_tokens': [4, 8, 16, 32, 64],
            # 'train_tokens': [4, 8, 16, 32, 64],
            # 'train_tokens': [65],
            'weight_decay': [0.1],
            'batch_size': [256],
            'scheduler_name': ['constant_with_warmup'],
            'scheduler_t_warmup': [3000],
            'scheduler_alpha_f': [0.1],
            'model_size': ['150m']
        }
    )

# Evaluations for SGD models
model_evaluations = pretrained_models.map(lambda model: ModelEvaluation(model=model))

# Convert to HF Models
hf_models = pretrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# Quantize HF Models
quantized_models_evaluations = build_quantized_model_evaluations(hf_models)

# Perturbed pretrained models and evaluations
perturbed_models = build_perturbed_models(pretrained_models)
perturbed_model_evaluations = build_perturbed_model_evaluations(perturbed_models)

# Anneal Pretrained models
annealed_models = build_annealed_models(pretrained_models)
annealed_model_evaluations = annealed_models.map(lambda model: ModelEvaluation(model=model))

# Perturbed annealed models and evaluations
perturbed_annealed_models = build_perturbed_models(annealed_models)
perturbed_annealed_evaluations = build_perturbed_model_evaluations(perturbed_annealed_models)

# CPT stages for SGD models
cpt_models = build_cpt_models(pretrained_models)
cpt_model_evaluations = build_cpt_model_evaluations(cpt_models)

# CPT stages for SGD models
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


# Stage SGD pretraining and evaluations
executor.stage('pretrain', pretrained_models)
executor.stage('pretrain_eval', model_evaluations)

executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)

executor.stage('quant_eval', quantized_models_evaluations)

executor.stage('perturb', perturbed_models)
executor.stage('perturb_eval', perturbed_model_evaluations)

executor.stage('anneal', annealed_models)
executor.stage('anneal_eval', annealed_model_evaluations)

executor.stage('perturb_anneal', perturbed_annealed_models)
executor.stage('perturb_anneal_eval', perturbed_annealed_evaluations)

# Stage CPT and its evaluations
executor.stage('cpt', cpt_models)
executor.stage('cpt_eval', cpt_model_evaluations)

# Stage Annealed CPT and its evaluations
# executor.stage('anneal_cpt', annealed_cpt_models)
# executor.stage('anneal_cpt_eval', annealed_cpt_model_evaluations)




if __name__ == '__main__':
    executor.auto_cli()


