"""SAM pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('large-scale-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation
from launch.cpt import build_cpt_models, build_cpt_model_evaluations
from launch.anneal import build_annealed_models, build_anneal_model_evaluations


# Build SAM-only pretrained models
sam_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sam'],
            'learning_rate': [3e-4],
            'train_tokens': [400],
            'weight_decay': [0.1],
            'batch_size': [256],
            'scheduler_name': ['constant_with_warmup'],
            'sam_rho': [0.05],
            'scheduler_t_warmup': [2000],
            'scheduler_alpha_f': [0.1],
            'sam_base_optimizer': ['adamw'],
            'model_size': ['60m']
        }
    )

# Evaluations for SAM models
# sam_model_evaluations = sam_pretrained_models.map(lambda model: ModelEvaluation(model=model))

# Anneal Pretrained models
sam_annealed_models = build_annealed_models(sam_pretrained_models)
sam_annealed_model_evaluations = sam_annealed_models.map(lambda model: ModelEvaluation(model=model))

# CPT stages for SAM models
sam_cpt_models = build_cpt_models(sam_pretrained_models)
sam_cpt_model_evaluations = build_cpt_model_evaluations(sam_cpt_models)

# CPT stages for Annealed SAM models
sam_annealed_cpt_models = build_cpt_models(sam_annealed_models)
sam_annealed_cpt_model_evaluations = build_cpt_model_evaluations(sam_annealed_cpt_models)


# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate olmo'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)


# Stage SAM pretraining and evaluations
executor.stage('sam_pretrain', sam_pretrained_models)
# executor.stage('sam_eval', sam_model_evaluations)

# Stage Annealed SAM models and evaluations
executor.stage('sam_anneal', sam_annealed_models)
executor.stage('sam_anneal_eval', sam_annealed_model_evaluations)

# Stage CPT and its evaluations
# executor.stage('sam_cpt', sam_cpt_models)
# executor.stage('sam_cpt_eval', sam_cpt_model_evaluations)

# Stage Annealed CPT and its evaluations
executor.stage('sam_anneal_cpt', sam_annealed_cpt_models)
executor.stage('sam_anneal_cpt_eval', sam_annealed_cpt_model_evaluations)


if __name__ == '__main__':
    executor.auto_cli()


