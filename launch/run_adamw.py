"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('large-scale-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation
from launch.cpt import build_cpt_models, build_cpt_model_evaluations
from launch.anneal import build_annealed_models, build_anneal_model_evaluations

# Build SGD-only pretrained models
adamw_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['adamw'],
            'learning_rate': [3e-4],
            'train_tokens': [400],
            'weight_decay': [0.1],
            'batch_size': [256],
            'scheduler_name': ['constant_with_warmup'],
            'scheduler_t_warmup': [2000],
            'scheduler_alpha_f': [0.1],
            'model_size': ['60m']
        }
    )

# Evaluations for SGD models
# adamw_model_evaluations = adamw_pretrained_models.map(lambda model: ModelEvaluation(model=model))


# Anneal Pretrained models
adamw_annealed_models = build_annealed_models(adamw_pretrained_models)
adamw_annealed_model_evaluations = adamw_annealed_models.map(lambda model: ModelEvaluation(model=model))

# CPT stages for SGD models
adamw_cpt_models = build_cpt_models(adamw_pretrained_models)
adamw_cpt_model_evaluations = build_cpt_model_evaluations(adamw_cpt_models)

# CPT stages for SGD models
adamw_annealed_cpt_models = build_cpt_models(adamw_annealed_models)
adamw_annealed_cpt_model_evaluations = build_cpt_model_evaluations(adamw_annealed_cpt_models)


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
executor.stage('adamw_pretrain', adamw_pretrained_models)
# executor.stage('adamw_eval', adamw_model_evaluations)

executor.stage('adamw_anneal', adamw_annealed_models)
executor.stage('adamw_anneal_eval', adamw_annealed_model_evaluations)

# Stage CPT and its evaluations
# executor.stage('adamw_cpt', adamw_cpt_models)
# executor.stage('adamw_cpt_eval', adamw_cpt_model_evaluations)

# Stage Annealed CPT and its evaluations
executor.stage('adamw_anneal_cpt', adamw_annealed_cpt_models)
executor.stage('adamw_anneal_cpt_eval', adamw_annealed_cpt_model_evaluations)




if __name__ == '__main__':
    executor.auto_cli()


