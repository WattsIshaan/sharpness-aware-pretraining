"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet, Project  # type: ignore
Project.init("muon")

from launch.artifacts import PretrainedModel, ModelEvaluation
from launch.cpt import build_cpt_models, build_cpt_model_evaluations


# Build SGD-only pretrained models
muon_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['muon'],
            'muon_learning_rate': [5e-2, 5e-3, 5e-4, 5e-5],
            'muon_momentum' : [0.95],
            'muon_weight_decay': 0.1,
            'train_tokens': [4,8,16,32,64],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'scheduler_alpha_f': [0.1],
            'pretrain_gpus' : 4,
            'learning_rate': 3e-4,
            'weight_decay': 0.1,
        }
    )

# Evaluations for SGD models
muon_model_evaluations = muon_pretrained_models.map(lambda model: ModelEvaluation(model=model))

# CPT stages for SGD models
muon_cpt_models = build_cpt_models(muon_pretrained_models)
muon_cpt_model_evaluations = build_cpt_model_evaluations(muon_cpt_models)


# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate myenv310'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)


# Stage SGD pretraining and evaluations
executor.stage('muon_pretrain', muon_pretrained_models)
executor.stage('muon_eval', muon_model_evaluations)

# Stage CPT and its evaluations
executor.stage('muon_cpt', muon_cpt_models)
executor.stage('muon_cpt_eval', muon_cpt_model_evaluations)


if __name__ == '__main__':
    executor.auto_cli()


