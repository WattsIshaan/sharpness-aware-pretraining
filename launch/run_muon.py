"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from launch import globals as G
G.load_project('muon')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation
from launch.cpt import build_cpt_models, build_cpt_model_evaluations


# Build SGD-only pretrained models
muon_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['muon'],
            'learning_rate': [5e-4],
            'momentum' : [0.1],
            'train_tokens': [1],
            'weight_decay': [0.02],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'scheduler_alpha_f': [0.1],
            'pretrain_gpus' : [1]
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
    'conda activate env310'
])

# Create executor
executor = SlurmExecutor(
    project=G.PROJECT_NAME,
    artifact_path=G.LOCAL_DATA_PATH,
    code_path=G.CODE_PATH,
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


