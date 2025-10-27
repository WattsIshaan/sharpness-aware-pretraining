"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from launch import globals as G
G.load_project('sgd-gridsearch')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation
from launch.cpt import build_cpt_models, build_cpt_model_evaluations


# Build SGD-only pretrained models
sgd_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sgd'],
            'learning_rate': [0.3 ,1],
            'train_tokens': [8, 16],
            'weight_decay': [1e-5],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'scheduler_alpha_f': [0.1],
        }
    )

# Evaluations for SGD models
sgd_model_evaluations = sgd_pretrained_models.map(lambda model: ModelEvaluation(model=model))

# CPT stages for SGD models
sgd_cpt_models = build_cpt_models(sgd_pretrained_models)
sgd_cpt_model_evaluations = build_cpt_model_evaluations(sgd_cpt_models)


# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    project=G.PROJECT_NAME,
    artifact_path=G.LOCAL_DATA_PATH,
    code_path=G.CODE_PATH,
    setup_command=setup_command,
)


# Stage SGD pretraining and evaluations
executor.stage('sgd_pretrain', sgd_pretrained_models)
executor.stage('sgd_eval', sgd_model_evaluations)

# Stage CPT and its evaluations
# executor.stage('sgd_cpt', sgd_cpt_models)
# executor.stage('sgd_cpt_eval', sgd_cpt_model_evaluations)


if __name__ == '__main__':
    executor.auto_cli()


