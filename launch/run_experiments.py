"""Main entry point for the Wikipedia pretraining experiment."""

from experiments import SlurmExecutor

from launch.globals import PROJECT_NAME, LOCAL_DATA_PATH, CODE_PATH
from launch.pretrain import sam_pretrained_models, sgd_pretrained_models, sam_model_evaluations, sgd_model_evaluations
from launch.cpt import sam_cpt_models, sgd_cpt_models, sam_cpt_model_evaluations, sgd_cpt_model_evaluations

# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    project=PROJECT_NAME,
    artifact_path=LOCAL_DATA_PATH,
    code_path=CODE_PATH,
    setup_command=setup_command,
)

# Create and register pretrained models with training and validation data
executor.stage(
    'sam_pretrain',
    sam_pretrained_models,
)

executor.stage(
    'sgd_pretrain',
    sgd_pretrained_models,
)

# # Add evaluation stages
executor.stage(
    'sam_eval',
    sam_model_evaluations,
)

executor.stage(
    'sgd_eval',
    sgd_model_evaluations,
)

# Add CPT (Continual PreTraining) stages
executor.stage(
    'sam_cpt',
    sam_cpt_models,
)

executor.stage(
    'sgd_cpt',
    sgd_cpt_models,
)

# Add CPT evaluation stages
executor.stage(
    'sam_cpt_eval',
    sam_cpt_model_evaluations,
)

executor.stage(
    'sgd_cpt_eval',
    sgd_cpt_model_evaluations,
)

if __name__ == '__main__':
    executor.auto_cli()