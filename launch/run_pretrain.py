"""Main entry point for the Wikipedia pretraining experiment."""

from experiments import SlurmExecutor

from launch.globals import PROJECT_NAME, LOCAL_DATA_PATH, CODE_PATH
from launch.pretrain import sam_pretrained_models, sgd_pretrained_models


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

if __name__ == '__main__':
    executor.auto_cli()