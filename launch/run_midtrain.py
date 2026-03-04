"""Midtraining runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('1b-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import MidtrainedModel, ModelEvaluationDownstream, HFModel
from launch.sft import build_sft_models
from launch.quantize import build_quantized_model_evaluation_downstreams

midtrained_models = ArtifactSet.from_product(
    cls=MidtrainedModel,
    params={
        'optimizer': ['sam'],
        'sam_rho': [5e-2],
        'midtrain_gpus': [8],
        'per_device_train_batch_size': [4],
    }
)

hf_models = midtrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluationDownstream(model=model))

quantized_model_evaluations = build_quantized_model_evaluation_downstreams(hf_models)

sft_models = build_sft_models(hf_models)
sft_model_evaluations = sft_models.map(lambda model: ModelEvaluationDownstream(model=model))

# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)

# Stages
executor.stage('midtrain', midtrained_models)
executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)
executor.stage('quant_eval', quantized_model_evaluations)
executor.stage('sft', sft_models)
executor.stage('sft_eval', sft_model_evaluations)

if __name__ == '__main__':
    executor.auto_cli()
