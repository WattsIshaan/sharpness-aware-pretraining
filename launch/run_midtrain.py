"""Midtraining runner."""

from experiments import Project # type: ignore
Project.init('1b-experiments')

from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import MidtrainedModel, ModelEvaluationDownstream, HFModel
from launch.quantize import build_quantized_model_evaluation_downstreams_olmo
from launch.sft_1b import build_sft_models, build_sft_model_evaluations

# Midtrain the base models
midtrained_models = ArtifactSet.from_product(
    cls=MidtrainedModel,
    params={
        'optimizer': ['sam', 'adamw'],
        'midtrain_gpus': [8],
        'per_device_train_batch_size': [4],
        'midtrain_tokens': [50],
        'global_train_batch_size': [1024],
    }
)
midtrain_eval = midtrained_models.map(lambda model: ModelEvaluationDownstream(model=model))

# HF conversion + downstream evals of the midtrained models
hf_models = midtrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations_olmo = hf_models.map(lambda model: ModelEvaluationDownstream(model=model, hf_model=True))
quantized_model_evaluations_olmo = build_quantized_model_evaluation_downstreams_olmo(hf_models)

# SFT (continued pretraining) on the midtrained models
sft_models = build_sft_models(midtrained_models)
sft_model_evaluations_ft_loss = build_sft_model_evaluations(sft_models)
sft_hf_models = sft_models.map(lambda model: HFModel(pretrained_model=model))
sft_hf_model_evaluations_olmo = sft_hf_models.map(lambda model: ModelEvaluationDownstream(model=model, hf_model=True))

# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)

# Midtrain stages
executor.stage('midtrain', midtrained_models)
executor.stage('midtrain_eval', midtrain_eval)

# HF + downstream eval stages
executor.stage('hf', hf_models)
executor.stage('hf_eval_olmo', hf_model_evaluations_olmo)
executor.stage('q_eval_olmo', quantized_model_evaluations_olmo)

# SFT stages
executor.stage('sft', sft_models)
executor.stage('sft_eval', sft_model_evaluations_ft_loss)
executor.stage('sft_hf', sft_hf_models)
executor.stage('sft_hf_eval', sft_hf_model_evaluations_olmo)

if __name__ == '__main__':
    executor.auto_cli()
