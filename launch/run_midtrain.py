"""Midtraining runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('1b-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import MidtrainedModel, ModelEvaluationDownstream, HFModel, ModelEvaluation
from launch.perturb import build_perturbed_models
from launch.quantize import build_quantized_model_evaluation_downstreams
midtrained_models = ArtifactSet.from_product(
    cls=MidtrainedModel,
    params={
        'optimizer': ['adamw', 'sam'],
        'sam_rho': [5e-2],
        'midtrain_gpus': [8],
        'per_device_train_batch_size': [4],
    }
)

downstream_evaluations = midtrained_models.map(lambda model: ModelEvaluationDownstream(model=model))
pplx_evaluations = midtrained_models.map(lambda model: ModelEvaluation(model=model))

perturbed_models = build_perturbed_models(midtrained_models)
perturbed_downstream_evaluations = perturbed_models.map(lambda model: ModelEvaluationDownstream(model=model))

hf_models = midtrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluationDownstream(model=model, hf_model=True))
hf_model_pplx_evaluations = hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

quantized_model_evaluations = build_quantized_model_evaluation_downstreams(hf_models)

# Setup command for the executor
setup_command = ' && '.join([
    'source ~/miniconda3/etc/profile.d/conda.sh',
    'conda activate forgetting2'
])

# Create executor
executor = SlurmExecutor(
    setup_command=setup_command,
)


# Stage midtraining
executor.stage('midtrain', midtrained_models)
executor.stage('eval', downstream_evaluations)
executor.stage('pplx_eval', pplx_evaluations)
executor.stage('perturb', perturbed_models)
executor.stage('perturb_eval', perturbed_downstream_evaluations)
executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)
executor.stage('hf_pplx_eval', hf_model_pplx_evaluations)
executor.stage('quant_eval', quantized_model_evaluations)
if __name__ == '__main__':
    executor.auto_cli()


