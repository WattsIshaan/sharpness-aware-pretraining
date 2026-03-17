"""Midtraining runner."""

# 1) Load project configuration as early as possible
from experiments import Project
Project.init('1b-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import MidtrainedModel, ModelEvaluationDownstream, HFModel, ModelEvaluation, ModelEvaluationDownstreamOLMo, CPTModel
from launch.quantize import build_quantized_model_evaluation_downstreams_olmo
from launch.cpt_1b import build_cpt_models, build_cpt_model_evaluations

OPTIM_INFO = {
    "adamw": [5e-2],
    # "sam": [5e-2, 1e-1, 1.5e-1, 2e-1],
    "sam": [5e-2, 1e-1],
}

# for i, optim in enumerate(["adamw", "sam"]):
for i, optim in enumerate(["adamw", "sam"]):
    if i == 0:
        midtrained_models = ArtifactSet.from_product(
            cls=MidtrainedModel,
            params={
                'optimizer': [optim],
                'sam_rho': OPTIM_INFO[optim],
                'midtrain_gpus': [8],
                'per_device_train_batch_size': [4],
                'midtrain_tokens': [5],
                'global_train_batch_size': [1024],
                'load_path': ['PretrainedModel/OLMo2-1b-tk4T-adamw/step950000-unsharded'],
                'learning_rate': [2.7699e-4],
                # 'anneal_sam': [True],
                # 'sam_per_microbatch': [True, False],
            }
        )
    else:
        midtrained_models += ArtifactSet.from_product(
            cls=MidtrainedModel,
            params={
                'optimizer': [optim],
                'sam_rho': OPTIM_INFO[optim],
                'midtrain_gpus': [8],
                'per_device_train_batch_size': [4],
                'midtrain_tokens': [5],
                'global_train_batch_size': [1024],
                'load_path': ['PretrainedModel/OLMo2-1b-tk4T-adamw/step950000-unsharded'],
                'learning_rate': [2.7699e-4],
                # 'anneal_sam': [True],
                # 'sam_per_microbatch': [True, False],
            }
        )

midtrain_eval = midtrained_models.map(lambda model: ModelEvaluationDownstreamOLMo(model=model))

cpt_models = build_cpt_models(midtrained_models)
cpt_model_evaluations = build_cpt_model_evaluations(cpt_models)

cpt_hf_models = cpt_models.map(lambda model: HFModel(pretrained_model=model))
cpt_hf_model_evaluations = cpt_hf_models.map(lambda model: ModelEvaluationDownstream(model=model, batch_size=16))

hf_models = midtrained_models.map(lambda model: HFModel(pretrained_model=model))
hf_model_evaluations = hf_models.map(lambda model: ModelEvaluationDownstream(model=model))
hf_model_evaluations_olmo = hf_models.map(lambda model: ModelEvaluationDownstreamOLMo(model=model, hf_model=True))



# quantized_model_evaluations = build_quantized_model_evaluation_downstreams(hf_models)
quantized_model_evaluations_olmo = build_quantized_model_evaluation_downstreams_olmo(hf_models)

# sft_models = build_sft_models(hf_models)
# sft_model_evaluations = build_sft_model_evaluations(sft_models)

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
executor.stage('midtrain_eval', midtrain_eval)
executor.stage('cpt', cpt_models)
executor.stage('cpt_eval', cpt_model_evaluations)
executor.stage('cpt_hf', cpt_hf_models)
executor.stage('cpt_hf_eval', cpt_hf_model_evaluations)
executor.stage('hf', hf_models)
executor.stage('hf_eval', hf_model_evaluations)
executor.stage('hf_eval_olmo', hf_model_evaluations_olmo)
# executor.stage('q_eval', quantized_model_evaluations)
executor.stage('q_eval_olmo', quantized_model_evaluations_olmo)
# executor.stage('sft', sft_models)
# executor.stage('sft_eval', sft_model_evaluations)

if __name__ == '__main__':
    executor.auto_cli()
