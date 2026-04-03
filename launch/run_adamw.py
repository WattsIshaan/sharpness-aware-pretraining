"""Muon (SGD-only) pretraining and evaluation runner."""

# 1) Load project configuration as early as possible
from experiments import Project # type: ignore
Project.init('60m-experiments')

# 2) Rest of imports
from experiments import SlurmExecutor, ArtifactSet  # type: ignore
from launch.artifacts import PretrainedModel, ModelEvaluation, HFModel, ModelEvaluationDownstreamOLMo
from launch.cpt import build_cpt_models, build_cpt_model_evaluations
from launch.ewc import build_ewc_cpt_models, build_ewc_cpt_model_evaluations
from launch.perturb import build_perturbed_models, build_perturbed_model_evaluations
from launch.quantize import build_quantized_model_evaluations
from launch.anneal import build_annealed_models, build_annealed_models2

# Build SGD-only pretrained models
pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['adamw'],
            # 'pt_lr': [1e-3],
            'pt_lr': [1e-4],
            # 'pt_lr': [-1],
            # 'train_tokens': [60, 120],
            # 'train_tokens': [12, 24, 48, 96, 192],
            # "train_tokens": [25],
            # "train_tokens": [61],
            'train_tokens': [200],
            # 'train_tokens': [61],
            # 'train_tokens': [15, 30, 60, 120],
            'weight_decay': [0.1],
            'batch_size': [256],
            # 'scheduler_name': ['cosine_with_warmup'],
            'scheduler_name': ['constant_with_warmup'],
            'scheduler_alpha_f': [0.1],
            'model_size': ['60m']
        }
    )

# Evaluations for SGD models
# model_evaluations = pretrained_models.map(lambda model: ModelEvaluation(model=model))

# Convert to HF Models
# hf_models = pretrained_models.map(lambda model: HFModel(pretrained_model=model))
# hf_model_evaluations = hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# # Quantize HF Models
# quantized_models_evaluations = build_quantized_model_evaluations(hf_models)

# # Perturbed pretrained models and evaluations
# perturbed_models = build_perturbed_models(pretrained_models)
# perturbed_model_evaluations = build_perturbed_model_evaluations(perturbed_models)

# # # Anneal Pretrained models
annealed_models = build_annealed_models(pretrained_models)
annealed_model_evaluations = annealed_models.map(lambda model: ModelEvaluation(model=model))


# # Perturbed annealed models and evaluations
# perturbed_annealed_models = build_perturbed_models(annealed_models)
# perturbed_annealed_evaluations = build_perturbed_model_evaluations(perturbed_annealed_models)

# # Convert to HF Models
# annealed_hf_models = annealed_models.map(lambda model: HFModel(pretrained_model=model))
# annealed_hf_model_evaluations = annealed_hf_models.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# # Quantize HF Models
# annealed_quantized_models_evaluations = build_quantized_model_evaluations(annealed_hf_models)

# Annealed models 2
# annealed_models2 = build_annealed_models2(pretrained_models)
# annealed_model_evaluations2 = annealed_models2.map(lambda model: ModelEvaluation(model=model))

# Perturbed annealed models 2 and evaluations
# perturbed_annealed_models2 = build_perturbed_models(annealed_models2)
# perturbed_annealed_evaluations2 = build_perturbed_model_evaluations(perturbed_annealed_models2)

# Convert to HF Models
# annealed_hf_models2 = annealed_models2.map(lambda model: HFModel(pretrained_model=model))
# annealed_hf_model_evaluations2 = annealed_hf_models2.map(lambda model: ModelEvaluation(model=model, hf_model=True))

# Quantize HF Models
# annealed_quantized_models_evaluations2 = build_quantized_model_evaluations(annealed_hf_models2)

# CPT stages for SGD models
# cpt_models = build_cpt_models(pretrained_models)
# cpt_model_evaluations = build_cpt_model_evaluations(cpt_models)
# cpt_models_evaluations_downstream = cpt_models.map(lambda model: ModelEvaluationDownstreamOLMo(model=model))

# CPT stages for SGD models
annealed_cpt_models = build_cpt_models(annealed_models)
annealed_cpt_model_evaluations = build_cpt_model_evaluations(annealed_cpt_models)

# CPT stages for Annealed models 2
# annealed_cpt_models2 = build_cpt_models(annealed_models2)
# annealed_cpt_model_evaluations2 = build_cpt_model_evaluations(annealed_cpt_models2)

# EWC CPT from base pretrained checkpoints (Elastic Weight Consolidation; ``launch/ewc.py``).
# Use ``build_ewc_cpt_models(annealed_models2)`` instead if you want EWC on annealed bases.
# ewc_cpt_models = build_ewc_cpt_models(pretrained_models)
# ewc_cpt_model_evaluations = build_ewc_cpt_model_evaluations(ewc_cpt_models)

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
executor.stage('pretrain', pretrained_models)
# executor.stage('pretrain_eval', model_evaluations)

# executor.stage('hf', hf_models)
# executor.stage('hf_eval', hf_model_evaluations)

# executor.stage('quant_eval', quantized_models_evaluations)

# executor.stage('perturb', perturbed_models)
# executor.stage('perturb_eval', perturbed_model_evaluations)

executor.stage('anneal', annealed_models)
executor.stage('anneal_eval', annealed_model_evaluations)
# executor.stage('anneal2', annealed_models2)
# executor.stage('anneal2_eval', annealed_model_evaluations2)

# executor.stage('anneal_perturb', perturbed_annealed_models)
# executor.stage('anneal_perturb_eval', perturbed_annealed_evaluations)
# executor.stage('anneal_hf', annealed_hf_models)
# executor.stage('anneal_hf_eval', annealed_hf_model_evaluations)

# executor.stage('anneal_quant_eval', annealed_quantized_models_evaluations)

# executor.stage('apt', perturbed_annealed_models2)
# executor.stage('apt_eval', perturbed_annealed_evaluations2)
# executor.stage('ahf', annealed_hf_models2)
# executor.stage('ahf_eval', annealed_hf_model_evaluations2)

# executor.stage('aq_eval', annealed_quantized_models_evaluations2)

# # Stage CPT and its evaluations
# executor.stage('cpt', cpt_models)
# executor.stage('cpt_eval', cpt_model_evaluations)
# executor.stage('cpt_eval_ds', cpt_models_evaluations_downstream)

# # Stage Annealed CPT and its evaluations
executor.stage('acpt', annealed_cpt_models)
executor.stage('acpt_eval', annealed_cpt_model_evaluations)
# executor.stage('acpt', annealed_cpt_models2)
# executor.stage('acpt_eval', annealed_cpt_model_evaluations2)

# executor.stage('ewc_cpt', ewc_cpt_models)
# executor.stage('ewc_cpt_eval', ewc_cpt_model_evaluations)




if __name__ == '__main__':
    executor.auto_cli()


