"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import AnnealedModel, ModelEvaluation

def build_annealed_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=AnnealedModel,
            params=dict(
                pretrained_model=pretrained_model,
                # pretrain_ckpt_step=[15000, 30000, 55000, 110000, 220000, 445000, 890000],
                # pretrain_ckpt_step=[670000, 40000, 85000, 165000, 335000],
                pretrain_ckpt_step=[40000, 85000],
                # pretrain_ckpt_step=[610000, 35000, 75000, 155000, 305000],
                # pretrain_ckpt_step=[45000, 90000, 175000, 350000, 700000],
                anneal_gpus=8,
                # anneal_steps=[1000, 2000, 40000],
                anneal_percent=[10]
            )
        )
    )

def build_anneal_model_evaluations(anneal_models: ArtifactSet) -> ArtifactSet:
    return anneal_models.map(lambda model: ModelEvaluation(model=model))