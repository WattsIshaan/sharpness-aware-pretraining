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
                pretrain_ckpt_step=[85000],
                anneal_gpus=4,
                anneal_steps=[1000, 2000, 4000],
                # anneal_percent=[5, 10, 20]
            )
        )
    )

def build_anneal_model_evaluations(anneal_models: ArtifactSet) -> ArtifactSet:
    return anneal_models.map(lambda model: ModelEvaluation(model=model))