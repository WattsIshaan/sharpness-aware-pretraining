"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import AnnealedModel, ModelEvaluation

def build_annealed_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=AnnealedModel,
            params=dict(
                pretrained_model=pretrained_model,
                anneal_gpus=8,
                anneal_optim=["adamw", "sam"],
                anneal_match="token",
                anneal_percent=[10],
                pt_token=[48, 96]
            )
        )
    )

def build_anneal_model_evaluations(anneal_models: ArtifactSet) -> ArtifactSet:
    return anneal_models.map(lambda model: ModelEvaluation(model=model))