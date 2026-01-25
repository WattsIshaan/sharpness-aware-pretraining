"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import AnnealedModel, ModelEvaluation

PT_TOKENS = {
    "20m": {
        17: {3e-3: [4, 8, 16]},
        65: {1e-3: [32, 64]},
    },
    "60m": {
        25 : {1e-3: [12, 24]},
        100: {6e-4: [48, 96]},
        200: {3e-4: [192]},
    },
    "150m": {
        31: {1e-3: [15, 30]},
        61: {6e-4: [60]},
        121: {3e-4: [120]},
    },
}

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
                pt_token=PT_TOKENS[pretrained_model.model_size][pretrained_model.train_tokens][pretrained_model.pt_lr]
            )
        )
    )

def build_anneal_model_evaluations(anneal_models: ArtifactSet) -> ArtifactSet:
    return anneal_models.map(lambda model: ModelEvaluation(model=model))