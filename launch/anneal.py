"""SFT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import AnnealedModel, AnnealedModel2, ModelEvaluation

PT_TOKENS = {
    "20m": {
        17: [4, 8, 16],
        65: [32, 64],
    },
    "60m": {
        25 : [12, 24],
        100: [48, 96],
        200: [192],
    },
    "150m": {
        31: [15, 30],
        61: [60],
        121: [120],
        250: [240],
    },
}

def build_annealed_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=AnnealedModel,
            params=dict(
                pretrained_model=pretrained_model,
                anneal_gpus=8,
                anneal_optim=["adamw", 'sam'],
                anneal_match="token", #"compute"
                anneal_percent=[10],
                pt_token=PT_TOKENS[pretrained_model.model_size][pretrained_model.train_tokens]
            )
        )
    )

def build_annealed_models2(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=AnnealedModel2,
            params=dict(
                pretrained_model=pretrained_model,
                anneal_gpus=8,
                anneal_optim=["adamw"],
                anneal_percent=[5, 10, 20, 50, 100],
                pt_token=PT_TOKENS[pretrained_model.model_size][pretrained_model.train_tokens]
            )
        )
    )

def build_anneal_model_evaluations(anneal_models: ArtifactSet) -> ArtifactSet:
    return anneal_models.map(lambda model: ModelEvaluation(model=model))