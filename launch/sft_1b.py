"""SFT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import SFTModel, ModelEvaluation

def build_sft_models(midtrained_models: ArtifactSet) -> ArtifactSet:
    return midtrained_models.map_flatten(
        lambda midtrained_model: ArtifactSet.from_product(
            cls=SFTModel,
            params=dict(
                train_tokens=[50],  # in Million
                pretrained_model=midtrained_model,
                sft_dataset='stackmathqa',
                optimizer='adamw',
                learning_rate=[2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 8e-5, 1e-4, 1.5e-4, 2e-4],
                weight_decay=[0],
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                sft_gpus=2,
            )
        )
    )


def build_sft_model_evaluations(sft_models: ArtifactSet) -> ArtifactSet:
    sft_eval = sft_models.map(lambda model: ModelEvaluation(model=model))
    return sft_eval