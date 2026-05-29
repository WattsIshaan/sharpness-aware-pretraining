"""SFT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import SFTModel, ModelEvaluation


def build_sft_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=SFTModel,
            params=dict(
                train_tokens=[10],  # in Million
                pretrained_model=pretrained_model,
                sft_dataset='starcoder', # tulu, stackmathqa, musicpile, gsm8k
                optimizer='adamw',
                learning_rate=[3e-4, 4e-4, 5e-4, 6e-4, 8e-4],
                weight_decay=[0.1],
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                sft_gpus=1,
            )
        )
    )


def build_sft_model_evaluations(sft_models: ArtifactSet) -> ArtifactSet:
    return sft_models.map(lambda model: ModelEvaluation(model=model))
