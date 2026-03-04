"""SFT artifacts for supervised fine-tuning experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import SFTModel, ModelEvaluationDownstream

def build_sft_models(hf_models: ArtifactSet) -> ArtifactSet:
    return hf_models.map_flatten(
        lambda hf_model: ArtifactSet.from_product(
            cls=SFTModel,
            params=dict(
                hf_model=hf_model,
                learning_rate=[3e-5],
                weight_decay=[0.0],
                num_train_epochs=[2],
                sft_gpus=8,
            )
        )
    )


def build_sft_model_evaluations(sft_models: ArtifactSet) -> ArtifactSet:
    return sft_models.map(lambda model: ModelEvaluationDownstream(model=model))
