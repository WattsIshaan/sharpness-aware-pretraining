"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import ModelEvaluation

def build_quantized_model_evaluations(hf_models: ArtifactSet) -> ArtifactSet:
    return hf_models.map_flatten(
        lambda model: ArtifactSet.from_product(
            cls=ModelEvaluation,
            params=dict(
                model=model,
                hf_model=True,
                quant_bit=[4, 8],
            )
        )
    )