"""Quantized evaluation artifacts for HFModel checkpoints."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import ModelEvaluation, ModelEvaluationDownstream, ModelEvaluationDownstreamOLMo

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

def build_quantized_model_evaluation_downstreams(hf_models: ArtifactSet) -> ArtifactSet:
    four_bit = hf_models.map(
        lambda model: ModelEvaluationDownstream(model=model, load_in_4bit=True)
    )
    eight_bit = hf_models.map(
        lambda model: ModelEvaluationDownstream(model=model, load_in_8bit=True)
    )
    return four_bit + eight_bit

def build_quantized_model_evaluation_downstreams_olmo(hf_models: ArtifactSet) -> ArtifactSet:
    return hf_models.map_flatten(
        lambda model: ArtifactSet.from_product(
            cls=ModelEvaluationDownstreamOLMo,
            params=dict(
                model=model,
                hf_model=True,
                quant_bit=[4],
            )
        )
    )
