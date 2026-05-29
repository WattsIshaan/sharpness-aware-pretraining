"""Perturbed model artifacts for evaluation runs."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import PerturbedModel, ModelEvaluation


def build_perturbed_models(base_models: ArtifactSet) -> ArtifactSet:
    return base_models.map_flatten(
        lambda model: ArtifactSet.from_product(
            cls=PerturbedModel,
            params=dict(
                base_model=model,
                sigma=[0.009, 0.013, 0.017, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1],
            ),
        )
    )


def build_perturbed_model_evaluations(perturbed_models: ArtifactSet) -> ArtifactSet:
    return perturbed_models.map(lambda model: ModelEvaluation(model=model))
