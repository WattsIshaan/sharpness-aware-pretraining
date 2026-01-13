"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import CPTModel, ModelEvaluation

def build_cpt_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=CPTModel,
            params=dict(
                train_tokens=[2, 50, 100],  # in Million
                pretrained_model=pretrained_model,
                cpt_dataset=['musicpile', 'starcoder'],
                # cpt_dataset=['tulu'],
                optimizer='adamw',
                # learning_rate=[1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 3e-3],
                # learning_rate=[1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4],
                # learning_rate=[2e-5, 4e-5, 8e-5],
                # learning_rate=[6e-4, 7e-4, 8e-4, 1.5e-3],
                # learning_rate=[2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4],
                learning_rate=[4e-5, 8e-5, 1e-4],
                weight_decay=0.1,
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                cpt_gpus=2
            )
        )
    )


def build_cpt_model_evaluations(cpt_models: ArtifactSet) -> ArtifactSet:
    return cpt_models.map(lambda model: ModelEvaluation(model=model))