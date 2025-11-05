"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import CPTModel, ModelEvaluation

def build_cpt_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=CPTModel,
            params=dict(
                train_tokens=[20],  # in Million
                pretrained_model=pretrained_model,
                training_dataset_name='starcoder',
                optimizer='adamw',
                learning_rate=[1e-3, 2e-4, 4e-5],
                weight_decay=0.1,
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                cpt_gpus=1
            )
        )
    )


def build_cpt_model_evaluations(cpt_models: ArtifactSet) -> ArtifactSet:
    return cpt_models.map(lambda model: ModelEvaluation(model=model))