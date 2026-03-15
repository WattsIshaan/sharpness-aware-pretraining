"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import CPTModel, ModelEvaluation, ModelEvaluationDownstreamOLMo

def build_cpt_models(midtrained_models: ArtifactSet) -> ArtifactSet:
    return midtrained_models.map_flatten(
        lambda midtrained_model: ArtifactSet.from_product(
            cls=CPTModel,
            params=dict(
                train_tokens=[20],  # in Million
                midtrained_model=midtrained_model,
                cpt_dataset=["starcoder", "musicpile"],
                optimizer='adamw',
                learning_rate=[1e-5, 2e-5, 4e-5, 8e-5],
                weight_decay=[0],
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                cpt_gpus=2
            )
        )
    )

def build_cpt_model_evaluations(cpt_models: ArtifactSet) -> ArtifactSet:
    base_eval = cpt_models.map(lambda model: ModelEvaluationDownstreamOLMo(model=model))
    cpt_eval = cpt_models.map(lambda model: ModelEvaluation(model=model))
    return base_eval + cpt_eval