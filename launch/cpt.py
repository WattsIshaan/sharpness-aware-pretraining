"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet
from launch.artifacts import CPTModel, ModelEvaluation
from launch.pretrain import sam_pretrained_models, sgd_pretrained_models

sam_cpt_models = sam_pretrained_models.map_flatten(
    lambda pretrained_model: ArtifactSet.from_product(
        cls=CPTModel,
        params=dict(
            train_tokens=[20], # in Million
            pretrained_model=pretrained_model,
            training_dataset_name='starcoder',
            optimizer='adamw',
            learning_rate=[1e-3, 2e-4, 4e-5],
            weight_decay=0.1,
            batch_size=64,
            scheduler_name='cosine_with_warmup',
            scheduler_alpha_f=0.1,
        )
    )
)

sgd_cpt_models = sgd_pretrained_models.map_flatten(
    lambda pretrained_model: ArtifactSet.from_product(
        cls=CPTModel,
        params=dict(
            train_tokens=[20], # in Million
            pretrained_model=pretrained_model,
            training_dataset_name='starcoder',
            optimizer='adamw',
            learning_rate=[1e-3, 2e-4, 4e-5],
            weight_decay=0.1,
            batch_size=64,
            scheduler_name='cosine_with_warmup',
            scheduler_alpha_f=0.1,
        )
    )
)

# Map model evaluations for each model
sam_cpt_model_evaluations = sam_cpt_models.map(lambda model: ModelEvaluation(model=model))
sgd_cpt_model_evaluations = sgd_cpt_models.map(lambda model: ModelEvaluation(model=model))