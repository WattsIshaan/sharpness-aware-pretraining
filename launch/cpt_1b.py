"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import CPTModel, ModelEvaluation, ModelEvaluationDownstreamOLMo

CPT_DATASET_INFO = {
    "alpaca": {
        "cpt_lr": [5e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4],
        "cpt_token": [20],
    },
    "codealpaca": {
        "cpt_lr": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
        "cpt_token": [20],
    },
    "gsm8k": {
        "cpt_lr": [5e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4],
        "cpt_token": [20],
    },
    "musicpile": {
        "cpt_lr": [2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4],
        "cpt_token": [50],
    },
    "siqa": {
        "cpt_lr": [3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4],
        "cpt_token": [20],
    },
    "stackmathqa": {
        "cpt_lr": [1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4],
        "cpt_token": [50],
    },
    "tulu": {
        "cpt_lr": [2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5],
        "cpt_token": [50],
    },
}

def build_cpt_models(midtrained_models: ArtifactSet) -> ArtifactSet:
    # for i, dataset in enumerate(["alpaca", "codealpaca", "gsm8k", "siqa"]):
    # for i, dataset in enumerate(["stackmathqa", "musicpile"]):
    for i, dataset in enumerate(["tulu"]):
        if i == 0:
            cpt_models = midtrained_models.map_flatten(
                lambda midtrained_model: ArtifactSet.from_product(
                    cls=CPTModel,
                    params=dict(
                        train_tokens=CPT_DATASET_INFO[dataset]["cpt_token"],
                        pretrained_model=midtrained_model,
                        cpt_dataset=dataset,
                        optimizer='adamw',
                        learning_rate=CPT_DATASET_INFO[dataset]["cpt_lr"],
                        weight_decay=0,
                        batch_size=64,
                        scheduler_name='cosine_with_warmup',
                        scheduler_alpha_f=0.1,
                        cpt_gpus=2,
                        use_checkpoint_cache=True,
                    )
                )
            )
        else:
            cpt_models += midtrained_models.map_flatten(
                lambda midtrained_model: ArtifactSet.from_product(
                    cls=CPTModel,
                    params=dict(
                    train_tokens=CPT_DATASET_INFO[dataset]["cpt_token"],
                    pretrained_model=midtrained_model,
                    cpt_dataset=dataset,
                    optimizer='adamw',
                    learning_rate=CPT_DATASET_INFO[dataset]["cpt_lr"],
                    weight_decay=0,
                    batch_size=64,
                    scheduler_name='cosine_with_warmup',
                    scheduler_alpha_f=0.1,
                    cpt_gpus=2,
                    use_checkpoint_cache=True,
                    )
                )
            )

    return cpt_models


def build_cpt_model_evaluations(cpt_models: ArtifactSet) -> ArtifactSet:
    base_eval = cpt_models.map(lambda model: ModelEvaluationDownstreamOLMo(model=model))
    cpt_eval = cpt_models.map(lambda model: ModelEvaluation(model=model))
    return base_eval + cpt_eval