"""Pre-training artifacts for the Wikipedia pretraining experiment."""

import math

from experiments import ArtifactSet
from launch.artifacts import PretrainedModel, ModelEvaluation

sam_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sam'],
            'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],
            'train_tokens': [1, 4],
            'weight_decay': [0.02],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'sam_rho': [0.05, 0.1, 0.2],
            # 'scheduler_t_warmup': [100],
            'scheduler_alpha_f': [0.1],
        }
    )

sgd_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sgd'],
            'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],
            'train_tokens': [1, 4],
            'weight_decay': [0.02, 0.005],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            # 'scheduler_t_warmup': [100],
            'scheduler_alpha_f': [0.1],
        }
    )

# Map model evaluations for each model
sam_model_evaluations = sam_pretrained_models.map(lambda model: ModelEvaluation(model=model))
sgd_model_evaluations = sgd_pretrained_models.map(lambda model: ModelEvaluation(model=model))