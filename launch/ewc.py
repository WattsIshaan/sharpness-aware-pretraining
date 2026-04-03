"""EWC CPT artifacts: same CPT grid as ``launch/cpt.py`` but using ``EWC_CPT`` / ``train_ewc.py``."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import EWC_CPT, ModelEvaluation

# Reuse LR tables and dataset selection from cpt.py
from launch.cpt import CPT_DATASET, CPT_LR, LRS


def build_ewc_cpt_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    """One ``EWC_CPT`` per (pretrained model, CPT LR) product, matching ``build_cpt_models`` hyperparameters."""
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=EWC_CPT,
            params=dict(
                train_tokens=[10],  # million CPT tokens; match ``launch/cpt.py``
                pretrained_model=pretrained_model,
                cpt_dataset=CPT_DATASET,
                optimizer="adamw",
                learning_rate=CPT_LR[LRS][pretrained_model.model_size][pretrained_model.train_tokens][CPT_DATASET],
                weight_decay=[0],
                batch_size=64,
                scheduler_name="cosine_with_warmup",
                scheduler_alpha_f=0.1,
                cpt_gpus=1,
                ewc_lambda=[4000.0],
                ewc_fisher_batches=[100],
                ewc_fisher_pretrain_subsample_tokens_billion=[1.0],
            ),
        )
    )


def build_ewc_cpt_model_evaluations(ewc_cpt_models: ArtifactSet) -> ArtifactSet:
    return ewc_cpt_models.map(lambda model: ModelEvaluation(model=model))
