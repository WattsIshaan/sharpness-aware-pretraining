"""EWC SFT artifacts: same SFT grid as ``launch/sft.py`` but using ``EWC_SFT`` / ``train_ewc.py``."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import EWC_SFT, ModelEvaluation

def build_ewc_sft_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    """One ``EWC_SFT`` per (pretrained model, SFT LR) product, matching ``build_sft_models`` hyperparameters."""
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=EWC_SFT,
            params=dict(
                train_tokens=[10],  # million SFT tokens; match ``launch/sft.py``
                pretrained_model=pretrained_model,
                sft_dataset='starcoder', # tulu, stackmathqa, musicpile, gsm8k
                optimizer="adamw",
                learning_rate=[3e-4, 4e-4, 5e-4, 6e-4, 8e-4],
                weight_decay=[0],
                batch_size=64,
                scheduler_name="cosine_with_warmup",
                scheduler_alpha_f=0.1,
                sft_gpus=1,
                ewc_lambda=[1000.0, 4000.0, 10000.0, 40000.0, 100000.0],
                ewc_fisher_batches=[100],
                ewc_fisher_pretrain_subsample_tokens_billion=[1.0],
            ),
        )
    )


def build_ewc_sft_model_evaluations(ewc_sft_models: ArtifactSet) -> ArtifactSet:
    return ewc_sft_models.map(lambda model: ModelEvaluation(model=model))
