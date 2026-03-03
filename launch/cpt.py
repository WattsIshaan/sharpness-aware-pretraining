"""CPT artifacts for the pretraining experiments."""

from experiments import ArtifactSet  # type: ignore
from launch.artifacts import CPTModel, ModelEvaluation


CPT_LR = {
    'cosine': {
        '60m': {
            192: {
                # 'starcoder': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3],
                # 'starcoder': [2.1e-4, 2.25e-4, 2.3e-4, 2.4e-4, 2.5e-4, 2.6e-4, 2.75e-4],
                'starcoder': [4e-6, 8e-6, 1e-4]
            },
        }
    },
    'wsd': {
        '150m': {
           31: {
            'starcoder': [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3],
            'musicpile': [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3],
            'tulu': [2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4],
            'stackmathqa': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 2e-3],
            'gsm8k': [1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4],
            'siqa': [4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4], # [101742]
           },
           61: {
            'starcoder': [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4],
            'musicpile': [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3],
            'tulu': [2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4],
            'stackmathqa': [4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 8e-4, 1e-3], # [101664]
            'gsm8k': [1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4], # [101671]
            'siqa': [1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4], # [101668]
           },
           121: {
            'starcoder': [8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4], # [101677]
            'musicpile': [8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4, 6e-4], # [101678]
            'tulu': [2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4,], # [101681]
            'stackmathqa': [4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 8e-4], # [101685]
            'gsm8k': [1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4], # 101694
            'siqa': [8e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1.5e-4, 2e-4], # [101697]
           },
        },
        '60m': {
            25: {
                'starcoder': [2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1.5e-3, 2e-3], # 101997
                'musicpile': [4e-4, 6e-4, 8e-4, 1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 3e-3], # 101998
                'tulu': [1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4], # 102001
                'stackmathqa': [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3], # 102003
                'siqa': [8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4, 6e-4], # 102012
                'gsm8k': [8e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3], # 102006
            },
            100: {
                'starcoder': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 1.5e-3, 2e-3], # 102035
                'musicpile': [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3], # 102028
                'tulu': [4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4], # 102025
                'stackmathqa': [8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3], # 102018
                'siqa': [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4], # 102015
                'gsm8k': [2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 3e-4, 4e-4, 8e-4], # 102018
            },
            200: {
                'starcoder': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3], # 102044, 102049
                'musicpile': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3], # 102052
                'tulu': [1e-6, 2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4], # 102055
                'stackmathqa': [4e-5, 8e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 9e-4], # 102058
                'siqa': [1.5e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1.5e-4, 2e-4], # 102064
                'gsm8k': [1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1.5e-4, 2e-4] # 102062
            }
        }
    }
}

CPT_DATASET='starcoder'
# CPT_DATASET='musicpile'
# CPT_DATASET='tulu'
# CPT_DATASET='stackmathqa'
# CPT_DATASET='gsm8k'
# CPT_DATASET='siqa'
LRS='cosine'
print(CPT_DATASET)

def build_cpt_models(pretrained_models: ArtifactSet) -> ArtifactSet:
    return pretrained_models.map_flatten(
        lambda pretrained_model: ArtifactSet.from_product(
            cls=CPTModel,
            params=dict(
                train_tokens=[5, 20],  # in Million
                pretrained_model=pretrained_model,
                cpt_dataset=CPT_DATASET,
                optimizer='adamw',
                learning_rate=CPT_LR[LRS][pretrained_model.model_size][pretrained_model.train_tokens][CPT_DATASET],
                # learning_rate=[8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3], # musicpile, starcoder
                # learning_rate=[8e-5, 1e-4, 1.5e-4, 1.75e-4, 2e-4, 2.5e-4, 2.75e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4], # musicpile, starcoder
                # learning_rate=[8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 6e-4], # musicpile, starcoder
                # learning_rate=[1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4.5e-4, 5e-4], # gsm8k
                # learning_rate=[1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3], # stackmathqa
                # learning_rate=[1e-6, 2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5], # siqa
                # learning_rate=[1e-6, 2e-6, 4e-6], # 8e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 8e-5, 1e-4], # tulu
                weight_decay=[0],
                batch_size=64,
                scheduler_name='cosine_with_warmup',
                scheduler_alpha_f=0.1,
                cpt_gpus=1
            )
        )
    )


def build_cpt_model_evaluations(cpt_models: ArtifactSet) -> ArtifactSet:
    return cpt_models.map(lambda model: ModelEvaluation(model=model))