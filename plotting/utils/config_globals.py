SIZE = [60, 150]
CPT_DATASET = ["starcoder", "musicpile", "tulu", "gsm8k", "siqa", 'stackmathqa'] #"alpaca", 'open-platypus', , 'helpsteer'
OPTIM = ['adamw', 'sam']
RHO = [5e-2]
VERBOSE = False

RESULTS_DIR = "/home/iwatts/catastrophic-forgetting/results/"

ANNEAL_CONFIG = {
    "anneal_optim": ["adamw", "sam"],
    "steps" : [1000, 2000, 4000],
    "percent" : [5, 10, 20]
}

LRS = ["wsd_adamw", "wsd_sam", "cosine"]
ALL_LRS = ["wsd_adamw", "wsd_sam", "cosine_adamw", "cosine_sam"]


TASKNAME_MAP = {
    # "c4_val": "eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000",
    # "starcoder_val": "preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001",
    # "tinygsm_val": "preprocessed_tinyGSM_mind_dolma2-tokenizer_part-01-00000-2M",
    # "starcoder_val": "preprocessed_starcoder_v1-decon-100_to_20k-2star-top_token_030_allenai_dolma2-tokenizer_part-010-00000-20M",
    "starcoder_val": "input_ids-starcoder",
    # "starcoder2_val": "preprocessed_starcoder_v1-decon-100_to_20k-2star-top_token_030_allenai_dolma2-tokenizer_part-010-00000-20M",
    # "starcoder3_val": "preprocessed_starcoder_v1-decon-100_to_20k-2star-top_token_030_allenai_dolma2-tokenizer_part-010-00000-20M",
    # "dclm_val": "preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-187-00004-20M",
    "dclm_val": "dclm-20m",
    "dclm_train": "dclm-train-20m",
    "musicpile_val": "input_ids-musicpile",
    "tulu_val": "input_ids-tulu",
    "gsm8k_val": "input_ids-gsm8k",
    "alpaca_val": "input_ids-alpaca",
    "siqa_val": "input_ids-siqa",
    "open-platypus_val": "input_ids-open-platypus",
    "stackmathqa_val": "input_ids-stackmathqa",
    "helpsteer_val": "input_ids-helpsteer",
    # "c4_val": "eval-data_perplexity_v3_small_dolma2-tokenizer_c4_en_val_part-0-00000",
}

PERTURBATIONS = [0.009, 0.013, 0.017, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1]
BITS = [4, 8]



TOKEN_LIST = {
    20: [4, 8, 16, 32, 64],
    60: [12, 24, 48, 96, 192],
    150: [15, 30, 60, 120]
}

PT_LR = {
    20 : {
        4: 3e-3,
        8: 3e-3,
        16: 3e-3,
        32: 1e-3,
        64: 1e-3,
    },
    60: {
        12: 1e-3,
        24: 1e-3,
        48: 6e-4,
        96: 6e-4,
        192: 3e-4,
    },
    150: {
        15: 1e-3,
        30: 1e-3,
        60: 6e-4,
        120: 3e-4,
    },
}

CHECKPOINT_MAP = {
    "compute": {
        20: {
            15000: 4,
            25000: 8,
            50000: 16,
            100000: 32,
            205000: 64,
        },
        60: {
            35000: 12,
            75000: 24,
            155000: 48,
            305000: 96,
            610000: 192,
        },
        150: {
            45000: 15,
            95000: 30,
            190000: 60,
            380000: 120,
        },
    },
    "token": {
        20: {
            15000: 4,
            30000: 8,
            55000: 16,
            110000: 32,
            220000: 64,
        },
        60: {
            40000: 12,
            85000: 24,
            165000: 48,
            335000: 96,
            670000: 192,
        },
        150: {
            50000: 15,
            105000: 30,
            205000: 60,
            415000: 120,
        },
    }
}






