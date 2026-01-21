SIZE = [20, 60, 150]
CPT_DATASET = ["starcoder", "musicpile", "tulu"]
OPTIM = ['adamw', 'sam']
RHO = [5e-2]
VERBOSE = False

TRADEOFF_THRESHOLD=0.03

RESULTS_DIR = "/home/iwatts/catastrophic-forgetting/results/"

ANNEAL_CONFIG = {
    "anneal_optim": ["adamw", "sam"],
    "steps" : [1000, 2000, 4000],
    "percent" : [5, 10, 20]
}


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
        120: 6e-4,
    },
}

CHECKPOINT_MAP = {
    20 : {
        15000: 4,
        30000: 8,
        55000: 16,
        110000: 32,
        220000: 64,
        # 445000: 128, 
        # 890000: 256 
    },
    60: {
        35000: 12,
        40000: 12,
        45000: 12,
        75000: 24,
        85000: 24,
        90000: 24,
        155000: 48,
        165000: 48,
        175000: 48,
        180000: 48,
        305000: 96,
        335000: 96,
        350000: 96, 
        365000: 96,
        610000: 192,
        670000: 192,
        700000: 192,
        730000: 192,
        # 1330000: 384
    },
    150: {
        57000: 15,
        114000: 30,
        228000: 60,
        457000: 120
    }
}






