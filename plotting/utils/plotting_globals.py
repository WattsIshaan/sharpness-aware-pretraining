
FONTSIZE = {
    "TICKS": 14,
    "TITLE": 16,
    "AXIS": 14,
    "LEGEND": 14,
}

# FIG_WIDTH = 6.75
FIG_WIDTH = 5.5
FIG_HEIGHT = 2.5
MARKERS = ["o", "v", "s", "p"]
ALPHA = 0.9
GRID_ALPHA = 0.3
OPTIM_MAP = {
    "adamw": "AdamW",
    "sam": "SAM"
}

CPT_DATASET_MAP = {
    "starcoder": "StarCoder",
    "musicpile": "MusicPile",
    "tulu": "Tulu",
    "gsm8k": "GSM8K",
    "siqa": "SIQA",
    "stackmathqa": "StackMathQA",
}
DOWNSTREAM_TASK_MAP = {
    "winogrande": "Winogrande",
    "mmlu": "MMLU",
    "sciq": "SciQ",
    "hellaswag": "HellaSwag",
    "copa": "COPA",
    "openbook_qa": "OpenBookQA",
}

XLABEL = {
    "TOKEN_RATIO": "Tokens / Param",
    "RELATIVE_FLOPS": "Relative FLOPs",
    "PT_LOSS": "Pretrain Loss",
    "PT_LOSS_BASE": "Pretrain Loss (Base)",
    "ANNEAL_PERCENT": "Anneal Percent",
    "LR": "Learning Rate",
}
YLABEL = {
    "PT_LOSS": "Pretrain Loss",
    "FT_LOSS": "Fine-tuning Loss",
    "PT_LOSS_FT": "Pretrain Loss\n(Fine-tuned)",
    "PT_LOSS_PERTURBED": "Pretrain Loss\n(Perturbed)",
    "PT_LOSS_QUANTIZED": "Pretrain Loss\n(Quantized)",
}

LEGEND_PARAM = {
    "LOC": "lower center",
    "BBOX_TO_ANCHOR": (0.5, -0.15),
    "BBOX_TO_ANCHOR_QUANTIZATION": (0.5, -0.20),
}

COLOR_MAP = {
    "adamw": "royalblue",
    "sam": "darkorange",
    "wsd_sam": "violet",
    "wsd_adamw": "forestgreen",
    "cosine_adamw": "royalblue",
    "cosine_sam": "darkorange",
    "cosine": "royalblue",
    1e-4: "grey",
    3e-4: "royalblue",
    6e-4: "forestgreen",
    "percent5": "grey",
    "percent10": "forestgreen",
    "percent20": "violet",
}

LRS_MAP = {
    "wsd_sam": "SAWD",
    "wsd_adamw": "WSD",
    "cosine": "Cosine",
    "cosine_adamw": "Cosine (AdamW)",
    "cosine_sam": "Cosine (SAM)",
}

# ---------------------------------------------------------------------------
# Midtrain – Base Model eval task display names
# ---------------------------------------------------------------------------
BASE_EVAL_TASK_MAP = {
    "arc_challenge": "ARC-C",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "winogrande": "Winogrande",
    "drop": "DROP",
    "naturalqs_open": "NQ Open",
    "agi_eval_english": "AGIEval",
    "gsm8k": "GSM8K",
    "mmlu_pro": "MMLU-Pro",
    "triviaqa": "TriviaQA",
}

# ---------------------------------------------------------------------------
# Midtrain – SFT Model eval task display names
# ---------------------------------------------------------------------------
SFT_EVAL_TASK_MAP = {
    "mmlu": "MMLU",
    "popqa": "PopQA",
    "truthfulqa": "TruthfulQA",
    "bbh": "BBH",
    "minerva_math": "Minerva Math",
    "drop": "DROP",
    "gsm8k": "GSM8K",
    "codex_humaneval": "HumanEval",
    "codex_humanevalplus": "HumanEval+",
    "ifeval": "IFEval",
}

# ---------------------------------------------------------------------------
# Midtrain – Model display names
# ---------------------------------------------------------------------------
MIDTRAIN_MODEL_MAP = {
    "olmo_base": "OLMo Base",
    "adamw": "AdamW",
    "sam": "SAM",
    "adamw_4bit": "AdamW 4-bit",
    "sam_4bit": "SAM 4-bit",
    "olmo_sft": "OLMo-SFT",
}
