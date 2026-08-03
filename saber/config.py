import os

# --- Model & Hardware Config ---
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_HARDWARE = "H100"  # Assumes single 80GB H100

# --- Training Configurations ---
# Early stopping is enforced via validate_dora.py checks on train loss vs eval accuracy.
CHECKPOINT_FREQ_STANDARD = 1.0  # Epochs
CHECKPOINT_FREQ_HIGH_STAKES = 0.5  # Epochs (for Planner & Meta-Reasoner)

# DoRA Target Module Presets
TARGET_MODULE_PRESETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mlp_only": ["gate_proj", "up_proj", "down_proj"],
    "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

# Default LoRA/DoRA hyperparameters
DORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_dora": True,
    "task_type": "CAUSAL_LM"
}
