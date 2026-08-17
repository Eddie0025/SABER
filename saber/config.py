import os

# --- Model & Hardware Config ---
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_HARDWARE = "H100"  # Assumes single 80GB H100

# --- Inference Settings ---
INFERENCE_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# --- Specialist Adapter Registry ---
# Maps domain names to their trained GRPO adapter paths.
# Falls back to _grpo if _grpo_final doesn't exist.
SPECIALIST_REGISTRY = {
    "cybersecurity":        "models/cybersecurity_grpo_final",
    "python":               "models/python_grpo_final",
    "javascript":           "models/javascript_grpo_final",
    "sql":                  "models/sql_grpo_final",
    "finance":              "models/finance_grpo_final",
    "medical":              "models/medical_grpo_final",
    "science":              "models/science_grpo_final",
    "architecture_qa":      "models/architecture_qa_grpo_final",
    "architecture_planner": "models/architecture_planner_grpo_final",
}

# Domains that should get higher token limits (long-form answers)
LONG_FORM_SPECIALISTS = ["medical", "science", "architecture_planner"]

# All valid domain labels for the intent classifier
ALL_DOMAINS = list(SPECIALIST_REGISTRY.keys()) + ["coding", "casual_chat"]

# --- Domain System Prompts (Domain-Agnostic Pattern) ---
DOMAIN_SYSTEM_PROMPTS = {
    "cybersecurity":        "You are a cybersecurity AI specialist.",
    "python":               "You are a Python programming AI specialist.",
    "javascript":           "You are a JavaScript programming AI specialist.",
    "sql":                  "You are a SQL and database AI specialist.",
    "finance":              "You are a finance and quantitative analysis AI specialist.",
    "medical":              "You are a medical and clinical reasoning AI specialist.",
    "science":              "You are a science AI specialist covering physics, chemistry, biology, and mathematics.",
    "architecture_qa":      "You are a software architecture and distributed systems AI specialist.",
    "architecture_planner": "You are a system design and architecture planning AI specialist.",
    "coding":               "You are a software engineering AI specialist.",
}

# --- Tier 1: Casual Chat Fast-Path Patterns ---
CASUAL_PATTERNS = {
    "hello", "hi", "hey", "howdy", "hola", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "hows it going", "whats up", "how do you do",
    "thanks", "thank you", "thx", "ty", "cheers",
    "bye", "goodbye", "see you", "later", "cya",
    "ok", "okay", "cool", "nice", "great", "awesome", "lol", "haha",
    "who are you", "what are you", "whats your name",
}

# --- Sentinel Config ---
SENTINEL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
SENTINEL_EMBEDDING_MODEL_LOCAL = "models/bge-base-en-v1.5"
SENTINEL_RELEVANCE_THRESHOLD = 0.4
OFFLINE_KB_DIR = "data/offline_kb"

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
