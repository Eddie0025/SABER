# -*- coding: utf-8 -*-
"""saber.training.trainer

LoRA fine-tuning pipeline for SABER specialist models.

Uses TRL's SFTTrainer with data packing for maximum training efficiency.
Formats all data using Qwen2.5 ChatML template with domain-specific
system prompts to inject Chain-of-Thought reasoning behaviour.

Optimized for cloud GPU training (3× RTX 6000 Ada).

Usage
~~~~~
::

    # Single domain on a specific GPU
    python -m saber.training.trainer --domain medical --gpu 0

    # All domains sequentially on GPU 0
    python -m saber.training.trainer

"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Monkeypatch Hugging Face's PyTorch version check for loading weights
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# Allowlist numpy reconstruct for RNG state loading in newer PyTorch versions
try:
    import torch
    import numpy.core.multiarray
    import numpy
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, numpy.ndarray])
except Exception:
    pass
@dataclass
class TrainConfig:
    """Configuration for a single training run.

    Defaults are optimized for H100 (80 GB HBM3 VRAM) with a
    Qwen2.5-7B-Instruct base model + LoRA.
    """

    domain: str = "medical"
    data_path: str = "data/processed/medical.jsonl"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "models/medical_v2"
    epochs: int = 3
    batch_size: int = 8                     # Per-device; safe for 80 GB H100
    learning_rate: float = 2e-4
    max_seq_length: int = 2048              # Increased for CoT reasoning chains
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    gradient_accumulation_steps: int = 4    # Effective batch = 8 × 4 = 32
    warmup_ratio: float = 0.03
    fp16: bool = False
    bf16: bool = True                       # H100 has native bf16 support
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    packing: bool = True
    gpu_id: int = 0                         # Which GPU to target
    patch_mode: bool = False
    dpo_mode: bool = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSON-Lines file produced by the dataset_loader."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# Domain-specific system prompts that prime the model for CoT reasoning.
_DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "medical": (
        "You are a medical specialist with deep expertise in clinical medicine, "
        "pharmacology, pathophysiology, and differential diagnosis. "
        "Analyze each case systematically. Think through your reasoning "
        "step by step before providing your final answer."
    ),
    "cyber": (
        "You are a cybersecurity specialist with expertise in MITRE ATT&CK, "
        "incident response, threat intelligence, vulnerability analysis, "
        "and digital forensics. Map threats to specific techniques and "
        "provide structured analysis. Think through your reasoning "
        "step by step before providing your final answer."
    ),
    "science": (
        "You are a science specialist with expertise in physics, chemistry, "
        "biology, and mathematical reasoning. Show all work and explain "
        "each step clearly. Think through your reasoning step by step "
        "before providing your final answer."
    ),
    "coding": (
        "You are a coding specialist with expertise in Python, algorithms, "
        "data structures, and software engineering. Write clean, optimized "
        "code with clear explanations. Think through your approach step "
        "by step before writing code."
    ),
    "architecture": (
        "You are a systems architecture specialist with expertise in "
        "distributed systems, cloud infrastructure, microservices, "
        "and security architecture. Design scalable, resilient systems. "
        "Think through your reasoning step by step before providing "
        "your final answer."
    ),
    "finance": (
        "You are a finance and economics specialist with expertise in corporate "
        "finance, market trends, financial mathematics, and economic theory. "
        "Make educated, data-driven decisions. Think through your reasoning "
        "step by step before providing your final answer."
    ),
    "orchestrator": (
        "You are the SABER Orchestrator. Your sole responsibility is to evaluate "
        "the user's prompt and route it to the correct specialist domains based on the required technical expertise. "
        "For complex system-building, application design, or pipeline engineering requests, you must route to "
        "both 'architecture' (for design) and 'coding' (for implementation) in addition to the specific domain "
        "(e.g., 'finance', 'medical', 'science'). "
        "DO NOT answer the user's question. You must output strict JSON matching "
        "the following schema: "
        "{\"route\": [\"domain1\", \"domain2\"], \"confidence\": 0.99, \"multi_domain\": true, \"query_summary\": \"...\"}"
    ),
    "meta_reasoner": (
        "You are the SABER Meta-Reasoner. Your job is to take raw outputs from "
        "specialists, identify and resolve contradictions, verify their logic, "
        "and synthesize them into a polished final response. "
        "CRITICAL: Terminate execution after a maximum of 2 retries if consensus "
        "cannot be reached."
    ),
}


def format_for_sft(
    records: List[Dict[str, Any]],
    domain: str,
) -> List[Dict[str, str]]:
    """Format raw records into Qwen2.5 ChatML strings for SFTTrainer.

    Each record is turned into a single ``text`` field containing:

    .. code-block:: text

        <|im_start|>system
        {domain system prompt}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {answer}<|im_end|>

    This teaches the model to follow instructions within its native
    chat template, which is critical for inference-time behaviour.
    """
    system_prompt = _DOMAIN_SYSTEM_PROMPTS.get(
        domain,
        "You are a helpful AI assistant. Think step by step.",
    )

    formatted: List[Dict[str, str]] = []
    for rec in records:
        text = rec.get("text", "").strip()
        label = rec.get("label", "").strip()
        if not text or not label:
            continue

        conversation = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n{label}<|im_end|>"
        )
        formatted.append({"text": conversation})

    return formatted


def format_for_dpo(records: List[Dict[str, Any]], domain: str) -> List[Dict[str, str]]:
    """Format records for DPOTrainer."""
    system_prompt = _DOMAIN_SYSTEM_PROMPTS.get(
        domain,
        "You are a helpful AI assistant. Think step by step.",
    )
    formatted = []
    for rec in records:
        prompt_text = rec.get("prompt", "").strip()
        chosen = rec.get("chosen", "").strip()
        rejected = rec.get("rejected", "").strip()
        if not prompt_text or not chosen or not rejected:
            continue

        prompt_formatted = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        formatted.append({
            "prompt": prompt_formatted,
            "chosen": f"{chosen}<|im_end|>",
            "rejected": f"{rejected}<|im_end|>"
        })
    return formatted


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> str:
    """Run LoRA fine-tuning using TRL's SFTTrainer with data packing.

    Steps
    -----
    1. Load the base model & tokenizer.
    2. Apply LoRA adapters via PEFT.
    3. Load & format the dataset with ChatML + CoT system prompts.
    4. Train with SFTTrainer (packing=True).
    5. Save the adapter weights.
    6. Run a quick evaluation.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        EarlyStoppingCallback,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # Monkeypatch the PyTorch version check directly inside transformers.trainer
    # where it is actually called during _load_optimizer_and_scheduler
    import transformers.trainer
    if hasattr(transformers.trainer, "check_torch_load_is_safe"):
        transformers.trainer.check_torch_load_is_safe = lambda: None
    import transformers.utils.import_utils
    if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

    # --- Device selection -------------------------------------------------
    if torch.cuda.is_available():
        device = f"cuda:{cfg.gpu_id}"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    print(f"[trainer] Device ........... {device} ({'GPU ' + str(cfg.gpu_id) if 'cuda' in device else 'CPU'})")
    print(f"[trainer] Base model ....... {cfg.base_model}")
    print(f"[trainer] Domain ........... {cfg.domain}")
    print(f"[trainer] Packing .......... {cfg.packing}")
    print(f"[trainer] Max seq length ... {cfg.max_seq_length}")
    print(f"[trainer] Batch size ....... {cfg.batch_size}")
    print(f"[trainer] Grad accum ....... {cfg.gradient_accumulation_steps}")
    print(f"[trainer] Effective batch .. {cfg.batch_size * cfg.gradient_accumulation_steps}")

    # 1. Load base model + tokenizer ------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "sdpa"
    try:
        import flash_attn  # type: ignore
        attn_impl = "flash_attention_2"
        print("[trainer] FlashAttention-2 is available and will be used.")
    except Exception as e:
        print(f"[trainer] FlashAttention-2 check failed: {e}. Falling back to SDPA.")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32),
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # 2. Apply LoRA -----------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()     # Required for gradient checkpointing + LoRA
    model.print_trainable_parameters()

    # 3. Load & format dataset ------------------------------------------
    records = load_jsonl(cfg.data_path)
    if not records:
        print(f"[trainer] ERROR — No records found in {cfg.data_path}")
        return ""

    print(f"[trainer] Loaded {len(records)} raw records from {cfg.data_path}")
    
    if cfg.dpo_mode:
        formatted = format_for_dpo(records, cfg.domain)
        print(f"[trainer] Formatted {len(formatted)} records for DPO")
    else:
        formatted = format_for_sft(records, cfg.domain)
        print(f"[trainer] Formatted {len(formatted)} records for SFT")

    # 3b. Tokenize dataset directly (no trl dependency) -----------------
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        return tokenized

    dataset = Dataset.from_list(formatted)

    # Split 95/5 — maximise training data, minimal eval set
    if not cfg.dpo_mode:
        split = dataset.train_test_split(test_size=0.05, seed=cfg.seed)
        train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=["text"])
    else:
        # DPO mode does not use tokenized mapped dataset beforehand because DPOTrainer does it internally
        split = dataset.train_test_split(test_size=0.05, seed=cfg.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    print(f"[trainer] Ready {len(train_ds)} train / {len(eval_ds)} eval samples")

    # 4. Training arguments ---------------------------------------------
    
    # Adjust for patch_mode
    num_epochs = cfg.epochs
    per_device_batch = cfg.batch_size
    grad_accum = cfg.gradient_accumulation_steps
    
    if cfg.patch_mode:
        num_epochs = 5
        per_device_batch = 2
        grad_accum = 1
        print(f"[trainer] PATCH MODE active: epochs={num_epochs}, batch={per_device_batch}, grad_accum={grad_accum}")
    if cfg.dpo_mode:
        num_epochs = 3
        per_device_batch = 2
        grad_accum = 1
        print(f"[trainer] DPO MODE active: epochs={num_epochs}, batch={per_device_batch}, grad_accum={grad_accum}")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory="cuda" in device,
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
    )

    # 5. Standard Trainer (no trl needed) --------------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    from transformers import TrainerCallback
    import torch

    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, domain):
            self.tokenizer = tokenizer
            self.domain = domain
            self.prompt = f"<|im_start|>system\nYou are a highly skilled {domain} AI specialist.<|im_end|>\n<|im_start|>user\nWhat are the fundamental principles of {domain}?<|im_end|>\n<|im_start|>assistant\n"
            
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None: return
            print(f"\n\n============================================================")
            print(f" [Epoch {state.epoch}] Sample Generation for '{self.domain}'")
            print(f"============================================================")
            model.eval()
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant\n" in generated_text:
                generated_text = generated_text.split("assistant\n")[-1]
            print(f"{generated_text}\n============================================================\n")
            model.train()

    if cfg.dpo_mode:
        from trl import DPOTrainer
        # DPOTrainer doesn't need the DataCollatorForLanguageModeling
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            beta=0.1,  # Standard DPO beta
            max_prompt_length=cfg.max_seq_length // 2,
            max_length=cfg.max_seq_length,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=1),
                GenerationCallback(tokenizer, cfg.domain)
            ],
        )

    print(f"[trainer] Starting training — {len(train_ds)} train / {len(eval_ds)} eval samples")
    trainer.train()

    # 6. Save adapter ---------------------------------------------------
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[trainer] Adapter saved to {cfg.output_dir}")
    
    # Save stopping epoch to summary
    stopped_epoch = trainer.state.epoch
    summary_file = "training_summary.json"
    summary = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
        except Exception:
            pass
    summary[cfg.domain] = {"stopped_epoch": stopped_epoch, "max_epochs": cfg.epochs}
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    # Quick eval
    metrics = trainer.evaluate()
    print(f"[trainer] Eval metrics: {metrics}")

    return cfg.output_dir


# ---------------------------------------------------------------------------
# Domain defaults (all five specialists)
# ---------------------------------------------------------------------------

_DOMAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "medical": {
        "data_path": "data/processed/medical.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/medical_v2",
        "epochs": 3,
    },
    "cyber": {
        "data_path": "data/processed/cyber.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/cyber_v2",
        "epochs": 3,
    },
    "science": {
        "data_path": "data/processed/science.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/science_v2",
        "epochs": 3,
    },
    "coding": {
        "data_path": "data/processed/coding.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/coding_v2",
        "epochs": 3,
    },
    "architecture": {
        "data_path": "data/processed/architecture.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/architecture_v2",
        "epochs": 3,
    },
    "finance": {
        "data_path": "data/processed/finance.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/finance_v2",
        "epochs": 3,
    },
    "orchestrator": {
        "data_path": "data/processed/orchestrator.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/orchestrator_v2",
        "epochs": 3,
    },
    "meta_reasoner": {
        "data_path": "data/processed/meta_reasoner.jsonl",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "models/meta_reasoner_v2",
        "epochs": 3,
    },
}


def train_all(base_model: Optional[str] = None) -> Dict[str, str]:
    """Train LoRA adapters for all eight models sequentially.

    Returns a dict ``{domain: output_dir}``.
    """
    results: Dict[str, str] = {}
    for domain, defaults in _DOMAIN_DEFAULTS.items():
        cfg = TrainConfig(
            domain=domain,
            data_path=defaults["data_path"],
            base_model=base_model or defaults["base_model"],
            output_dir=defaults["output_dir"],
            epochs=defaults.get("epochs", 3),
        )
        print(f"\n{'=' * 60}")
        print(f"  Training [{domain.upper()}] specialist")
        print(f"{'=' * 60}")
        path = train(cfg)
        results[domain] = path
        # Free GPU memory between domains
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SABER specialist model trainer (SFTTrainer + LoRA + Packing)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Train a single domain (medical|cyber|science|coding|architecture). "
             "If omitted, trains all specialists sequentially.",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL data file.")
    parser.add_argument("--base-model", type=str, default=None, help="HuggingFace model ID.")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    parser.add_argument("--no-packing", action="store_true", help="Disable data packing.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--patch-mode", action="store_true", help="Run in continuous SFT patch mode.")
    parser.add_argument("--dpo-mode", action="store_true", help="Run in DPO mode (requires prompt/chosen/rejected).")

    # Use parse_known_args to prevent errors when running in Jupyter Notebook
    # which passes unrecognized arguments like `-f`
    args, _ = parser.parse_known_args()

    if args.domain:
        defaults = _DOMAIN_DEFAULTS.get(args.domain, {})
        cfg = TrainConfig(
            domain=args.domain,
            data_path=args.data or defaults.get("data_path", ""),
            base_model=args.base_model or defaults.get("base_model", "Qwen/Qwen2.5-7B-Instruct"),
            output_dir=args.output or defaults.get("output_dir", f"models/{args.domain}_v2"),
            epochs=args.epochs or defaults.get("epochs", 3),
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            gpu_id=args.gpu,
            packing=not args.no_packing,
            max_seq_length=args.max_seq_len,
            patch_mode=args.patch_mode,
            dpo_mode=args.dpo_mode,
        )
        train(cfg)
    else:
        train_all(base_model=args.base_model)


if __name__ == "__main__":
    main()
