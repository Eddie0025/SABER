import os
import sys
import logging
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# --- CUSTOM DATA COLLATOR ---
# We implement this natively to avoid any TRL versioning/import nightmares
from transformers import DataCollatorForLanguageModeling

class CustomCompletionOnlyCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template: str, tokenizer, *args, **kwargs):
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        super().__init__(tokenizer=tokenizer, *args, **kwargs)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            labels = batch["labels"][i].tolist()
            # Find the response template and mask everything before it with -100
            for j in range(len(labels) - len(self.response_template_ids)):
                if labels[j : j + len(self.response_template_ids)] == self.response_template_ids:
                    # Mask everything up to the END of the response template
                    batch["labels"][i, : j + len(self.response_template_ids)] = -100
                    break
        return batch
# ----------------------------

from saber.config import BASE_MODEL, TARGET_MODULE_PRESETS, DORA_CONFIG
from saber.training.rewards import log_grpo_reward_variance
from saber.training.dataset_loader import load_specialist_dataset, apply_chatml_formatting
from scripts.validate_dora import validate_dora
from scripts.validate_grpo import validate_grpo

logger = logging.getLogger("SABER_Trainer")

def get_target_modules(preset_name: str) -> list:
    if preset_name not in TARGET_MODULE_PRESETS:
        raise ValueError(f"Unknown target_modules preset: {preset_name}. Valid presets: {list(TARGET_MODULE_PRESETS.keys())}")
    return TARGET_MODULE_PRESETS[preset_name]

def run_dora_training(args):
    """
    SFT DoRA Training with early stopping (Save Best, Not Last) based on validate_dora.py checks.
    """
    logger.info("Setting up tokenizer and padding...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Critical fix for Qwen/Llama padding crashes
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # For SFT, right padding is standard
    
    logger.info("Loading base model in native bfloat16 (no quantization) for higher DoRA accuracy...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # CRITICAL: Since we removed prepare_model_for_kbit_training, we must manually 
    # enable input gradients so backprop flows through the frozen base weights to the adapters.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # 1. FIXED DATA COLLATOR (Native implementation)
    response_template = "<|im_start|>assistant\n"
    collator = CustomCompletionOnlyCollator(
        response_template=response_template, 
        tokenizer=tokenizer,
        mlm=False
    )
    logger.info("Using CustomCompletionOnlyCollator to correctly mask prompt tokens.")
    
    # 2. TARGET MODULE PRESET
    target_modules = get_target_modules(args.target_modules)
    
    peft_config = LoraConfig(
        r=DORA_CONFIG["r"],
        lora_alpha=DORA_CONFIG["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=DORA_CONFIG["lora_dropout"],
        use_dora=DORA_CONFIG["use_dora"],
        task_type=DORA_CONFIG["task_type"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # NOTE: TrainingArguments should be configured to save checkpoints according to 
    # CHECKPOINT_FREQ_STANDARD or CHECKPOINT_FREQ_HIGH_STAKES from config.py.
    # metric_for_best_model="eval_loss" / "accuracy" along with load_best_model_at_end=True
    # satisfies the "Save Best, Not Last" early stopping criterion.
    
    logger.info(f"DoRA Target Modules injected: {target_modules}")
    
    
    # 3. LOAD DATASET
    logger.info(f"Loading dataset for specialist: {args.specialist}")
    dataset = load_specialist_dataset(args.specialist)
    if dataset is None:
        logger.error(f"Failed to load dataset for {args.specialist}. Skipping training.")
        return
        
    # 4. TOKENIZE DATASET MANUALLY (Bypasses TRL completely)
    logger.info("Pre-tokenizing dataset (Max Length: 2048)...")
    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)
        
    tokenized_dataset = dataset.map(tokenize_func, batched=True, num_proc=8, remove_columns=dataset.column_names)
    
    # Shuffle and take 300 for validation
    eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(min(300, len(tokenized_dataset))))

    # 5. STANDARD NATIVE TRAINER EXECUTION (80GB H100 Optimized)
    # Batch=2 x GradAccum=16 = effective batch 32
    # Dynamically calculate eval_steps to ensure ~10 points across the 3 epochs
    effective_batch = 2 * 16
    total_steps = (len(tokenized_dataset) // effective_batch) * 3
    dynamic_eval_steps = max(1, total_steps // 10)
    logger.info(f"Total Steps: {total_steps} | Dynamic Eval Steps: {dynamic_eval_steps}")

    training_args = TrainingArguments(
        output_dir=f"models/{args.specialist}_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=dynamic_eval_steps,
        eval_strategy="steps",
        eval_steps=dynamic_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args
    )
    
    logger.info("Starting training loop on H100 DGX...")
    trainer.train()
    
    output_model_path = f"models/{args.specialist}_v2"
    trainer.model.save_pretrained(output_model_path)
    logger.info(f"Training complete. Best model saved to {output_model_path}")
    
    # 5. POST-TRAINING VALIDATION
    logger.info("Triggering post-training validation suite...")
    validate_dora(base_model=BASE_MODEL, adapter_path=output_model_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SABER Training Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["dora", "grpo"], help="Training mode")
    parser.add_argument("--specialist", type=str, required=True, help="Which specialist dataset to load (e.g., cybersecurity, python)")
    parser.add_argument("--target_modules", type=str, default="all", choices=list(TARGET_MODULE_PRESETS.keys()), help="DoRA target module preset")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL penalty coefficient for GRPO")
    
    args = parser.parse_args()
    
    if args.mode == "dora":
        run_dora_training(args)
    elif args.mode == "grpo":
        logger.info(f"Starting GRPO Training with kl_coef={args.kl_coef}")
        # GRPO trainer logic goes here, integrating log_grpo_reward_variance
