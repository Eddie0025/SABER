import os
import sys
import json
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

logger = logging.getLogger("SABER_Trainer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_target_modules(preset_name: str) -> list:
    if preset_name not in TARGET_MODULE_PRESETS:
        raise ValueError(f"Unknown target_modules preset: {preset_name}. Valid presets: {list(TARGET_MODULE_PRESETS.keys())}")
    return TARGET_MODULE_PRESETS[preset_name]

def run_dora_training(args):
    """
    SFT DoRA Training with best-checkpoint selection across 5 epochs.
    """
    logger.info("Setting up tokenizer and padding...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Critical fix for Qwen/Llama padding crashes
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # For SFT, right padding is standard
    
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

    # 5. STANDARD NATIVE TRAINER EXECUTION (80GB H100 Optimized - 5 Epochs)
    # Batch=2 x GradAccum=16 = effective batch 32
    # Dynamically calculate eval_steps to evaluate ~15 times across 5 epochs
    effective_batch = 2 * 16
    num_epochs = 5
    total_steps = (len(tokenized_dataset) // effective_batch) * num_epochs
    dynamic_eval_steps = max(1, total_steps // 15)
    logger.info(f"Total Steps: {total_steps} (5 Epochs) | Dynamic Eval & Save Steps: {dynamic_eval_steps}")

    training_args = TrainingArguments(
        output_dir=f"models/{args.specialist}_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=num_epochs,
        save_strategy="steps",
        save_steps=dynamic_eval_steps,
        eval_strategy="steps",
        eval_steps=dynamic_eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    
    logger.info("Starting 5-epoch training loop on H100 DGX...")
    train_result = trainer.train()
    
    # 1. RETRIEVE BEST CHECKPOINT METRICS & SAVE
    best_checkpoint = trainer.state.best_model_checkpoint or "Final Step"
    best_metric = trainer.state.best_metric if trainer.state.best_metric is not None else "N/A"
    logger.info(f"Best Checkpoint Identified: {best_checkpoint} with Best Eval Loss: {best_metric}")

    output_model_path = f"models/{args.specialist}_v2"
    trainer.model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    logger.info(f"Best model weights successfully saved to {output_model_path}")
    
    # Save checkpoint metadata
    best_info = {
        "specialist": args.specialist,
        "best_checkpoint": best_checkpoint,
        "best_eval_loss": best_metric,
        "total_epochs": num_epochs,
        "total_steps": total_steps
    }
    os.makedirs(output_model_path, exist_ok=True)
    with open(os.path.join(output_model_path, "best_checkpoint_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # 2. SAVE LOGS SAFELY
    try:
        os.makedirs("logs", exist_ok=True)
        dora_log_path = f"logs/{args.specialist}_dora_epoch_logs.json"
        with open(dora_log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Saved DoRA epoch logs to {dora_log_path}")
        
        summary_path = f"logs/{args.specialist}_dora_summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# DoRA Training 5-Epoch Summary: {args.specialist}\n\n")
            f.write(f"- **Best Checkpoint**: `{best_checkpoint}`\n")
            f.write(f"- **Best Eval Loss**: `{best_metric}`\n")
            f.write(f"- **Total Epochs**: {num_epochs}\n\n")
            f.write("| Step | Epoch | Train Loss | Eval Loss | Grad Norm | Learning Rate |\n")
            f.write("|---|---|---|---|---|---|\n")
            for entry in trainer.state.log_history:
                step = entry.get("step", "N/A")
                epoch = f"{entry.get('epoch', 0):.2f}" if "epoch" in entry else "N/A"
                t_loss = f"{entry.get('loss', 'N/A')}"
                e_loss = f"{entry.get('eval_loss', 'N/A')}"
                gnorm = f"{entry.get('grad_norm', 'N/A')}"
                lr = f"{entry.get('learning_rate', 'N/A')}"
                f.write(f"| {step} | {epoch} | {t_loss} | {e_loss} | {gnorm} | {lr} |\n")
        logger.info(f"Saved DoRA training summary to {summary_path}")
    except Exception as e:
        logger.warning(f"Non-fatal error while writing DoRA logs: {e}")
    
    # 3. POST-TRAINING VALIDATION (NON-FATAL)
    try:
        from scripts.validate_dora import validate_dora
        logger.info("Triggering post-training validation suite...")
        validate_dora(model=trainer.model, tokenizer=tokenizer, adapter_path=output_model_path)
    except Exception as e:
        logger.warning(f"Post-training validation encountered an error (non-fatal, model is saved): {e}")
    

def run_grpo_training(args):
    """
    GRPO Training Phase.
    Loads the fixed DoRA adapter (_v2) as the starting/reference model.
    """
    # --- PyTorch Compatibility Monkeypatch ---
    # TRL's lazy loader crashes on PyTorch < 2.2 because FSDPModule doesn't exist.
    # We are using single-GPU (device_map="auto"), so FSDP is irrelevant. Safe to stub.
    try:
        import torch.distributed.fsdp
        if not hasattr(torch.distributed.fsdp, "FSDPModule"):
            torch.distributed.fsdp.FSDPModule = type("FSDPModule", (), {})
    except Exception:
        pass
        
    try:
        from trl import GRPOTrainer, GRPOConfig
    except (ImportError, RuntimeError) as e:
        logger.error(f"trl library error (cannot import GRPO): {e}")
        return
        
    logger.info("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation
    
    # 1. Load the fixed DoRA adapter as the reference model
    reference_adapter_path = f"models/{args.specialist}_v2"
    if not os.path.exists(reference_adapter_path):
        logger.error(f"Cannot run GRPO: DoRA checkpoint {reference_adapter_path} not found.")
        return
        
    logger.info(f"Loading Base Model in bfloat16 for GRPO...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Merge DoRA weights into the base model to create a solid reference
    from peft import PeftModel
    logger.info(f"Fusing DoRA weights from {reference_adapter_path} into base model...")
    model = PeftModel.from_pretrained(model, reference_adapter_path)
    model = model.merge_and_unload()
    
    # We apply a NEW DoRA adapter for the RL phase
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        
    target_modules = get_target_modules(args.target_modules)
    peft_config = LoraConfig(
        r=DORA_CONFIG["r"],
        lora_alpha=DORA_CONFIG["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=DORA_CONFIG["lora_dropout"],
        use_dora=DORA_CONFIG["use_dora"],
        task_type=DORA_CONFIG["task_type"]
    )
    
    # 2. LOAD DATASET (GRPO needs far fewer samples than SFT — quality > quantity)
    logger.info(f"Loading dataset for GRPO...")
    dataset = load_specialist_dataset(args.specialist, max_samples=3000)
    if dataset is None:
        return
        
    # Format for GRPO: needs a "prompt" column and an "answer" column
    def format_for_grpo(example):
        question = example.get("question", "")
        prompt = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_text, "answer": example.get("answer", "")}
        
    grpo_dataset = dataset.map(format_for_grpo, remove_columns=[c for c in dataset.column_names if c not in ["prompt", "answer"]])
    
    # 3. CONFIGURE GRPO (Single GPU Optimized for Speed + Quality)
    # - num_generations=4: good advantage estimation
    # - max_completion_length=512: full answer length
    # - batch_size=2, grad_accum=4: effective batch of 8
    # - 3k dataset cap keeps steps at ~1500 → ~10 hours per specialist
    grpo_args = GRPOConfig(
        output_dir=f"models/{args.specialist}_grpo",
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        generation_batch_size=4,
        max_completion_length=512,
        beta=args.kl_coef,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    
    from saber.training.rewards import combined_reward
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],
        args=grpo_args,
        train_dataset=grpo_dataset,
        peft_config=peft_config
    )
    
    logger.info("Starting GRPO Training loop...")
    trainer.train()
    
    # 1. ALWAYS SAVE FINAL MODEL FIRST
    final_model_path = f"models/{args.specialist}_grpo_final"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"GRPO Training Complete. Final model saved to {final_model_path}")
    
    # 2. SAVE LOGS SAFELY
    try:
        os.makedirs("logs", exist_ok=True)
        grpo_log_path = f"logs/{args.specialist}_grpo_epoch_logs.json"
        with open(grpo_log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Saved GRPO epoch logs to {grpo_log_path}")
    except Exception as e:
        logger.warning(f"Non-fatal error while writing GRPO logs: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SABER Training Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["dora", "grpo"], help="Training mode")
    parser.add_argument("--specialist", type=str, required=True, help="Which specialist dataset to load (e.g., cybersecurity, python)")
    parser.add_argument("--target_modules", type=str, default="all", choices=list(TARGET_MODULE_PRESETS.keys()), help="DoRA target module preset")
    parser.add_argument("--kl_coef", type=float, default=0.04, help="KL penalty coefficient for GRPO")
    
    args = parser.parse_args()
    
    if args.mode == "dora":
        run_dora_training(args)
    elif args.mode == "grpo":
        logger.info(f"Starting GRPO Training with kl_coef={args.kl_coef}")
        run_grpo_training(args)
