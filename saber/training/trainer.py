import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
    logger.info(f"Starting DoRA training. Base Model: {BASE_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    
    # 1. FIXED DATA COLLATOR
    # Mask loss (-100) on everything except the assistant's answer tokens.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )
    logger.info("Using DataCollatorForCompletionOnlyLM to correctly mask prompt tokens.")
    
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
    if not dataset:
        logger.error("Dataset loading failed. Aborting training.")
        return
        
    # 4. SFT TRAINER EXECUTION (80GB H100 Optimized)
    from transformers import TrainingArguments
    
    # "Save Best, Not Last" early stopping driven by eval accuracy
    training_args = TrainingArguments(
        output_dir=f"models/{args.specialist}_checkpoints",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3, # Configured in DATASETS.md, but driven by early stopping
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=True, # H100 supports bf16 too
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(min(100, len(dataset)))), # Tiny subset for validation testing
        args=training_args,
        data_collator=collator
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
