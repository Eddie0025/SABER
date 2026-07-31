import json
import logging
import numpy as np
from pathlib import Path

# Configure logger
logger = logging.getLogger("SABER_Rewards")

def log_grpo_reward_variance(step: int, prompt_id: str, group_rewards: list, log_path: str = "logs/grpto_reward_variance.jsonl", threshold: float = 0.05):
    """
    Computes and logs reward variance for a single group of rollouts during GRPO training.
    """
    rewards = np.array(group_rewards)
    mean_val = float(np.mean(rewards))
    std_val = float(np.std(rewards))
    min_val = float(np.min(rewards))
    max_val = float(np.max(rewards))
    
    # Log warning if variance is near-zero (no learning signal)
    if std_val < threshold:
        logger.warning(f"Step {step}, Prompt {prompt_id}: Reward std-dev {std_val:.4f} is below threshold {threshold}. Group lacks advantage signal.")

    log_entry = {
        "step": step,
        "prompt_id": prompt_id,
        "group_size": len(rewards),
        "reward_mean": mean_val,
        "reward_std": std_val,
        "reward_min": min_val,
        "reward_max": max_val
    }
    
    # Append to JSONL
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

def summarize_reward_variance(log_path: str, threshold: float = 0.05):
    """
    Reads the JSONL log and reports what % of groups fell below the std-dev threshold.
    """
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return
        
    total_groups = 0
    low_variance_groups = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            total_groups += 1
            data = json.loads(line)
            if data.get("reward_std", 1.0) < threshold:
                low_variance_groups += 1
                
    if total_groups == 0:
        return
        
    percentage = (low_variance_groups / total_groups) * 100
    print(f"--- GRPO Reward Variance Summary ---")
    print(f"Total Groups Evaluated: {total_groups}")
    print(f"Groups below threshold ({threshold}): {low_variance_groups} ({percentage:.2f}%)")
    
    if percentage > 30.0:
        logger.warning(f"FLAG: >30% ({percentage:.2f}%) of groups had near-zero variance. Check reward function bounds.")
    
    return {
        "total_groups": total_groups,
        "low_variance_groups": low_variance_groups,
        "percentage": percentage
    }
