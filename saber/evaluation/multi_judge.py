# -*- coding: utf-8 -*-
"""SABER Architecture Evaluation — Multi-Judge Agreement

Supports manual input of scores from free chat interfaces (Claude, GPT-4o, Gemini)
to generate an Inter-Judge Agreement table and final averaged scores.
"""

import json
import os
from typing import Dict, List
from prettytable import PrettyTable

# A small subset of MT-Bench System Design prompts for manual testing
EVAL_PROMPTS = [
    "Design a globally scalable e-commerce backend. Focus on databases and caching.",
    "Explain how to implement Zero Trust Architecture in a hybrid cloud environment.",
    "Design a rate limiter for a public API that handles 10,000 requests per second."
]

def calculate_agreement(scores: List[Dict[str, float]]) -> float:
    """Calculate the average variance between the 3 judges across all questions.
    Lower variance = higher agreement. We convert this to a percentage agreement score.
    """
    if not scores:
        return 0.0
    
    total_variance = 0.0
    for q_scores in scores:
        vals = list(q_scores.values())
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        total_variance += variance
        
    avg_variance = total_variance / len(scores)
    # A variance of 0 means 100% agreement. Max possible variance for scores 1-10 is 20.25 (e.g. 1, 10)
    # We use a simple heuristic: Agreement = 100 - (avg_variance * 4.93)
    agreement = max(0.0, 100.0 - (avg_variance * 4.93))
    return round(agreement, 2)


def run_manual_evaluation(model_name: str) -> None:
    """Prompt the user for manual scores from 3 judges."""
    print(f"\n{'='*60}")
    print(f" MULTI-JUDGE ARCHITECTURE EVALUATION (Manual Mode)")
    print(f" Target Model: {model_name}")
    print(f" Scoring Rubric: 1.0 (Terrible) to 10.0 (Perfect)")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n--- Question {i} / {len(EVAL_PROMPTS)} ---")
        print(f"Prompt: {prompt}")
        print("Please paste the model's response into Claude, GPT-4o, and Gemini, and ask them to grade it (1-10).\n")
        
        try:
            claude = float(input("Enter Claude 3.5 Sonnet's score (1-10): "))
            gpt4o = float(input("Enter GPT-4o's score (1-10): "))
            gemini = float(input("Enter Gemini 1.5 Pro's score (1-10): "))
        except ValueError:
            print("[!] Invalid input. Defaulting scores to 0.0 for this question.")
            claude, gpt4o, gemini = 0.0, 0.0, 0.0
            
        results.append({
            "Claude": claude,
            "GPT-4o": gpt4o,
            "Gemini": gemini
        })
        
    generate_report(model_name, results)


def generate_report(model_name: str, results: List[Dict[str, float]]) -> None:
    print(f"\n\n{'='*60}")
    print(f" EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    
    table = PrettyTable()
    table.field_names = ["Question", "Claude", "GPT-4o", "Gemini", "Mean Score"]
    
    total_mean = 0.0
    for i, res in enumerate(results, 1):
        mean = round(sum(res.values()) / 3, 2)
        total_mean += mean
        table.add_row([f"Q{i}", res["Claude"], res["GPT-4o"], res["Gemini"], mean])
        
    print(table)
    
    final_avg = round(total_mean / len(results), 2)
    agreement = calculate_agreement(results)
    
    print(f"\nFinal Averaged Score:    {final_avg} / 10.0")
    print(f"Inter-Judge Agreement:   {agreement}%")
    print(f"{'='*60}\n")

    # Save to disk
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{model_name.replace('/', '_')}_architecture_eval.json", "w") as f:
        json.dump({
            "model": model_name,
            "raw_scores": results,
            "final_score": final_avg,
            "inter_judge_agreement": agreement
        }, f, indent=4)
        
if __name__ == "__main__":
    run_manual_evaluation("SABER-Architecture-V2")
