import json
import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_prometheus_grading():
    parser = argparse.ArgumentParser(description="Grade SABER outputs using Prometheus 2 (8B)")
    parser.add_argument("--input_file", type=str, default="benchmark_outputs.json", help="Path to benchmark_outputs.json")
    parser.add_argument("--output_file", type=str, default="prometheus_benchmark_report.json", help="Output path for final graded report")
    parser.add_argument("--model_id", type=str, default="kaist-ai/prometheus-7b-v2.0", help="Hugging Face model ID for Prometheus 2")
    args = parser.parse_args()

    print("==========================================================")
    print(f"   PROMETHEUS 2 (8B/7B) LOCAL EVALUATION ENGINE")
    print("==========================================================\n")

    if not os.path.exists(args.input_file):
        print(f"[!] Input file '{args.input_file}' not found. Run benchmark generation first.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"[*] Loaded {len(cases)} cases from '{args.input_file}'.")
    print(f"[*] Loading Prometheus 2 Judge Model ({args.model_id})...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    rubric_template = (
        "### Task Description:\n"
        "Evaluate the AI model's answer based on technical accuracy, logical reasoning, and correctness against the ground truth answer.\n\n"
        "### Question:\n{question}\n\n"
        "### Reference Answer:\n{reference}\n\n"
        "### Candidate Answer:\n{candidate}\n\n"
        "### Score Rubric:\n"
        "[Score 5]: The response is completely accurate, flawless, and perfectly matches the reference answer.\n"
        "[Score 4]: The response is mostly accurate with minor missing details.\n"
        "[Score 3]: The response is partially correct but has notable omissions.\n"
        "[Score 2]: The response has major inaccuracies.\n"
        "[Score 1]: The response is completely wrong or irrelevant.\n\n"
        "### Feedback:"
    )

    scores_by_mode = {}

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        ref = case["expected"]
        case_id = case["case_id"]

        print(f"[{idx}/{len(cases)}] Grading Case {case_id} [{case['dataset']}]...")
        sys.stdout.flush()

        for mode_name, resp_data in case.get("responses", {}).items():
            candidate_text = resp_data.get("answer", "")
            
            # Append CoT claims or Sentinel flags if available for holistic evaluation
            if "claims" in resp_data and resp_data["claims"]:
                candidate_text += "\n\n[CoT Claims]: " + " | ".join(resp_data["claims"])
            if "sentinel_flags" in resp_data and resp_data["sentinel_flags"]:
                candidate_text += "\n\n[Sentinel Audit Flags]: " + str(resp_data["sentinel_flags"])

            prompt = rubric_template.format(question=q, reference=ref, candidate=candidate_text)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            
            eval_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract 1-5 score from Prometheus format
            import re
            score_match = re.search(r"\[Score\s*([1-5])\]", eval_output) or re.search(r"\b([1-5])\b", eval_output)
            score_val = float(score_match.group(1)) * 20.0 if score_match else 50.0 # Convert 1-5 to 0-100%

            resp_data["prometheus_eval"] = {
                "score_pct": score_val,
                "feedback": eval_output.strip()
            }

            if mode_name not in scores_by_mode:
                scores_by_mode[mode_name] = []
            scores_by_mode[mode_name].append(score_val)

        if idx % 10 == 0:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print("\n==========================================================")
    print("   PROMETHEUS 2 LOCAL EVALUATION COMPLETE SUMMARY (%)")
    print("==========================================================")
    for mode, s_list in scores_by_mode.items():
        avg = sum(s_list) / len(s_list) if s_list else 0.0
        print(f" Mode: {mode:25s} | Prometheus Score: {avg:.2f}%")
    print("==========================================================\n")

if __name__ == "__main__":
    run_prometheus_grading()
