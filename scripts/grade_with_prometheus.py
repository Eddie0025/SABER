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

    # MCQ Rubric: Strict Letter Option & Reasoning Verification
    mcq_rubric_template = (
        "### Task Description:\n"
        "The question is a Multiple Choice Question (MCQ). Evaluate the Candidate Response against the Reference Answer letter/option.\n"
        "Note: Advanced modes (CoT and Sentinel) contain internal reasoning chains, extracted claims, and audit flags. "
        "Search through the candidate response, CoT claims, or Sentinel statements to determine if the correct option letter (A, B, C, or D) or correct factual choice is selected.\n\n"
        "### Question:\n{question}\n\n"
        "### Reference Answer:\n{reference}\n\n"
        "### Candidate Response & Agent Reasoning:\n{candidate}\n\n"
        "### Score Rubric:\n"
        "[Score 5]: The correct MCQ option (or letter) is explicitly selected or logically derived in the response or CoT/Sentinel statements.\n"
        "[Score 1]: An incorrect MCQ option is selected or no valid choice is derived.\n\n"
        "### Feedback:"
    )

    # Open-Ended Rubric: Deep Technical & SEC Financial Evaluation
    open_rubric_template = (
        "### Task Description:\n"
        "The question is Open-Ended (Financial SEC analysis or Software Architecture design). Evaluate technical accuracy, numerical precision, and trade-off depth against the reference answer.\n"
        "Note: Advanced modes (CoT and Sentinel) include multi-step reasoning chains and verification audit flags. Evaluate the candidate's final synthesis, reasoning claims, and revised statements.\n\n"
        "### Question:\n{question}\n\n"
        "### Reference Answer:\n{reference}\n\n"
        "### Candidate Response & Agent Reasoning:\n{candidate}\n\n"
        "### Score Rubric:\n"
        "[Score 5]: The response is completely accurate, contains all exact numerical/architectural details, and perfectly aligns with the reference answer.\n"
        "[Score 4]: The response is mostly accurate with minor missing details.\n"
        "[Score 3]: The response is partially correct but has notable omissions.\n"
        "[Score 2]: The response has major inaccuracies or wrong numbers.\n"
        "[Score 1]: The response is completely wrong or irrelevant.\n\n"
        "### Feedback:"
    )

    scores_by_mode = {}
    dataset_scores = {}

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        ref = str(case["expected"]).strip()
        case_id = case["case_id"]
        c_type = case.get("type", "open_text")
        dataset_name = case.get("dataset", "unknown")

        # Detect if MCQ
        is_mcq = ref.upper() in ["A", "B", "C", "D"] or c_type == "exact"

        print(f"[{idx}/{len(cases)}] Grading Case {case_id} [{'MCQ' if is_mcq else 'OPEN-ENDED'}]...")
        sys.stdout.flush()

        for mode_name, resp_data in case.get("responses", {}).items():
            candidate_text = resp_data.get("answer", "")
            
            # Extract and format CoT claims and Sentinel revision statements for Prometheus
            extra_context = []
            if "claims" in resp_data and resp_data["claims"]:
                extra_context.append("--- Extracted Chain-of-Thought (CoT) Claims ---")
                for c_idx, claim_stmt in enumerate(resp_data["claims"], 1):
                    extra_context.append(f"Claim {c_idx}: {claim_stmt}")
            
            if "sentinel_flags" in resp_data and resp_data["sentinel_flags"]:
                extra_context.append("--- Sentinel Audit Verification Flags ---")
                for f_idx, flag in enumerate(resp_data["sentinel_flags"], 1):
                    extra_context.append(f"Flag {f_idx}: {flag}")

            if extra_context:
                candidate_text += "\n\n" + "\n".join(extra_context)

            # Select MCQ vs Open-Ended Rubric
            if is_mcq:
                prompt = mcq_rubric_template.format(question=q, reference=ref, candidate=candidate_text)
            else:
                prompt = open_rubric_template.format(question=q, reference=ref, candidate=candidate_text)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            
            eval_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract 1-5 score from Prometheus format
            import re
            score_match = re.search(r"\[Score\s*([1-5])\]", eval_output) or re.search(r"\b([1-5])\b", eval_output)
            score_val = float(score_match.group(1)) * 20.0 if score_match else 50.0 # Convert 1-5 to 0-100%

            if dataset_name not in dataset_scores:
                dataset_scores[dataset_name] = defaultdict(list)
            dataset_scores[dataset_name][mode_name].append(score_val)

        if idx % 10 == 0:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    # =====================================================================
    # Format Separate Markdown Tables per Dataset & Respective Modes (%)
    # =====================================================================
    mode_names = ["Base Qwen", "Qwen with Adaptors", "Qwen Adaptor + CoT", "Sentinel 2 Pass"]
    table_lines = [
        "# SABER Benchmark Results — Prometheus 2 (8B) Evaluation Report\n",
        "## Overall Dataset Score Summary (%)\n",
        f"| Dataset | {' | '.join(mode_names)} |",
        f"| :--- | {' | '.join([':---'] * len(mode_names))} |"
    ]

    print("\n======================================================================")
    print(" 📊 PROMETHEUS 2 (8B) BENCHMARK REPORT — PER-DATASET PERCENTAGES (%)")
    print("======================================================================\n")

    for ds, m_dict in dataset_scores.items():
        row = [ds.upper()]
        for mode in mode_names:
            s_list = m_dict.get(mode, [])
            if s_list:
                avg_score = sum(s_list) / len(s_list)
                row.append(f"{avg_score:.1f}%")
            else:
                row.append("N/A")
        table_lines.append("| " + " | ".join(row) + " |")

    report_md = "\n".join(table_lines)
    print(report_md)
    print("\n======================================================================\n")

    md_file = args.output_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(report_md + "\n")

    print(f"[+] Markdown report saved to '{md_file}'.")
    print(f"[+] Full JSON report saved to '{args.output_file}'.")

if __name__ == "__main__":
    run_prometheus_grading()
