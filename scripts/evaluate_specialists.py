import json
import os
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
TEST_SET_FILE = "saber/eval/hard_test_set.json"
TEMP_ANSWERS_FILE = "results/raw_answers.json"
FINAL_REPORT_FILE = "results/hard_eval_report.json"

PROMETHEUS_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{answer}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and comprehensive based on the reference answer?]
Score 1: The response is completely incorrect or irrelevant.
Score 2: The response has significant errors or misses key points.
Score 3: The response is partially correct but lacks depth.
Score 4: The response is mostly correct and comprehensive.
Score 5: The response is perfectly correct, accurate, and comprehensive.

###Feedback:
"""

def phase_1_generate():
    os.makedirs("results", exist_ok=True)
    with open(TEST_SET_FILE, "r") as f:
        test_set = json.load(f)

    raw_answers = {}

    for specialist, questions in test_set.items():
        adapter_path = f"models/{specialist}_grpo_final"
        # Fallback if _grpo_final is missing (e.g., architecture_planner)
        if not os.path.exists(adapter_path):
            adapter_path = f"models/{specialist}_grpo"
        
        if not os.path.exists(adapter_path):
            print(f"Skipping {specialist} - adapter not found at {adapter_path}")
            continue

        print(f"\n--- Loading {specialist} ---")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.padding_side = "left"

        print(f"Loading Base Model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"Loading Adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
        specialist_answers = []

        print(f"Generating answers for {len(questions)} questions...")
        for q in questions:
            prompt = [{"role": "user", "content": q["question"]}]
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            specialist_answers.append({
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "generated_answer": answer
            })
            print(f"Answered: {q['question'][:50]}...")
        
        raw_answers[specialist] = specialist_answers

        # CRITICAL: Offload model entirely
        print(f"Offloading {specialist} model from GPU...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    with open(TEMP_ANSWERS_FILE, "w") as f:
        json.dump(raw_answers, f, indent=4)
    print(f"Phase 1 complete. Saved to {TEMP_ANSWERS_FILE}")


def phase_2_evaluate():
    if not os.path.exists(TEMP_ANSWERS_FILE):
        print(f"{TEMP_ANSWERS_FILE} not found. Run Phase 1 first.")
        return

    with open(TEMP_ANSWERS_FILE, "r") as f:
        raw_answers = json.load(f)

    print(f"\n--- Loading Prometheus 2.0 in bfloat16 ---")
    tokenizer = AutoTokenizer.from_pretrained(PROMETHEUS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PROMETHEUS_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    final_report = {}

    for specialist, answers in raw_answers.items():
        print(f"\nGrading {specialist}...")
        final_report[specialist] = []
        for item in answers:
            prompt = PROMETHEUS_PROMPT_TEMPLATE.format(
                question=item["question"],
                answer=item["generated_answer"],
                reference_answer=item["reference_answer"]
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0
                )
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            feedback_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            score = 0
            match = re.search(r"\[RESULT\]\s*(\d+)", feedback_output)
            if match:
                score = int(match.group(1))
            else:
                print(f"Warning: Could not parse score from output: {feedback_output}")

            final_report[specialist].append({
                "question": item["question"],
                "generated_answer": item["generated_answer"],
                "prometheus_feedback": feedback_output,
                "score": score
            })
            print(f"  Score: {score}/5")

    with open(FINAL_REPORT_FILE, "w") as f:
        json.dump(final_report, f, indent=4)
    print(f"\nPhase 2 complete. Final report saved to {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    phase_1_generate()
    phase_2_evaluate()
