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

def generate_answers(model, tokenizer, questions, cot=False):
    answers = []
    for q in questions:
        q_text = q["question"]
        if cot:
            q_text += "\n\nLet's think step by step. Explain your reasoning."
        
        prompt = [{"role": "user", "content": q_text}]
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
        answers.append({
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "generated_answer": answer
        })
    return answers

def offload(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def phase_1_generate():
    os.makedirs("results", exist_ok=True)
    with open(TEST_SET_FILE, "r") as f:
        test_set = json.load(f)

    raw_answers = {spec: {"base": [], "specialist": [], "specialist_cot": [], "full_saber": []} for spec in test_set}

    print("\n=== Evaluating Base Model ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.eval()

    for spec, questions in test_set.items():
        print(f"Base Model -> {spec}")
        raw_answers[spec]["base"] = generate_answers(base_model, tokenizer, questions)
    
    offload(base_model, tokenizer)

    for spec, questions in test_set.items():
        print(f"\n=== Evaluating Specialist: {spec} ===")
        # Specialist v2 (DoRA SFT)
        v2_path = f"models/{spec}_v2"
        if os.path.exists(v2_path):
            print(f"Loading {v2_path}...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
            model = PeftModel.from_pretrained(model, v2_path)
            model.eval()
            
            print(f"Running 'specialist' mode...")
            raw_answers[spec]["specialist"] = generate_answers(model, tokenizer, questions, cot=False)
            print(f"Running 'specialist_cot' mode...")
            raw_answers[spec]["specialist_cot"] = generate_answers(model, tokenizer, questions, cot=True)
            
            offload(model, tokenizer)
        else:
            print(f"Missing {v2_path}, skipping specialist modes.")

        # Full SABER (GRPO Final)
        grpo_path = f"models/{spec}_grpo_final"
        if not os.path.exists(grpo_path):
            grpo_path = f"models/{spec}_grpo"
        
        if os.path.exists(grpo_path):
            print(f"Loading {grpo_path} (Full SABER)...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
            model = PeftModel.from_pretrained(model, grpo_path)
            model.eval()
            
            print(f"Running 'full_saber' mode...")
            raw_answers[spec]["full_saber"] = generate_answers(model, tokenizer, questions, cot=False)
            
            offload(model, tokenizer)
        else:
            print(f"Missing {grpo_path}, skipping full_saber mode.")

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

    for spec, modes in raw_answers.items():
        print(f"\nGrading {spec}...")
        final_report[spec] = {}
        for mode_name, answers in modes.items():
            if not answers:
                continue
            print(f"  Mode: {mode_name}")
            final_report[spec][mode_name] = []
            
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
                
                final_report[spec][mode_name].append({
                    "question": item["question"],
                    "generated_answer": item["generated_answer"],
                    "prometheus_feedback": feedback_output,
                    "score": score
                })
                print(f"    Score: {score}/5")

    with open(FINAL_REPORT_FILE, "w") as f:
        json.dump(final_report, f, indent=4)
    print(f"\nPhase 2 complete. Final report saved to {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    phase_1_generate()
    phase_2_evaluate()
