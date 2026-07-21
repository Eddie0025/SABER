# -*- coding: utf-8 -*-
"""scripts.generate_benchmark

Generates the SABER_BENCHMARK_v1 dataset.
This dataset is strictly isolated from training data.
"""

import json
import os
from saber.llm_engine import LLMEngine

BENCHMARK_FILE = "data/benchmark/saber_benchmark_v1.jsonl"

CYBER_PROMPTS = [
    "Generate a complex Incident Response reasoning question. It must require understanding of both network forensics and malware analysis.",
    "Generate a Threat Hunting scenario involving a sophisticated APT using living-off-the-land techniques.",
    "Generate a difficult question regarding IAM role misconfigurations in AWS leading to privilege escalation.",
]

SCIENCE_PROMPTS = [
    "Generate a difficult physics question combining classical mechanics and thermodynamics.",
    "Generate a complex biology question regarding cellular respiration and enzymatic inhibitors.",
    "Generate a chemistry question about calculating equilibrium concentrations in a multi-step reaction.",
]

CROSS_PROMPTS = [
    "A hospital ransomware attack affects MRI systems. Discuss the cybersecurity implications, medical implications, and operational implications.",
    "An industrial control system (ICS) at a chemical plant is breached, altering the pressure valves of a reactor. Discuss the cyber attack vectors and the physical chemistry consequences.",
    "A pharmaceutical research database containing proprietary genomic sequences is exfiltrated. Discuss the cybersecurity failure and the scientific/medical impact of the stolen data.",
]

MEDICAL_PROMPTS = [
    "Generate a complex clinical case study involving a patient presenting with overlapping symptoms of Lupus and Rheumatoid Arthritis, requiring differential diagnosis.",
    "Generate a difficult question regarding rare pharmacological interactions between a newly prescribed anti-arrhythmic and the patient's existing psychiatric medication.",
    "Generate an advanced neuro-oncology scenario detailing the diagnostic imaging and treatment plan for a glioblastoma multiforme."
]

CODING_PROMPTS = [
    "Generate a difficult software engineering scenario where a distributed system experiences race conditions during high-concurrency database writes.",
    "Generate a complex debugging question involving a memory leak in a C++ application using custom memory allocators.",
    "Generate a question asking to optimize a highly recursive, computationally expensive graph traversal algorithm to reduce time complexity from O(N!) to O(N^2)."
]

ARCHITECTURE_PROMPTS = [
    "Generate an architecture design scenario for migrating a monolithic on-premise application to a microservices architecture on AWS with zero downtime.",
    "Generate a complex threat modeling scenario for a cloud-native financial application, detailing the implementation of Zero Trust principles.",
    "Generate a system design question focusing on the trade-offs between eventual consistency and strong consistency in a globally distributed multi-region database."
]

def generate_questions(engine, prompts, domain, prefix, count=5):
    """Generate questions using the LLM."""
    questions = []
    system_prompt = (
        "You are an expert dataset creator for advanced AI benchmarks. "
        "Output ONLY a valid JSON object with the following keys: "
        '"question" (the scenario/question), "difficulty" ("hard"), "ground_truth" (the correct comprehensive answer), '
        'and "reasoning_points" (a list of 3-5 key logical points that must be hit).'
    )
    
    idx = 1
    for prompt in prompts:
        # Just generate 1 per prompt for the sample
        try:
            res = engine.generate(prompt, system_prompt=system_prompt)
            clean_res = res.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_res)
            data["question_id"] = f"{prefix}-{idx:03d}"
            data["domain"] = domain
            questions.append(data)
            idx += 1
        except Exception as e:
            print(f"Failed to generate for {prefix}: {e}")
            
    return questions

def main():
    os.makedirs(os.path.dirname(BENCHMARK_FILE), exist_ok=True)
    
    questions = []
    print("Generating SABER_BENCHMARK_v1...")
    
    # We use a larger/better model if available, but fallback to local Qwen
    try:
        with LLMEngine("Qwen/Qwen2.5-7B") as engine:
            print("Generating Cyber questions...")
            questions.extend(generate_questions(engine, CYBER_PROMPTS, "cyber", "CYBER"))
            
            print("Generating Science questions...")
            questions.extend(generate_questions(engine, SCIENCE_PROMPTS, "science", "SCI"))
            
            print("Generating Medical questions...")
            questions.extend(generate_questions(engine, MEDICAL_PROMPTS, "medical", "MED"))
            
            print("Generating Coding questions...")
            questions.extend(generate_questions(engine, CODING_PROMPTS, "coding", "CODE"))
            
            print("Generating Architecture questions...")
            questions.extend(generate_questions(engine, ARCHITECTURE_PROMPTS, "architecture", "ARCH"))
            
            print("Generating Cross-Domain questions...")
            questions.extend(generate_questions(engine, CROSS_PROMPTS, "cross_domain", "XDOMAIN"))
    except Exception as e:
        print(f"LLM Engine failed: {e}")
        return

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    print(f"Successfully generated {len(questions)} benchmark questions at {BENCHMARK_FILE}")

if __name__ == "__main__":
    main()
