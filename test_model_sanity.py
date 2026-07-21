from saber.llm_engine import LLMEngine
import os

q = "A correlation study shows that people who drink more coffee live longer. Explain why this result alone does not establish causation and identify possible confounding variables."

print("=== Base Qwen Output ===")
with LLMEngine("Qwen/Qwen2.5-7B-Instruct") as engine:
    print(engine.generate(q, system_prompt="You are a science specialist with expertise in physics, chemistry, biology, and mathematical reasoning. Show all work and explain each step clearly. Think through your reasoning step by step before providing your final answer."))

print("\n=== Science Specialist (LoRA Adapter) Output ===")
with LLMEngine("models/science_v2") as engine:
    print(engine.generate(q, system_prompt="You are a science specialist with expertise in physics, chemistry, biology, and mathematical reasoning. Show all work and explain each step clearly. Think through your reasoning step by step before providing your final answer."))
