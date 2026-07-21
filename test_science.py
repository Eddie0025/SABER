from saber.specialists.science import ScienceSpecialist
import os
os.environ["SABER_BENCHMARK_MODE"] = "1"
import sys

spec = ScienceSpecialist()
# Use base model so it downloads fast/is already cached
spec.load_model("Qwen/Qwen2.5-7B")

q = """Question: Two quantum states with energies E1 and E2 have a lifetime of 10^...
Options:
A: option 1
B: option 2
C: option 3
D: option 4

Answer the following multiple choice question. The last line of your response MUST strictly follow the format: ANSWER: LETTER (where LETTER is A, B, C, or D)."""

claims = spec.process_task(q)
print("---- RAW RESPONSE ----")
print(getattr(spec, "_last_raw_response", "MISSING"))
print("----------------------")
