import sys
import os
sys.path.append(os.path.abspath('.'))
from saber.registry import SpecialistRegistry

reg = SpecialistRegistry()
reg.auto_discover()
science = reg.get("science")

print("---")
# Let's mock a typical LLM response
mock_raw = """
The correct answer is B.

Reasoning:
Mitochondria are known as the powerhouse of the cell because they are responsible for cellular respiration, which produces ATP.
"""
print("Parsed claims from mock:")
for c in science.parse_raw_output_to_claims(mock_raw):
    print("-", c.statement)
