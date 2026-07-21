import sys
import os
import json
sys.path.append(os.path.abspath('.'))

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

config = SaberConfig()
registry = SpecialistRegistry()
registry.auto_discover()
for domain, specialist in registry.all().items():
    specialist.load_model("Qwen/Qwen2.5-7B")

audit = AuditLogger()
orch = Orchestrator(config=config, registry=registry, audit=audit)

# Dummy GPQA case
q = """Question: Which of the following is true?
Options:
A: Apples are blue
B: Apples are red
C: Apples are green
D: Apples are purple"""
expected = "B"

res = orch.process_query(q, tier=VerificationTier.TIER_0)
print(json.dumps(res, indent=2))

ans = res.get("answer", "").strip()
print("FINAL ANSWER:", ans)

is_correct = False
if expected.lower() in ans.lower():
    is_correct = True
print("CORRECT?", is_correct)
