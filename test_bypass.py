from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator
import os
import json

os.environ['SABER_BENCHMARK_MODE'] = '1'

config = SaberConfig()
registry = SpecialistRegistry()
registry.auto_discover()
audit = AuditLogger()
orch = Orchestrator(config=config, registry=registry, audit=audit)

q = """Question: Two quantum states with energies E1 and E2 have a lifetime of 10^...
Options:
A: option 1
B: option 2
C: option 3
D: option 4

Answer the following multiple choice question. The last line of your response MUST strictly follow the format: ANSWER: LETTER (where LETTER is A, B, C, or D)."""

print('Testing TIER_0 (CoT) Bypass...')
d_scores = orch.classify_domains(q)
activated = orch.select_specialists(d_scores)
print(f"Activated domains: {activated}")

res = orch.process_query(q, tier=VerificationTier.TIER_0, bypass_meta=True)
ans = res.get('answer', '')
print("----- FINAL ANSWER RETURNED -----")
print(ans)
print("---------------------------------")
