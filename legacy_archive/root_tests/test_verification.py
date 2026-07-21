from saber.specialists.science import ScienceSpecialist
from saber.signal import Signal, SignalType
import os
os.environ["SABER_BENCHMARK_MODE"] = "1"

spec = ScienceSpecialist()
spec.load_model("models/science_v2")

ver_sig = Signal(
    signal_type=SignalType.VERIFICATION_SIGNAL,
    query_id="test-query",
    source_id="BENCHMARK",
    target_id="science",
    payload={
        "issue_type": "factual_error",
        "reasoning": "The statement 'D' is a placeholder.",
        "proposed_fix": "Replace 'D' with a specific conclusion.",
        "compiled_text": "REASONING:\nEvidence suggests that\n\nCONCLUSION:\nD"
    }
).freeze_and_hash()

res = spec.handle_signal(ver_sig)
print("=== Revised Text ===")
print(res.payload.get("revised_text"))
print("====================")
