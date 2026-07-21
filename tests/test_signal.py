# -*- coding: utf-8 -*-
"""tests/test_signal.py

Unit tests for SABER Signal & Claim schemas.
Verifies hashing, immutability checks, and payload serialization.
"""

import unittest
from saber.signal import Signal, SignalType, Claim, ClaimStatus


class TestSignalSchema(unittest.TestCase):

    def test_claim_creation(self):
        claim = Claim(
            statement="Quantum states are orthogonal.",
            confidence=0.95,
            domain="science"
        )
        self.assertTrue(claim.claim_id.startswith("C-"))
        self.assertEqual(claim.status, ClaimStatus.UNVERIFIED)
        self.assertEqual(claim.confidence, 0.95)

    def test_signal_hashing_and_verification(self):
        sig = Signal(
            signal_type=SignalType.TASK_SIGNAL,
            query_id="query-101",
            source_id="ORCHESTRATOR",
            target_id="science",
            payload={"objective": "Solve physics problem"}
        ).freeze_and_hash()

        self.assertTrue(len(sig.integrity_hash) == 64)
        self.assertTrue(sig.verify_integrity())

        # Mutate payload and verify hash failure
        sig.payload["objective"] = "Mutated physics problem"
        self.assertFalse(sig.verify_integrity())

    def test_signal_serialization(self):
        sig = Signal(
            signal_type=SignalType.OUTPUT_SIGNAL,
            query_id="query-102",
            source_id="science",
            target_id="META_REASONER",
            payload={"raw_response": "ANSWER: C"}
        ).freeze_and_hash()

        dumped = sig.model_dump_json()
        restored = Signal.model_validate_json(dumped)
        self.assertEqual(sig.signal_id, restored.signal_id)
        self.assertTrue(restored.verify_integrity())


if __name__ == "__main__":
    unittest.main()
