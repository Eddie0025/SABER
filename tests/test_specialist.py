# -*- coding: utf-8 -*-
"""tests/test_specialist.py

Unit tests for SABER Specialist lifecycle and signal processing.
"""

import unittest
from unittest.mock import MagicMock, patch
from saber.signal import Signal, SignalType
from saber.specialists.science import ScienceSpecialist
from saber.specialists.cybersecurity import CyberSpecialist


class TestSpecialistLifecycle(unittest.TestCase):

    def setUp(self):
        # Patch LLMEngine to prevent heavy model downloads during unit tests
        self.engine_patcher = patch("saber.llm_engine.LLMEngine.__enter__")
        self.mock_enter = self.engine_patcher.start()
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "CLAIMS:\n1. Mocked claim statement\nANSWER: A"
        self.mock_enter.return_value = mock_engine

    def tearDown(self):
        self.engine_patcher.stop()

    def test_specialist_initialization(self):
        spec = ScienceSpecialist()
        self.assertEqual(spec.domain, "science")
        self.assertTrue(spec.meta.specialist_id.startswith("SPEC-SCIENCE-"))
        self.assertIn("physics", spec.keywords)

    def test_task_signal_handling(self):
        spec = ScienceSpecialist()
        spec.meta.model_path = "Qwen/Qwen2.5-7B-Instruct"
        task_sig = Signal(
            signal_type=SignalType.TASK_SIGNAL,
            query_id="query-spec-01",
            source_id="BENCHMARK",
            target_id="science",
            payload={"objective": "Calculate momentum of 5kg object moving at 10m/s."}
        ).freeze_and_hash()

        out_sig = spec.handle_signal(task_sig)
        self.assertEqual(out_sig.signal_type, SignalType.COT_SIGNAL)
        self.assertEqual(out_sig.query_id, "query-spec-01")
        self.assertTrue(out_sig.verify_integrity())
        self.assertIn("claims", out_sig.payload)

    def test_verification_signal_handling(self):
        spec = CyberSpecialist()
        ver_sig = Signal(
            signal_type=SignalType.VERIFICATION_SIGNAL,
            query_id="query-spec-02",
            source_id="BENCHMARK",
            target_id="cyber",
            payload={
                "issue_type": "factual_error",
                "reasoning": "Missing technique ID.",
                "proposed_fix": "Add T1059.",
                "compiled_text": "Port 80 is HTTP.",
                "question": "Which port is used for web traffic?"
            }
        ).freeze_and_hash()

        res_sig = spec.handle_signal(ver_sig)
        self.assertEqual(res_sig.signal_type, SignalType.VERIFICATION_SIGNAL)
        self.assertTrue(res_sig.verify_integrity())


if __name__ == "__main__":
    unittest.main()
