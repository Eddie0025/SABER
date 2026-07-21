# -*- coding: utf-8 -*-
"""tests/test_pipeline_flow.py

End-to-end integration test for SABER system and architecture flow.
Simulates a query moving through Orchestrator -> Specialist -> Sentinel -> Meta-Reasoner -> Audit Ledger.
"""

import unittest
from unittest.mock import MagicMock, patch
from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator
from saber.specialists.science import ScienceSpecialist
from saber.specialists.cybersecurity import CyberSpecialist


class TestFullPipelineFlow(unittest.TestCase):

    def setUp(self):
        # Patch LLMEngine to simulate fast offline responses
        self.engine_patcher = patch("saber.llm_engine.LLMEngine.__enter__")
        self.mock_enter = self.engine_patcher.start()
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "CLAIMS:\n1. Momentum p = m * v\nANSWER: C"
        self.mock_enter.return_value = mock_engine

        self.config = SaberConfig(verification_tier=VerificationTier.TIER_1)
        self.registry = SpecialistRegistry()
        self.registry.register(ScienceSpecialist())
        self.registry.register(CyberSpecialist())
        self.audit = AuditLogger("data/test_pipeline_audit.jsonl")
        self.orchestrator = Orchestrator(self.config, self.registry, self.audit)

    def tearDown(self):
        self.engine_patcher.stop()

    def test_end_to_end_single_domain_flow(self):
        query = "Calculate kinetic energy and momentum of a 10kg mass moving at 5m/s in physics."
        res = self.orchestrator.process_query(query, bypass_meta=True)

        self.assertIn("query_id", res)
        self.assertIn("science", res["domains_activated"])
        self.assertEqual(res["verification_tier"], VerificationTier.TIER_1)
        self.assertTrue(len(res["answer"]) > 0)

    def test_ambiguous_query_flow(self):
        query = "it is this"
        res = self.orchestrator.process_query(query)

        self.assertEqual(res["status"], "clarification_needed")
        self.assertGreater(res["ambiguity_score"], 0.4)


if __name__ == "__main__":
    unittest.main()
