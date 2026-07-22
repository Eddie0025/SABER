# -*- coding: utf-8 -*-
"""tests/test_live_simulation.py

Integration test for SABER v2.0 full pipeline execution.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator
from saber.specialists.science import ScienceSpecialist
from saber.specialists.cybersecurity import CyberSpecialist


class TestLivePipelineSimulation(unittest.TestCase):

    def test_full_pipeline_simulation(self):
        config = SaberConfig(verification_tier=VerificationTier.TIER_1)
        registry = SpecialistRegistry()
        registry.register(ScienceSpecialist())
        registry.register(CyberSpecialist())
        audit = AuditLogger("data/test_sim_audit.jsonl")

        orchestrator = Orchestrator(config, registry, audit)

        with patch("saber.llm_engine.LLMEngine.__enter__") as mock_enter:
            mock_engine = MagicMock()
            mock_engine.generate.return_value = (
                "## CLAIM EXTRACTION\n1. Science: E=mc^2\n\n"
                "## FINAL ANSWER\nThe physics calculation is verified."
            )
            mock_enter.return_value = mock_engine

            query = "Calculate energy of mass in physics and analyze port 445."
            result = orchestrator.process_query(query)

            self.assertIn("query_id", result)
            self.assertIn("science", result["domains_activated"])
            self.assertTrue(len(result["answer"]) > 0)
            self.assertIn("Verified by SABER Sentinel", result["answer"])


if __name__ == "__main__":
    unittest.main()
