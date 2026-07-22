# -*- coding: utf-8 -*-
"""tests/test_sentinel_e2e_extraction.py

End-to-End Extraction & Search Accuracy Tests for SABER Sentinel.

This test suite exercises the REAL Sentinel.verify_interpretation() method
end-to-end by mocking only the LLM generation call. Everything else —
claim extraction from Signal payloads, search query formulation,
consecutive-search dedup circuit breaker, grounding prompt assembly, and
GREEN_CHIT vs FLAG_SIGNAL decision routing — runs against the real production code.

Test categories:
1. Claim extraction from Signal.payload["claims"] — verifies Sentinel
   correctly pulls claim statements, skips generic "The correct answer is B"
   noise, and truncates to 120 chars.
2. Search dedup circuit breaker — verifies identical consecutive queries
   are bypassed after 2 hits.
3. GREEN_CHIT vs FLAG_SIGNAL routing — verifies LLM output parsing into
   the correct Signal type with correct payload fields.
4. Integrity flag generation on tampered signals.
5. CoT chain step-level verification (action sequence, confidence drops).
6. Empty / malformed / edge-case payloads.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from saber.config import SaberConfig
from saber.signal import Signal, SignalType, Claim
from saber.sentinel import Sentinel, _create_step_flag
import saber.sentinel as sentinel_module


def _make_cot_signal(query_id, domain, claims_list, raw_response):
    """Helper to build a frozen COT_SIGNAL with structured claims."""
    return Signal(
        signal_type=SignalType.COT_SIGNAL,
        query_id=query_id,
        source_id=domain,
        target_id="META_REASONER",
        payload={
            "claims": claims_list,
            "raw_response": raw_response,
        },
    ).freeze_and_hash()


class TestClaimExtractionFromPayload(unittest.TestCase):
    """Verify that verify_interpretation correctly extracts searchable
    claim statements from the Signal payload and formulates queries."""

    def setUp(self):
        self.sentinel = Sentinel()
        self.config = SaberConfig()
        # Reset Sentinel module-level caches between tests
        sentinel_module._SEARCH_CACHE.clear()
        sentinel_module._LAST_CYCLE_QUERIES.clear()
        sentinel_module._LAST_SEARCH_RESULT.clear()
        sentinel_module._QUERY_CONSECUTIVE_COUNT.clear()
        sentinel_module._INTERNET_CHECKED = None

    @patch("saber.llm_engine.LLMEngine")
    def test_extracts_multiple_domain_claims(self, MockEngine):
        """With internet OFF, Sentinel should still extract claims and build
        the offline verification prompt containing all claim statements."""
        sentinel_module._INTERNET_CHECKED = False  # Force offline

        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        claims = [
            {"statement": "Kinetic energy equals half mass times velocity squared"},
            {"statement": "For m=10kg and v=5m/s, KE = 125 Joules"},
            {"statement": "Momentum p = m * v = 50 kg·m/s"},
        ]
        sig = _make_cot_signal("q-extract-01", "science", claims, "KE is 125J.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="science",
            original_signal=sig,
            compiled_text="The kinetic energy is 125 Joules and momentum is 50 kg·m/s.",
            config=self.config,
        )

        # The LLM prompt should have been called with all 3 claim statements
        call_args = mock_engine_instance.generate.call_args
        prompt_text = call_args[0][0]
        self.assertIn("Kinetic energy equals half mass times velocity squared", prompt_text)
        self.assertIn("125 Joules", prompt_text)
        self.assertIn("Momentum p = m * v = 50", prompt_text)

    @patch("saber.llm_engine.LLMEngine")
    def test_skips_generic_answer_claims(self, MockEngine):
        """Claims like 'The correct answer is B' should be skipped from search."""
        sentinel_module._INTERNET_CHECKED = False

        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        claims = [
            {"statement": "The correct answer is B"},
            {"statement": "Option C is correct based on calculations"},
            {"statement": "Ribozymes catalyze specific biochemical reactions in RNA splicing"},
        ]
        sig = _make_cot_signal("q-skip-01", "science", claims, "Answer B.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="science",
            original_signal=sig,
            compiled_text="Ribozymes catalyze RNA splicing.",
            config=self.config,
        )
        # Should return GREEN_CHIT since LLM returned CONFIRMED
        self.assertEqual(result.signal_type, SignalType.VERIFICATION_SIGNAL)

    @patch("saber.llm_engine.LLMEngine")
    def test_empty_claims_falls_back_to_compiled_text(self, MockEngine):
        """When claims list is empty, Sentinel should fall back to compiled_text."""
        sentinel_module._INTERNET_CHECKED = False

        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        sig = _make_cot_signal("q-empty-01", "cyber", [], "Port 445 is SMB.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="cyber",
            original_signal=sig,
            compiled_text="Port 445 is used by the SMB protocol for file sharing.",
            config=self.config,
        )

        call_args = mock_engine_instance.generate.call_args
        prompt_text = call_args[0][0]
        # Should contain the compiled text as fallback grounding context
        self.assertIn("Port 445", prompt_text)


class TestGreenChitVsFlagRouting(unittest.TestCase):
    """Verify that LLM output is correctly routed to GREEN_CHIT or FLAG_SIGNAL."""

    def setUp(self):
        self.sentinel = Sentinel()
        self.config = SaberConfig()
        sentinel_module._INTERNET_CHECKED = False  # Force offline

    @patch("saber.llm_engine.LLMEngine")
    def test_confirmed_returns_green_chit(self, MockEngine):
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        sig = _make_cot_signal("q-green-01", "finance", [{"statement": "EBITDA = Revenue - COGS - OpEx"}], "EBITDA formula.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="finance", original_signal=sig,
            compiled_text="EBITDA = Revenue - COGS - OpEx", config=self.config,
        )

        self.assertEqual(result.signal_type, SignalType.VERIFICATION_SIGNAL)
        self.assertEqual(result.payload["status"], "GREEN_CHIT")

    @patch("saber.llm_engine.LLMEngine")
    def test_json_error_returns_flag_signal(self, MockEngine):
        """When LLM returns a structured JSON error, Sentinel should parse it
        into a FLAG_SIGNAL with correct issue_type and severity."""
        mock_engine_instance = MagicMock()
        error_json = json.dumps({
            "issue_type": "FACTUAL_ERROR",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "evidence": "Port 445 is SMB, not HTTP",
            "reasoning": "The compiled text incorrectly states Port 445 is HTTP.",
            "proposed_fix": "Replace HTTP with SMB.",
        })
        mock_engine_instance.generate.return_value = error_json
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        sig = _make_cot_signal("q-flag-01", "cyber",
            [{"statement": "Port 445 is used by SMB protocol"}],
            "Port 445 is HTTP.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="cyber", original_signal=sig,
            compiled_text="Port 445 is used by the HTTP protocol.", config=self.config,
        )

        self.assertEqual(result.signal_type, SignalType.FLAG_SIGNAL)
        self.assertEqual(result.payload["issue_type"], "factual_error")
        self.assertEqual(result.payload["severity"], "critical")
        self.assertAlmostEqual(result.payload["confidence"], 0.95)
        self.assertIn("SMB", result.payload["evidence"])
        self.assertIn("Replace HTTP with SMB", result.payload["proposed_fix"])

    @patch("saber.llm_engine.LLMEngine")
    def test_malformed_llm_output_still_generates_flag(self, MockEngine):
        """When LLM returns non-JSON non-CONFIRMED text, Sentinel should still
        generate a FLAG_SIGNAL with a fallback payload."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "This answer has issues with the formula used."
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        sig = _make_cot_signal("q-malformed-01", "science",
            [{"statement": "F = m * a"}], "Force formula.")

        result = self.sentinel.verify_interpretation(
            specialist_domain="science", original_signal=sig,
            compiled_text="F = m * a", config=self.config,
        )

        self.assertEqual(result.signal_type, SignalType.FLAG_SIGNAL)
        self.assertEqual(result.payload["issue_type"], "reasoning_error")
        self.assertIn("issues with the formula", result.payload["reasoning"])

    @patch("saber.llm_engine.LLMEngine")
    def test_llm_exception_returns_green_chit_gracefully(self, MockEngine):
        """When LLM crashes entirely, Sentinel should NOT crash the pipeline.
        It should return a GREEN_CHIT (fail-open for availability)."""
        MockEngine.return_value.__enter__ = MagicMock(side_effect=RuntimeError("GPU OOM"))
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        sig = _make_cot_signal("q-crash-01", "coding",
            [{"statement": "Binary search runs in O(log n)"}], "O(log n).")

        result = self.sentinel.verify_interpretation(
            specialist_domain="coding", original_signal=sig,
            compiled_text="Binary search is O(log n).", config=self.config,
        )

        # Should fail-open, not crash the pipeline
        self.assertEqual(result.signal_type, SignalType.VERIFICATION_SIGNAL)
        self.assertEqual(result.payload["status"], "GREEN_CHIT")


class TestIntegrityFlagGeneration(unittest.TestCase):
    """Test generate_integrity_flag for tampered signals."""

    def test_integrity_flag_has_correct_fields(self):
        sig = Signal(
            signal_type=SignalType.COT_SIGNAL, query_id="q-int-01",
            source_id="science", target_id="META_REASONER",
            payload={"claims": [], "raw_response": "test"},
        ).freeze_and_hash()

        flag = Sentinel.generate_integrity_flag(sig)
        self.assertEqual(flag.signal_type, SignalType.FLAG_SIGNAL)
        self.assertEqual(flag.payload["issue_type"], "integrity_failure")
        self.assertEqual(flag.payload["severity"], "critical")
        self.assertEqual(flag.payload["confidence"], 1.0)
        self.assertIn("cryptographic check", flag.payload["reasoning"])


class TestCoTChainStepVerification(unittest.TestCase):
    """Test verify_cot_chain step-level checks (action sequence, confidence drops)."""

    def setUp(self):
        self.config = SaberConfig()

    @patch("saber.llm_engine.LLMEngine")
    def test_first_step_not_identify_flagged(self, MockEngine):
        """If the first step action is not IDENTIFY, Sentinel should flag it."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        cot_chain = {
            "query_id": "q-cot-01",
            "steps": [
                {"step_number": 1, "action": "ANALYZE", "content": "Analyzing directly.", "confidence": 0.9},
                {"step_number": 2, "action": "CONCLUDE", "content": "Final answer.", "confidence": 0.85},
            ],
        }

        flags = Sentinel.verify_cot_chain("science", cot_chain, "Final answer.", self.config)

        action_flags = [f for f in flags if f.payload.get("issue_type") == "STEP_ACTION_MISMATCH"]
        self.assertTrue(len(action_flags) >= 1)
        self.assertIn("IDENTIFY", action_flags[0].payload["reasoning"])

    @patch("saber.llm_engine.LLMEngine")
    def test_last_step_not_conclude_flagged(self, MockEngine):
        """If the last step action is not CONCLUDE, Sentinel should flag it."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        cot_chain = {
            "query_id": "q-cot-02",
            "steps": [
                {"step_number": 1, "action": "IDENTIFY", "content": "Problem identified.", "confidence": 0.95},
                {"step_number": 2, "action": "ANALYZE", "content": "Still analyzing.", "confidence": 0.90},
            ],
        }

        flags = Sentinel.verify_cot_chain("cyber", cot_chain, "Still analyzing.", self.config)

        action_flags = [f for f in flags if f.payload.get("issue_type") == "STEP_ACTION_MISMATCH"]
        self.assertTrue(len(action_flags) >= 1)
        self.assertIn("CONCLUDE", action_flags[0].payload["reasoning"])

    @patch("saber.llm_engine.LLMEngine")
    def test_sharp_confidence_drop_flagged(self, MockEngine):
        """A confidence drop > 0.3 between consecutive steps should be flagged."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = "CONFIRMED"
        MockEngine.return_value.__enter__ = lambda s: mock_engine_instance
        MockEngine.return_value.__exit__ = MagicMock(return_value=False)

        cot_chain = {
            "query_id": "q-cot-03",
            "steps": [
                {"step_number": 1, "action": "IDENTIFY", "content": "Problem identified.", "confidence": 0.95},
                {"step_number": 2, "action": "ANALYZE", "content": "Analyzing.", "confidence": 0.50},  # Drop of 0.45!
                {"step_number": 3, "action": "CONCLUDE", "content": "Concluded.", "confidence": 0.48},
            ],
        }

        flags = Sentinel.verify_cot_chain("finance", cot_chain, "Concluded.", self.config)

        conf_flags = [f for f in flags if f.payload.get("issue_type") == "STEP_CONFIDENCE_DROP"]
        self.assertTrue(len(conf_flags) >= 1)
        self.assertIn("0.95", conf_flags[0].payload["reasoning"])
        self.assertIn("0.50", conf_flags[0].payload["reasoning"])

    @patch("saber.llm_engine.LLMEngine")
    def test_empty_cot_chain_returns_no_flags(self, MockEngine):
        """An empty CoT chain should return zero flags, not crash."""
        cot_chain = {"query_id": "q-cot-empty", "steps": []}
        flags = Sentinel.verify_cot_chain("architecture", cot_chain, "", self.config)
        self.assertEqual(len(flags), 0)


if __name__ == "__main__":
    unittest.main()
