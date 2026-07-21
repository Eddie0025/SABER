# -*- coding: utf-8 -*-
"""tests/test_sentinel.py

Unit tests for SABER Sentinel verification kernel.
"""

import unittest
from saber.signal import Signal, SignalType
from saber.sentinel import Sentinel


class TestSentinelKernel(unittest.TestCase):

    def test_signal_integrity_verification(self):
        sig = Signal(
            signal_type=SignalType.TASK_SIGNAL,
            query_id="query-sentinel-01",
            source_id="ORCHESTRATOR",
            target_id="science",
            payload={"objective": "Test objective"}
        ).freeze_and_hash()

        self.assertTrue(Sentinel.verify_signal_integrity(sig))
        sig.payload["objective"] = "Tampered objective"
        self.assertFalse(Sentinel.verify_signal_integrity(sig))

    def test_verification_routing(self):
        route_cyber = Sentinel.get_verification_route("cyber")
        self.assertEqual(route_cyber.get("technical_accuracy"), "cyber")
        self.assertEqual(route_cyber.get("logical_reasoning"), "science")

        route_science = Sentinel.get_verification_route("science")
        self.assertEqual(route_science.get("factual_accuracy"), "science")


if __name__ == "__main__":
    unittest.main()
