# -*- coding: utf-8 -*-
"""tests/test_rewards.py

Unit tests for SABER GRPO modular reward functions.
"""

import unittest
from saber.training.rewards import definitive_reward_function, open_ended_reward_function


class TestGRPORewardFunctions(unittest.TestCase):

    def test_definitive_reward_format_and_outcome(self):
        prompts = ["Question: What is 2+2?\nOptions:\nA: 3\nB: 4\nC: 5\nD: 6"]
        completions = ["REASONING:\nTo solve what is 2+2, we add the two integers 2 and 2 together step by step to obtain 4, which corresponds to option B.\n\nANSWER: B"]
        expected = ["B"]

        rewards = definitive_reward_function(prompts, completions, expected, domain="science")
        self.assertEqual(len(rewards), 1)
        # Should get +1.0 (format) + +2.0 (outcome) = 3.0
        self.assertGreaterEqual(rewards[0], 3.0)

    def test_open_ended_code_reward(self):
        prompts = ["Write a function to return square of a number."]
        completions = ["```python\ndef square(x):\n    return x * x\n```"]

        rewards = open_ended_reward_function(prompts, completions, domain="coding")
        self.assertEqual(len(rewards), 1)
        # Should get +2.0 for valid code compilation
        self.assertGreaterEqual(rewards[0], 2.0)


if __name__ == "__main__":
    unittest.main()
