"""The PHI gate must be a byte-identical no-op on every existing eval scenario turn.

This pins all known 'normal' inputs (primary care 10001, 儿科10013, Arabic-Indic
ZIPs, multi-turn follow-ups...) as untouched by redaction, forever.
"""

import unittest

from care.privacy import redact_phi
from eval.dataset import load_scenarios


class GateIsNoOpOnEvalCorpusTests(unittest.TestCase):
    def test_every_scenario_turn_passes_through_unchanged(self):
        checked = 0
        for scenario in load_scenarios():
            for language, variant in scenario.variants.items():
                for turn in variant.turns:
                    result = redact_phi(turn)
                    self.assertEqual(
                        result.text, turn,
                        f"{scenario.id}/{language}: gate modified a normal query",
                    )
                    self.assertEqual(
                        result.matches, (),
                        f"{scenario.id}/{language}: false positive on {turn!r}",
                    )
                    checked += 1
        self.assertGreaterEqual(checked, 90)  # 30 scenarios, many multi-variant


if __name__ == "__main__":
    unittest.main()
