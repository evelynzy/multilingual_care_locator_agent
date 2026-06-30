from __future__ import annotations

import unittest

from eval.dataset import GoldLabels, LANGUAGES, Scenario, load_scenarios


class DatasetTests(unittest.TestCase):
    def test_loads_fifteen_scenarios(self):
        scenarios = load_scenarios()
        self.assertEqual(len(scenarios), 15)
        self.assertTrue(all(isinstance(s, Scenario) for s in scenarios))

    def test_ids_are_unique(self):
        ids = [s.id for s in load_scenarios()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_scenario_has_english_seed(self):
        for s in load_scenarios():
            self.assertIn("en", s.variants)
            self.assertEqual(s.variants["en"].source, "english_seed")
            self.assertGreaterEqual(len(s.variants["en"].turns), 1)

    def test_gold_labels_typed(self):
        gold = load_scenarios()[0].gold
        self.assertIsInstance(gold, GoldLabels)
        self.assertIsInstance(gold.expect_followup, bool)
        self.assertIsInstance(gold.expect_nonzero_providers, bool)

    def test_languages_constant(self):
        self.assertEqual(LANGUAGES, ("en", "zh", "es", "ar", "ko"))

    def test_emergency_scenarios_present(self):
        golds = [s.gold for s in load_scenarios()]
        self.assertTrue(any(g.expect_emergency_routing for g in golds))

    def test_rejects_unknown_language_key(self):
        with self.assertRaises(ValueError):
            load_scenarios("tests/eval/fixtures/bad_scenarios.json")

    def test_rejects_missing_required_key(self):
        # A scenario missing the "gold" key must surface as ValueError, not KeyError.
        with self.assertRaises(ValueError):
            load_scenarios("tests/eval/fixtures/missing_key_scenarios.json")


if __name__ == "__main__":
    unittest.main()
