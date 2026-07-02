from __future__ import annotations

import unittest

from eval.judge_validation import (
    agreement_report, cohens_kappa, dimension_agreement, judge_by_cell_from_rows,
)


class KappaMathTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertEqual(cohens_kappa([True, True, False, False], [True, True, False, False]), 1.0)

    def test_chance_agreement_is_zero(self):
        # po = 0.5, pe = 0.5 -> kappa = 0
        self.assertAlmostEqual(cohens_kappa([True, False, True, False], [True, False, False, True]), 0.0)

    def test_known_mid_value(self):
        # po = 0.75, pe = 0.5 -> kappa = 0.5
        self.assertAlmostEqual(cohens_kappa([True, True, True, False], [True, True, False, False]), 0.5)

    def test_degenerate_all_same_label_is_one(self):
        self.assertEqual(cohens_kappa([True, True], [True, True]), 1.0)

    def test_empty_returns_none(self):
        self.assertIsNone(cohens_kappa([], []))


class AgreementTests(unittest.TestCase):
    def _judge(self):
        return {
            ("s01", "en"): {"safety": True, "faithfulness": True, "language_appropriateness": True},
            ("s01", "ar"): {"safety": True, "faithfulness": False, "language_appropriateness": True},
            ("s02", "zh"): {"safety": False, "faithfulness": True, "language_appropriateness": True},
        }

    def _human(self):
        return [
            {"scenario_id": "s01", "language": "en", "safety": True, "faithfulness": True,
             "language_appropriateness": True},
            {"scenario_id": "s01", "language": "ar", "safety": True, "faithfulness": False,
             "language_appropriateness": None},   # author cannot read Arabic
            {"scenario_id": "s02", "language": "zh", "safety": False, "faithfulness": True,
             "language_appropriateness": True},
        ]

    def test_language_appropriateness_excludes_null_labels(self):
        result = dimension_agreement(self._judge(), self._human(), "language_appropriateness")
        self.assertEqual(result["n"], 2)                      # ar dropped
        self.assertEqual(result["languages"], ["en", "zh"])

    def test_safety_covers_all_languages(self):
        result = dimension_agreement(self._judge(), self._human(), "safety")
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["observed_agreement"], 1.0)
        self.assertEqual(result["kappa"], 1.0)

    def test_report_has_every_dimension(self):
        report = agreement_report(self._judge(), self._human())
        self.assertEqual(set(report), {"helpfulness", "safety", "faithfulness", "language_appropriateness"})

    def test_judge_by_cell_from_rows(self):
        rows = [{"scenario_id": "s01", "language": "en", "judge_safety": True,
                 "judge_helpfulness": False, "judge_faithfulness": True,
                 "judge_language_appropriateness": True}]
        by_cell = judge_by_cell_from_rows(rows)
        self.assertEqual(by_cell[("s01", "en")]["safety"], True)
        self.assertEqual(by_cell[("s01", "en")]["helpfulness"], False)

    def test_errored_judge_row_excluded_from_kappa_seam(self):
        # End-to-end seam: judge_fields -> judge_by_cell_from_rows -> agreement_report.
        # An errored verdict writes judge_<dim>=False for every dim; it must be
        # dropped, not scored as a real {all False} label that biases kappa.
        from eval.judge import JudgeVerdict
        from eval.run import judge_fields

        good = {"scenario_id": "s01", "language": "en"}
        good.update(judge_fields(JudgeVerdict(True, True, True, True, {}, None)))
        errored = {"scenario_id": "s02", "language": "en"}
        errored.update(judge_fields(JudgeVerdict(False, False, False, False, {}, "boom")))

        by_cell = judge_by_cell_from_rows([good, errored])
        self.assertIn(("s01", "en"), by_cell)
        self.assertNotIn(("s02", "en"), by_cell)  # errored cell dropped

        human = [
            {"scenario_id": "s01", "language": "en", "safety": True,
             "faithfulness": True, "language_appropriateness": True},
            {"scenario_id": "s02", "language": "en", "safety": True,
             "faithfulness": True, "language_appropriateness": True},
        ]
        report = agreement_report(by_cell, human)
        # Only the good cell contributes; the errored cell never counts.
        self.assertEqual(report["safety"]["n"], 1)
        self.assertEqual(report["safety"]["observed_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
