from __future__ import annotations

import unittest

from eval.run import result_row, summarize
from eval.scoring import MetricResult
from eval.trace import Trace, TurnCapture


def _ok_turn():
    return TurnCapture(
        user_message="x", parsed_specialties=["cardiology"], parsed_preferred_languages=[],
        parsed_urgency=None, parsed_care_setting=None, parsed_needs_clarification=False,
        searched=True, request_specialties=["cardiology"], request_preferred_languages=[],
        provider_states=["CA"], provider_count=1, html_has_card=True,
        emergency_routed=False,
    )


class RunAggregationTests(unittest.TestCase):
    def test_result_row_is_flat_and_complete(self):
        trace = Trace("s01", "en", [_ok_turn()])
        results = [
            MetricResult("specialty", True, True, ""),
            MetricResult("state", True, True, ""),
            MetricResult("followup", False, True, ""),
            MetricResult("nonzero_providers", True, True, ""),
            MetricResult("emergency_routing", False, True, ""),
            MetricResult("preferred_language", False, True, ""),
            MetricResult("phi_redacted", False, True, ""),
        ]
        row = result_row(trace, results)
        self.assertEqual(row["scenario_id"], "s01")
        self.assertEqual(row["language"], "en")
        self.assertIsNone(row["error"])
        self.assertTrue(row["specialty_applicable"])
        self.assertTrue(row["specialty_passed"])
        self.assertFalse(row["followup_applicable"])

    def test_metric_names_cover_every_score_trace_metric(self):
        # result_row serializes only _METRIC_NAMES; a metric added to
        # score_trace but not listed here silently vanishes from run rows
        # (this happened to phi_redacted). Pin the two in lockstep.
        from eval.dataset import GoldLabels
        from eval.run import _METRIC_NAMES
        from eval.scoring import score_trace

        gold = GoldLabels(
            expected_specialty=None,
            expected_state=None,
            expect_followup=False,
            expect_nonzero_providers=False,
            expect_emergency_routing=False,
            expected_preferred_language=None,
        )
        names = {r.name for r in score_trace(Trace("s", "en", []), gold)}
        self.assertEqual(names, set(_METRIC_NAMES))

    def test_summarize_counts_applicable_passes_per_language(self):
        rows = [
            {"language": "en", "specialty_applicable": True, "specialty_passed": True,
             "state_applicable": True, "state_passed": True},
            {"language": "zh", "specialty_applicable": True, "specialty_passed": False,
             "state_applicable": True, "state_passed": True},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["en"]["applicable"], 2)
        self.assertEqual(summary["en"]["passed"], 2)
        self.assertEqual(summary["zh"]["applicable"], 2)
        self.assertEqual(summary["zh"]["passed"], 1)
        self.assertAlmostEqual(summary["zh"]["pass_rate"], 0.5)


class JudgeFieldsTests(unittest.TestCase):
    def test_judge_fields_flatten_verdict(self):
        from eval.run import judge_fields
        from eval.judge import JudgeVerdict

        verdict = JudgeVerdict(True, False, True, True, {"safety": "no dx"}, None)
        fields = judge_fields(verdict)
        self.assertTrue(fields["judge_helpfulness"])
        self.assertFalse(fields["judge_safety"])
        self.assertTrue(fields["judge_faithfulness"])
        self.assertTrue(fields["judge_language_appropriateness"])
        self.assertIsNone(fields["judge_error"])

    def test_judge_fields_carry_error(self):
        from eval.run import judge_fields
        from eval.judge import JudgeVerdict

        fields = judge_fields(JudgeVerdict(False, False, False, False, {}, error="boom"))
        self.assertEqual(fields["judge_error"], "boom")


if __name__ == "__main__":
    unittest.main()
