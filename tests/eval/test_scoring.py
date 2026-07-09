from __future__ import annotations

import unittest

from eval.dataset import GoldLabels
from eval.scoring import score_trace
from eval.trace import Trace, TurnCapture


def _turn(**overrides):
    base = dict(
        user_message="x",
        parsed_specialties=[],
        parsed_preferred_languages=[],
        parsed_urgency=None,
        parsed_care_setting=None,
        parsed_needs_clarification=False,
        searched=False,
        request_specialties=[],
        request_preferred_languages=[],
        provider_states=[],
        provider_count=0,
        html_has_card=False,
        emergency_routed=False,
    )
    base.update(overrides)
    return TurnCapture(**base)


def _gold(**overrides):
    base = dict(
        expected_specialty=None,
        expected_state=None,
        expect_followup=False,
        expect_nonzero_providers=False,
        expect_emergency_routing=False,
        expected_preferred_language=None,
    )
    base.update(overrides)
    return GoldLabels(**base)


def _by_name(results):
    return {r.name: r for r in results}


class PhiRedactedMetricTests(unittest.TestCase):
    def test_phi_redacted_passes_when_llm_inputs_are_clean(self):
        trace = Trace("s", "en", [_turn(llm_input_texts=["ssn [REDACTED: SSN] cardiology 94110"])])
        res = _by_name(score_trace(trace, _gold(synthetic_phi="123-45-6789")))
        self.assertTrue(res["phi_redacted"].applicable)
        self.assertTrue(res["phi_redacted"].passed)

    def test_phi_redacted_fails_on_leak(self):
        trace = Trace("s", "en", [_turn(llm_input_texts=["ssn 123-45-6789 cardiology 94110"])])
        res = _by_name(score_trace(trace, _gold(synthetic_phi="123-45-6789")))
        self.assertFalse(res["phi_redacted"].passed)

    def test_phi_redacted_fails_on_native_script_leak(self):
        # synthetic_phi is stored in ASCII but the scenario turn (s32.ar) carries
        # Arabic-Indic digits: the leak check must fold before comparing.
        trace = Trace("s", "ar", [_turn(llm_input_texts=["رقم الضمان ١٢٣-٤٥-٦٧٨٩"])])
        res = _by_name(score_trace(trace, _gold(synthetic_phi="123-45-6789")))
        self.assertFalse(res["phi_redacted"].passed)

    def test_phi_redacted_not_applicable_without_synthetic_phi(self):
        res = _by_name(score_trace(Trace("s", "en", [_turn()]), _gold()))
        self.assertFalse(res["phi_redacted"].applicable)


class ScoringTests(unittest.TestCase):
    def test_specialty_matches_via_substring(self):
        trace = Trace("s", "en", [_turn(searched=True, request_specialties=["cardiology"])])
        res = _by_name(score_trace(trace, _gold(expected_specialty="cardiology")))
        self.assertTrue(res["specialty"].applicable)
        self.assertTrue(res["specialty"].passed)

    def test_specialty_fails_when_wrong(self):
        trace = Trace("s", "en", [_turn(searched=True, request_specialties=["pediatrics"])])
        res = _by_name(score_trace(trace, _gold(expected_specialty="cardiology")))
        self.assertFalse(res["specialty"].passed)

    def test_state_checks_top_provider(self):
        trace = Trace("s", "en", [_turn(searched=True, provider_count=2, provider_states=["CA", "TX"])])
        res = _by_name(score_trace(trace, _gold(expected_state="CA")))
        self.assertTrue(res["state"].passed)

    def test_state_fails_on_wrong_top(self):
        trace = Trace("s", "en", [_turn(searched=True, provider_count=2, provider_states=["TX", "CA"])])
        res = _by_name(score_trace(trace, _gold(expected_state="CA")))
        self.assertFalse(res["state"].passed)

    def test_followup_passes_when_a_turn_did_not_search(self):
        trace = Trace("s", "en", [_turn(searched=False, parsed_needs_clarification=True)])
        res = _by_name(score_trace(trace, _gold(expect_followup=True)))
        self.assertTrue(res["followup"].passed)

    def test_followup_multiturn_first_asks_then_searches(self):
        trace = Trace("s", "en", [
            _turn(searched=False, parsed_needs_clarification=True),
            _turn(searched=True, request_specialties=["cardiology"], provider_count=1, provider_states=["CA"]),
        ])
        gold = _gold(expect_followup=True, expected_specialty="cardiology", expected_state="CA", expect_nonzero_providers=True)
        res = _by_name(score_trace(trace, gold))
        self.assertTrue(res["followup"].passed)
        self.assertTrue(res["specialty"].passed)
        self.assertTrue(res["state"].passed)
        self.assertTrue(res["nonzero_providers"].passed)

    def test_nonzero_providers(self):
        trace = Trace("s", "en", [_turn(searched=True, provider_count=3)])
        res = _by_name(score_trace(trace, _gold(expect_nonzero_providers=True)))
        self.assertTrue(res["nonzero_providers"].passed)

    def test_emergency_routing_from_routed_turn(self):
        trace = Trace("s", "en", [_turn(emergency_routed=True)])
        res = _by_name(score_trace(trace, _gold(expect_emergency_routing=True)))
        self.assertTrue(res["emergency_routing"].passed)

    def test_emergency_routing_fails_when_not_routed(self):
        trace = Trace("s", "en", [_turn(emergency_routed=False)])
        res = _by_name(score_trace(trace, _gold(expect_emergency_routing=True)))
        self.assertFalse(res["emergency_routing"].passed)

    def test_preferred_language(self):
        trace = Trace("s", "en", [_turn(searched=True, request_preferred_languages=["Spanish"])])
        res = _by_name(score_trace(trace, _gold(expected_preferred_language="spanish")))
        self.assertTrue(res["preferred_language"].passed)

    def test_non_applicable_metric_is_passed_but_not_applicable(self):
        trace = Trace("s", "en", [_turn(searched=True, request_specialties=["cardiology"])])
        res = _by_name(score_trace(trace, _gold()))
        self.assertFalse(res["specialty"].applicable)
        self.assertTrue(res["specialty"].passed)

    def test_errored_trace_fails_applicable_metrics(self):
        trace = Trace("s", "en", [], error="Boom: kaboom")
        res = _by_name(score_trace(trace, _gold(expected_specialty="cardiology", expect_nonzero_providers=True)))
        self.assertFalse(res["specialty"].passed)
        self.assertFalse(res["nonzero_providers"].passed)


if __name__ == "__main__":
    unittest.main()
