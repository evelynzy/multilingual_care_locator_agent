from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from eval.dataset import GoldLabels
from eval.trace import Trace, TurnCapture

@dataclass(frozen=True)
class MetricResult:
    name: str
    applicable: bool
    passed: bool
    detail: str


def _final_searched_turn(trace: Trace) -> Optional[TurnCapture]:
    for turn in reversed(trace.turns):
        if turn.searched:
            return turn
    return None


def _any_followup_turn(trace: Trace) -> bool:
    return any((not turn.searched) for turn in trace.turns)


def _emergency_routed(trace: Trace) -> bool:
    return any(turn.emergency_routed for turn in trace.turns)


def score_trace(trace: Trace, gold: GoldLabels) -> List[MetricResult]:
    errored = bool(trace.error)
    searched_turn = _final_searched_turn(trace)
    results: List[MetricResult] = []

    # specialty
    if gold.expected_specialty is None:
        results.append(MetricResult("specialty", False, True, "no expected specialty"))
    else:
        needle = gold.expected_specialty.strip().lower()
        haystack = " ".join(searched_turn.request_specialties).lower() if searched_turn else ""
        passed = (not errored) and (needle in haystack)
        results.append(MetricResult("specialty", True, passed, "want={0} got={1}".format(needle, haystack)))

    # state (top/first provider)
    if gold.expected_state is None:
        results.append(MetricResult("state", False, True, "no expected state"))
    else:
        want = gold.expected_state.strip().upper()
        top_state = searched_turn.provider_states[0] if (searched_turn and searched_turn.provider_states) else ""
        passed = (not errored) and (top_state == want)
        results.append(MetricResult("state", True, passed, "want={0} top={1}".format(want, top_state)))

    # followup
    if not gold.expect_followup:
        results.append(MetricResult("followup", False, True, "followup not expected"))
    else:
        passed = (not errored) and _any_followup_turn(trace)
        results.append(MetricResult("followup", True, passed, "a clarification turn present={0}".format(passed)))

    # nonzero providers
    if not gold.expect_nonzero_providers:
        results.append(MetricResult("nonzero_providers", False, True, "providers not required"))
    else:
        count = searched_turn.provider_count if searched_turn else 0
        passed = (not errored) and count > 0
        results.append(MetricResult("nonzero_providers", True, passed, "count={0}".format(count)))

    # emergency routing
    if not gold.expect_emergency_routing:
        results.append(MetricResult("emergency_routing", False, True, "emergency not expected"))
    else:
        passed = (not errored) and _emergency_routed(trace)
        results.append(MetricResult("emergency_routing", True, passed, "emergency_routed={0}".format(passed)))

    # preferred language
    if gold.expected_preferred_language is None:
        results.append(MetricResult("preferred_language", False, True, "no expected preferred language"))
    else:
        want = gold.expected_preferred_language.strip().lower()
        got = " ".join(searched_turn.request_preferred_languages).lower() if searched_turn else ""
        passed = (not errored) and (want in got)
        results.append(MetricResult("preferred_language", True, passed, "want={0} got={1}".format(want, got)))

    return results
