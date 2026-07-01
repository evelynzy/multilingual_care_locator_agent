from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from eval.dataset import Scenario


@dataclass(frozen=True)
class TurnCapture:
    user_message: str
    parsed_specialties: List[str]
    parsed_preferred_languages: List[str]
    parsed_urgency: Optional[str]
    parsed_care_setting: Optional[str]
    parsed_needs_clarification: bool
    searched: bool
    request_specialties: List[str]
    request_preferred_languages: List[str]
    provider_states: List[str]
    provider_count: int
    html_has_card: bool
    emergency_routed: bool


@dataclass(frozen=True)
class Trace:
    scenario_id: str
    language: str
    turns: List[TurnCapture]
    error: Optional[str] = None


def trace_to_dict(trace: Trace) -> dict:
    return asdict(trace)


def trace_from_dict(data: dict) -> Trace:
    turns = [TurnCapture(**turn) for turn in data["turns"]]
    return Trace(
        scenario_id=data["scenario_id"],
        language=data["language"],
        turns=turns,
        error=data.get("error"),
    )


def _cache_path(cache_dir: str, scenario_id: str, language: str, turns: Tuple[str, ...], model: str) -> str:
    key_source = json.dumps([scenario_id, language, list(turns), model], ensure_ascii=False)
    digest = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, "{0}.{1}.{2}.json".format(scenario_id, language, digest))


_STATE_RE = re.compile(r"\b([A-Za-z]{2})\s+\d{5}(?:-\d{4})?\b")


def _provider_state(provider) -> str:
    """Best-effort 2-letter state code.

    Uses the structured field when present, else parses it from the address:
    the ClinicalTables source leaves state/city unpopulated and embeds them in
    the address string (e.g. '710 LAWRENCE EXPY, SANTA CLARA, CA 95051').
    """
    state = (getattr(provider, "state", None) or "").strip().upper()
    if state:
        return state
    for field in (getattr(provider, "address", None), getattr(provider, "location_summary", None)):
        if field:
            match = _STATE_RE.search(field)
            if match:
                return match.group(1).upper()
    return ""


def _capture_turn(user_message: str, agent, html: str) -> TurnCapture:
    parsed = agent.last_parsed_query
    request = agent.provider_search_service.last_request
    response = agent.provider_search_service.last_response

    provider_states: List[str] = []
    provider_count = 0
    if response is not None:
        provider_count = len(response.provider_results)
        provider_states = [
            _provider_state(r.provider)
            for r in response.provider_results
        ]

    return TurnCapture(
        user_message=user_message,
        parsed_specialties=list(parsed.specialties) if parsed else [],
        parsed_preferred_languages=list(parsed.preferred_languages) if parsed else [],
        parsed_urgency=(parsed.urgency if parsed else None),
        parsed_care_setting=(parsed.care_setting if parsed else None),
        parsed_needs_clarification=(bool(parsed.needs_clarification) if parsed else False),
        searched=request is not None,
        request_specialties=list(request.specialties) if request is not None else [],
        request_preferred_languages=list(request.preferred_languages) if request is not None else [],
        provider_states=provider_states,
        provider_count=provider_count,
        html_has_card=("provider-card" in (html or "")),
        emergency_routed=(getattr(agent, "last_navigation_mode", None) == "emergency"),
    )


def run_trace(
    scenario: Scenario,
    language: str,
    agent,
    client,
    settings: dict,
    cache_dir: str = "eval/.cache",
    use_cache: bool = True,
) -> Trace:
    variant = scenario.variants[language]
    turns_tuple = tuple(variant.turns)
    model = settings.get("model_id", "")
    path = _cache_path(cache_dir, scenario.id, language, turns_tuple, model)

    if use_cache and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return trace_from_dict(json.load(handle))

    history: List[dict] = []
    captures: List[TurnCapture] = []
    error: Optional[str] = None
    try:
        for user_message in variant.turns:
            agent.reset_capture()
            html = agent.handle_request(
                client,
                user_message,
                history,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
            )
            turn = _capture_turn(user_message, agent, html)
            captures.append(turn)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": html})
    except Exception as exc:  # noqa: BLE001 - record, do not crash the matrix
        error = "{0}: {1}".format(type(exc).__name__, exc)

    trace = Trace(scenario_id=scenario.id, language=language, turns=captures, error=error)

    # Do not cache errored traces: a transient failure (rate limit, timeout)
    # must not permanently poison a scenario's cache entry.
    if use_cache and error is None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(trace_to_dict(trace), handle, ensure_ascii=False, indent=2)
    return trace
