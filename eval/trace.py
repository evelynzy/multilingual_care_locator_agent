from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from eval.dataset import Scenario

_TRACE_SCHEMA_VERSION = "v2-judge"


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
    rendered_text: str = ""
    provider_details: List[dict] = field(default_factory=list)
    # Post-gate user text handed to the intent LLM call (message + history user
    # turns), recorded by TracingAgent; empty for traces cached before this field.
    llm_input_texts: List[str] = field(default_factory=list)


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
    key_source = json.dumps(
        [scenario_id, language, list(turns), model, _TRACE_SCHEMA_VERSION], ensure_ascii=False
    )
    digest = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, "{0}.{1}.{2}.json".format(scenario_id, language, digest))


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(raw: str) -> str:
    """Flatten rendered HTML to the visible text a human evaluator would read."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


# ZIP may be 5 digits, hyphenated ZIP+4, or — in NPPES-enriched addresses —
# the ZIP+4 concatenated without a hyphen ('CA 981015173').
_STATE_RE = re.compile(r"\b([A-Za-z]{2})\s+\d{5}(?:-?\d{4})?\b")


def _provider_state(provider) -> str:
    """Best-effort 2-letter state code.

    Order: structured .state field; the raw ClinicalTables practice-address
    state (present on NPPES-enriched records); finally parsed from the
    address string (e.g. '710 LAWRENCE EXPY, SANTA CLARA, CA 98101').
    """
    state = (getattr(provider, "state", None) or "").strip().upper()
    if state:
        return state
    raw = getattr(provider, "raw", None)
    if isinstance(raw, dict):
        raw_state = str(raw.get("addr_practice.state") or "").strip().upper()
        if raw_state:
            return raw_state
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
    provider_details: List[dict] = []
    provider_count = 0
    if response is not None:
        provider_count = len(response.provider_results)
        for r in response.provider_results:
            provider = r.provider
            provider_states.append(_provider_state(provider))
            provider_details.append({
                "name": (getattr(provider, "name", "") or ""),
                "specialties": list(getattr(provider, "specialties", ()) or ()),
                "languages": list(getattr(provider, "languages", ()) or ()),
                "state": _provider_state(provider),
                "city": (getattr(provider, "city", None) or ""),
            })

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
        rendered_text=_html_to_text(html),
        provider_details=provider_details,
        llm_input_texts=list(getattr(agent, "captured_llm_inputs", []) or []),
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
