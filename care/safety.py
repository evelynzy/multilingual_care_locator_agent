"""Required trust/safety guidance (7 languages) and emergency signal detection."""
from __future__ import annotations

import re
from typing import Optional

from care.language import (
    _is_unknown_response_language,
    _lookup_language_alias,
    _normalize_response_language,
)
from care.locales_loader import load_locales as _load_locales

_REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE = {
    "english": (
        "Important safety and trust notes:\n"
        "- This tool supports care navigation only and does not diagnose, prescribe, or replace licensed medical advice.\n"
        "- Directory matches are informational, not referrals, endorsements, or guarantees of clinical fit.\n"
        "- Insurance/network participation, referral requirements, new-patient availability, location, and appointment availability are not verified unless the source explicitly says so. Call the provider and insurer to confirm before seeking care.\n"
        "- Do not share personal health information such as full names, addresses, Social Security numbers, or medical record numbers.\n"
        "- If symptoms are severe or life-threatening, call emergency services (911 in the U.S.) or go to the nearest emergency room."
    ),
}

# Localized footers come from the committed locale files, each carrying its
# own localized "auto-translated from English" mark as a final bullet line —
# honest labeling for machine translation the owner cannot personally verify.
for _language_key, _locale in _load_locales().items():
    _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[_language_key] = (
        _locale["trust_guidance"].rstrip()
        + "\n- "
        + _locale["auto_translated_mark"].strip()
    )

_REQUIRED_TRUST_GUIDANCE = _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE["english"]

_EMERGENCY_PATTERNS = (
    "emergency",
    "life-threatening",
    "life threatening",
    "chest pain",
    "trouble breathing",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "cant breathe",
    "shortness of breath",
    "stroke",
    "heart attack",
    "overdose",
    "seizure",
    "anaphylaxis",
    "weakness on one side",
    "numbness on one side",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "suicide",
)

_EMERGENCY_URGENCY_VALUES = {
    "emergency",
    "emergent",
    "life-threatening",
    "life threatening",
    "critical",
}


def _get_prewritten_required_trust_guidance(response_language: Optional[str]) -> Optional[str]:
    if _is_unknown_response_language(response_language):
        return _REQUIRED_TRUST_GUIDANCE

    language_key = _lookup_language_alias(_normalize_response_language(response_language))
    if language_key is None:
        return None
    return _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[language_key]


class SafetyMixin:
    # ------------------------------------------------------------------
    def _contains_emergency_signal(self, text: str) -> bool:
        if re.search(r"(?<!\d)911(?!\d)", text):
            return True
        if re.search(r"\b9[\s-]*1[\s-]*1\b", text):
            return True
        return self._contains_any(text, _EMERGENCY_PATTERNS)

    # ------------------------------------------------------------------
    def _query_signals_emergency(self, query: ParsedCareQuery) -> bool:
        if (query.care_setting or "").strip().lower() == "emergency":
            return True
        return (query.urgency or "").strip().lower() in _EMERGENCY_URGENCY_VALUES

    # ------------------------------------------------------------------
    def _append_required_trust_guidance(
        self,
        content: str,
        response_language: Optional[str] = None,
    ) -> str:
        trust_guidance = _get_prewritten_required_trust_guidance(response_language)
        if trust_guidance is None:
            trust_guidance = _REQUIRED_TRUST_GUIDANCE

        if any(note in content for note in _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE.values()):
            return content
        if trust_guidance in content:
            return content
        return f"{content}\n\n{trust_guidance}"
