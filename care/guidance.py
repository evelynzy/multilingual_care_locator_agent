"""Navigation modes, care-setting classification, clarification question bank, referral guidance."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from provider_search.specialty_families import (
    normalize_query_specialty_family_id,
)


_URGENT_PATTERNS = (
    "same-day",
    "same day",
    "today",
    "urgent",
    "asap",
    "right away",
    "immediate",
)

_ROUTINE_PATTERNS = (
    "routine",
    "ongoing",
    "follow-up",
    "follow up",
    "checkup",
    "check-up",
    "annual",
    "preventive",
    "primary care",
    "pcp",
)

_SPECIALIST_PATTERNS = (
    "specialist",
    "specialty",
    "referral",
    "cardiologist",
    "dermatologist",
    "neurologist",
    "psychiatrist",
    "ent",
    "orthopedic",
    "orthopaedic",
    "gastroenterologist",
    "endocrinologist",
    "urologist",
    "oncologist",
)
_GENERIC_SPECIALIST_PATTERNS = (
    "specialist",
    "specialty",
    "referral",
)

_INSURANCE_PATTERNS = (
    "insurance",
    "covered",
    "coverage",
    "plan",
    "hmo",
    "ppo",
    "pos",
    "medicaid",
    "medicare",
    "chip",
)

_LANGUAGE_PATTERNS = (
    "speaks",
    "speaker",
    "bilingual",
    "language",
    "spanish",
    "mandarin",
    "cantonese",
    "french",
    "arabic",
    "hindi",
)

_REFERRAL_PATTERNS = (
    "referral",
    "referrals",
    "referred",
    "need a referral",
    "needs a referral",
)

_PLAN_TYPE_PATTERNS = ("hmo", "ppo", "pos")


class GuidanceMixin:
    # ------------------------------------------------------------------
    def _build_navigation_guidance(
        self, query: ParsedCareQuery, message: str
    ) -> Dict[str, Any]:
        request_text = self._combined_request_text(query, message)
        care_setting = self._classify_care_setting(query, request_text)

        follow_up_questions: List[str] = []

        if query.medical_need and not self._has_specific_location(
            query,
            request_text,
            raw_message=message,
        ):
            follow_up_questions.append(
                "What city and state or ZIP code should I search?"
            )

        if query.medical_need and not self._has_clear_care_need(query, request_text):
            follow_up_questions.append(
                "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?"
            )

        insurance_requested = self._contains_any(request_text, _INSURANCE_PATTERNS)
        if insurance_requested and not query.insurance:
            follow_up_questions.append(
                "Which insurance plan should I use when tailoring listed-insurance guidance?"
            )

        language_requested = self._contains_any(request_text, _LANGUAGE_PATTERNS)
        if language_requested and not query.preferred_languages:
            follow_up_questions.append(
                "Do you want a provider who speaks a specific language?"
            )

        specialist_plan_guidance = None
        if care_setting == "specialist":
            specialist_plan_guidance = (
                "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents."
            )
            if (
                self._contains_any(request_text, _REFERRAL_PATTERNS)
                and not self._has_plan_type(query, request_text)
            ):
                follow_up_questions.append(
                    "What plan type do you have, if you want me to tailor referral guidance (for example HMO, PPO, or POS)?"
                )

        if care_setting == "emergency":
            return {
                "mode": "emergency",
                "care_setting_guidance": (
                    "If symptoms are severe or life-threatening, call emergency services now or go to the nearest emergency room."
                ),
                "follow_up_questions": [],
                "specialist_plan_guidance": None,
                "location_only": False,
            }

        if not follow_up_questions:
            guidance_parts = []
            if care_setting == "urgent_care":
                guidance_parts.append(
                    "For same-day, non-emergency care, urgent care is usually the best fit."
                )
            elif care_setting == "pcp":
                guidance_parts.append(
                    "For routine or ongoing care, primary care is usually the best fit."
                )
            elif care_setting == "specialist":
                guidance_parts.append(
                    "For a known specialty or referral need, a specialist is usually the right route."
                )

            return {
                "mode": "search",
                "care_setting_guidance": " ".join(guidance_parts).strip() or None,
                "follow_up_questions": [],
                "specialist_plan_guidance": specialist_plan_guidance,
                "location_only": False,
            }

        location_only = (
            len(follow_up_questions) == 1
            and follow_up_questions[0].startswith("What city and state or ZIP code")
        )

        if query.medical_need and care_setting == "specialist" and specialist_plan_guidance:
            follow_up_questions = self._dedupe_preserve_order(follow_up_questions)
        elif care_setting == "specialist" and specialist_plan_guidance:
            follow_up_questions = self._dedupe_preserve_order(follow_up_questions)

        care_setting_note = None
        if care_setting == "urgent_care":
            care_setting_note = (
                "For same-day, non-emergency care, urgent care is usually the best fit."
            )
        elif care_setting == "pcp":
            care_setting_note = (
                "For routine or ongoing care, primary care is usually the best fit."
            )
        elif care_setting == "specialist":
            care_setting_note = (
                "For a known specialty or referral need, a specialist is usually the right route."
            )

        return {
            "mode": "clarification",
            "care_setting_guidance": care_setting_note,
            "follow_up_questions": self._dedupe_preserve_order(follow_up_questions),
            "specialist_plan_guidance": specialist_plan_guidance,
            "location_only": location_only,
        }

    # ------------------------------------------------------------------
    def _combined_request_text(self, query: ParsedCareQuery, message: str) -> str:
        parts = [
            message,
            query.summary,
            " ".join(query.specialties),
            " ".join(query.insurance),
            " ".join(query.preferred_languages),
            " ".join(query.keywords),
            query.patient_context or "",
            query.care_setting or "",
            query.urgency or "",
        ]
        return " ".join(part for part in parts if part).lower()

    # ------------------------------------------------------------------
    def _contains_any(self, text: str, patterns: Tuple[str, ...]) -> bool:
        normalized_text = text.lower()
        return any(
            self._contains_phrase(normalized_text, pattern)
            for pattern in patterns
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        normalized_phrase = phrase.lower().strip()
        if not normalized_phrase:
            return False

        escaped_parts = [
            re.escape(part)
            for part in re.split(r"\s+", normalized_phrase)
            if part
        ]
        if not escaped_parts:
            return False

        pattern = (
            r"(?<![A-Za-z0-9])"
            + r"[\s-]+".join(escaped_parts)
            + r"(?![A-Za-z0-9])"
        )
        return re.search(pattern, text) is not None

    # ------------------------------------------------------------------
    def _has_plan_type(self, query: ParsedCareQuery, text: str) -> bool:
        if self._contains_any(text, _PLAN_TYPE_PATTERNS):
            return True
        return any(self._contains_any(item.lower(), _PLAN_TYPE_PATTERNS) for item in query.insurance)

    # ------------------------------------------------------------------
    def _has_specific_location(
        self,
        query: ParsedCareQuery,
        text: str,
        raw_message: Optional[str] = None,
    ) -> bool:
        evidence_text = raw_message if raw_message is not None else text
        has_procedure_code_intent = self._has_explicit_procedure_code_intent(
            evidence_text
        )

        if has_procedure_code_intent:
            return bool(self._trusted_location_for_procedure_intent(evidence_text))

        if query.location and self._location_has_city(query.location):
            return True
        if query.location and self._extract_zip_code(query.location):
            return True
        if self._match_city_state(text):
            return True
        return bool(self._extract_zip_code(text))

    # ------------------------------------------------------------------
    def _has_clear_care_need(self, query: ParsedCareQuery, text: str) -> bool:
        if query.specialties:
            return True
        if self._requires_specialty_clarification(query.follow_up_focus):
            return False
        if self._contains_any(text, _ROUTINE_PATTERNS + _URGENT_PATTERNS):
            return True
        if self._has_specific_specialist_intent(query, text):
            return True
        return False

    # ------------------------------------------------------------------
    def _classify_care_setting(self, query: ParsedCareQuery, text: str) -> str:
        if self._contains_emergency_signal(text) or self._query_signals_emergency(query):
            return "emergency"
        if (
            not query.specialties
            and self._requires_specialty_clarification(query.follow_up_focus)
        ):
            return "unclear"
        if self._contains_any(text, _URGENT_PATTERNS):
            return "urgent_care"
        if self._has_specific_specialist_intent(query, text):
            return "specialist"
        if self._has_primary_care_specialty(query):
            return "pcp"
        if self._contains_any(text, _ROUTINE_PATTERNS):
            return "pcp"
        if self._contains_any(text, _GENERIC_SPECIALIST_PATTERNS):
            return "unclear"
        return "unclear"

    def _has_specific_specialist_intent(
        self,
        query: ParsedCareQuery,
        text: str,
    ) -> bool:
        if self._has_non_primary_specialty(query):
            return True

        return self._contains_any(
            text,
            tuple(
                pattern
                for pattern in _SPECIALIST_PATTERNS
                if pattern not in _GENERIC_SPECIALIST_PATTERNS
            ),
        )

    def _has_primary_care_specialty(self, query: ParsedCareQuery) -> bool:
        return "primary-care" in self._query_specialty_family_ids(query.specialties)

    def _has_non_primary_specialty(self, query: ParsedCareQuery) -> bool:
        family_ids = self._query_specialty_family_ids(query.specialties)
        return any(family_id != "primary-care" for family_id in family_ids)

    def _query_specialty_family_ids(self, specialties: List[str]) -> set[str]:
        family_ids: set[str] = set()
        for specialty in specialties:
            family_id = normalize_query_specialty_family_id(specialty)
            if family_id is not None:
                family_ids.add(family_id)
        return family_ids


    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
