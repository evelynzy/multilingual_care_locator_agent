"""Intent extraction, repair pipeline, numeric/location trust boundary, multi-turn merge."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from care.language import (
    _dominant_user_script_language,
    _is_unknown_response_language,
    _lookup_language_alias,
    _message_has_language_signal,
    _normalize_response_language,
    normalize_chat_messages,
)
from care.privacy import fold_digits
from provider_search.specialty_families import (
    QUERY_SPECIALTY_FAMILY_ALIASES_BY_ID,
    SPECIALTY_FAMILY_BY_ID,
    normalize_query_specialty_family_id,
)

logger = logging.getLogger(__name__)


_US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

_US_STATE_NAMES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


_LOCATION_NOISE_TOKENS = {
    "plan",
    "insurance",
    "coverage",
    "policy",
    "need",
    "looking",
    "search",
    "find",
    "for",
    "provider",
    "doctor",
    "dentist",
    "dental",
    "care",
    "help",
    "support",
    "area",
    "region",
    "state",
    "province",
}

_PROCEDURE_CODE_INTENT_PATTERNS = (
    "cpt",
    "procedure code",
    "procedure",
    "billing code",
)

_INTERPRET_SPECIALTY_LABEL_OVERRIDES = {
    "ent": "ENT",
    "obstetrics-gynecology": "OB/GYN",
    "psychiatry-behavioral-health": "Psychiatry",
    "oncology-hematology": "Oncology",
}

_SPECIALTY_CLARIFICATION_FOCUS = "specialty clarification"
_CHILD_CARE_AMBIGUITY_PATTERNS = (
    "child",
    "children",
    "kid",
    "kids",
    "pediatric",
    "pediatrics",
    "pediatrician",
    "child health",
)
_GENERIC_ALLERGY_AMBIGUITY_PATTERNS = (
    "allergy",
    "allergies",
)
_EXPLICIT_ALLERGY_SPECIALTY_PATTERNS = (
    "allergy immunology",
    "immunology",
)


INTERPRET_MAX_TOKENS = 1024

INTERPRET_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "care_intent",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "detected_language",
                "response_language",
                "summary",
                "medical_need",
                "location",
                "specialties",
                "insurance",
                "preferred_languages",
                "keywords",
                "patient_context",
                "care_setting",
                "urgency",
                "needs_clarification",
                "follow_up_focus",
            ],
            "properties": {
                "detected_language": {"type": "string"},
                "response_language": {"type": "string"},
                "summary": {"type": "string"},
                "medical_need": {"type": "boolean"},
                "location": {"type": ["string", "null"]},
                "specialties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specialty names in ENGLISH, translated from the user's "
                        "language (e.g. 儿科 -> pediatrics, 心脏科 -> cardiology). "
                        "The provider directory only matches English terms."
                    ),
                },
                "insurance": {"type": "array", "items": {"type": "string"}},
                "preferred_languages": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "patient_context": {"type": ["string", "null"]},
                "care_setting": {"type": ["string", "null"]},
                "urgency": {"type": ["string", "null"]},
                "needs_clarification": {"type": "boolean"},
                "follow_up_focus": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


@dataclass
class ParsedCareQuery:
    """Structured representation of the user's care request."""

    detected_language: str
    response_language: str
    summary: str
    medical_need: bool
    location: Optional[str]
    specialties: List[str]
    insurance: List[str]
    preferred_languages: List[str]
    keywords: List[str]
    patient_context: Optional[str]
    care_setting: Optional[str] = None
    urgency: Optional[str] = None
    needs_clarification: bool = False
    follow_up_focus: List[str] = field(default_factory=list)


class IntentMixin:

    def _interpret_completion(self, client, messages):
        """Call chat_completion with structured-output request_format.
        If the provider rejects it (any exception), retry once without
        response_format so the rest of the parse/repair/rescue chain can
        still run.
        """
        try:
            return client.chat_completion(
                messages,
                max_tokens=INTERPRET_MAX_TOKENS,
                temperature=0.0,
                top_p=0.1,
                response_format=INTERPRET_RESPONSE_FORMAT,
            )
        except Exception:
            logger.warning(
                "Structured-output interpret call failed; retrying without response_format",
                exc_info=True,
            )
            return client.chat_completion(
                messages,
                max_tokens=INTERPRET_MAX_TOKENS,
                temperature=0.0,
                top_p=0.1,
            )

    def _interpret_user_need(
        self, client: InferenceClient, message: str, history: List[Dict[str, str]]
    ) -> ParsedCareQuery:
        guidance = self.prompts.get("interpret") or (
            "You are a healthcare triage analyst. Given a user request in any language, "
            "extract structured search criteria for finding care providers. Respond with "
            "strict JSON describing detected language, response language, summary, a boolean medical_need, location, "
            "specialties, insurance, preferred languages, keywords, patient context, care_setting, urgency, needs_clarification, and follow_up_focus."
        )

        logger.debug("Interpret prompt loaded. length=%s", len(guidance))

        messages = normalize_chat_messages(
            [{"role": "system", "content": guidance}]
            + history
            + [{"role": "user", "content": message}]
        )

        completion = self._interpret_completion(client, messages)

        first_choice = completion.choices[0] if completion.choices else None
        content = self._content_from_completion_choice(first_choice) or ""
        logger.debug("Interpret response received. length=%s", len(content))
        parsed_payload = self._safe_json_parse(content)

        if not parsed_payload:
            logger.warning("Interpret response did not yield valid JSON; retrying with clarification prompt")
            retry_messages = normalize_chat_messages(
                messages
                + [{"role": "assistant", "content": content}]
                + [
                    {
                        "role": "user",
                        "content": (
                            "Please repeat the previous JSON using valid strict JSON syntax only. "
                            "Do not include explanations—return just the JSON object."
                        ),
                    }
                ]
            )
            retry_completion = self._interpret_completion(client, retry_messages)
            retry_first_choice = (
                retry_completion.choices[0]
                if retry_completion.choices
                else None
            )
            retry_content = self._content_from_completion_choice(retry_first_choice) or ""
            logger.debug("Interpret retry response received. length=%s", len(retry_content))
            parsed_payload = self._safe_json_parse(retry_content)

        if not parsed_payload:
            parsed_payload = self._rescue_interpret_payload_from_message(message)

        parsed_payload = self._reconcile_interpret_payload_specialties(
            parsed_payload,
            message,
        )
        parsed_payload = self._reconcile_interpret_payload_location(
            parsed_payload,
            message,
        )
        parsed_payload = self._sanitize_interpret_payload_trust_boundary(
            parsed_payload,
            message,
        )

        detected_language = str(parsed_payload.get("detected_language", "unknown"))
        response_language = parsed_payload.get("response_language") or detected_language or "English"

        if (
            response_language
            and detected_language
            and response_language.lower().strip() == "english"
            and detected_language.lower().strip() != "english"
        ):
            response_language = detected_language

        # Fold native digit scripts in the LLM-returned location: this field
        # bypasses message-side ZIP extraction and flows to the English-only
        # provider API (CASE_STUDY §4 / FINDINGS F9).
        raw_location = parsed_payload.get("location")
        parsed_query = ParsedCareQuery(
            detected_language=detected_language,
            response_language=str(response_language or "English"),
            summary=str(parsed_payload.get("summary", "")),
            medical_need=bool(parsed_payload.get("medical_need", True)),
            location=fold_digits(raw_location) if isinstance(raw_location, str) else raw_location,
            specialties=self._ensure_list(parsed_payload.get("specialties")),
            insurance=self._ensure_list(parsed_payload.get("insurance")),
            preferred_languages=self._ensure_list(parsed_payload.get("preferred_languages")),
            keywords=self._ensure_list(parsed_payload.get("keywords")),
            patient_context=parsed_payload.get("patient_context"),
            care_setting=parsed_payload.get("care_setting"),
            urgency=parsed_payload.get("urgency"),
            needs_clarification=bool(parsed_payload.get("needs_clarification", False)),
            follow_up_focus=self._ensure_list(parsed_payload.get("follow_up_focus")),
        )
        self._log_local_debug_interpret(parsed_query)
        return parsed_query

    # ------------------------------------------------------------------
    def _rescue_interpret_payload_from_message(self, message: str) -> Dict[str, Any]:
        rescued_payload = self._default_interpret_payload(message)
        rescued_location = None
        if not self._has_explicit_procedure_code_intent(message):
            rescued_location = self._rescue_location_from_message(message)
            if rescued_location:
                rescued_payload["location"] = rescued_location

        logger.warning(
            "Interpret response remained invalid after retry; applied deterministic rescue. location_present=%s specialties=%s",
            bool(rescued_location),
            0,
        )
        return rescued_payload

    # ------------------------------------------------------------------
    def _reconcile_interpret_payload_location(
        self,
        parsed_payload: Dict[str, Any],
        message: str,
    ) -> Dict[str, Any]:
        if not isinstance(parsed_payload, dict):
            return parsed_payload

        parsed_location = self._clean_card_value(parsed_payload.get("location"))
        if self._has_explicit_procedure_code_intent(message):
            trusted_location = self._trusted_location_for_procedure_intent(message)
            if trusted_location == parsed_location:
                return parsed_payload

            reconciled_payload = dict(parsed_payload)
            reconciled_payload["location"] = trusted_location
            logger.info(
                "Interpret payload adjusted location trust boundary for procedure intent. location_present=%s trusted_location=%s",
                bool(parsed_location),
                trusted_location or "",
            )
            return reconciled_payload

        if parsed_location:
            return parsed_payload

        specialties = self._ensure_list(parsed_payload.get("specialties"))
        if not specialties:
            return parsed_payload

        rescued_location = self._rescue_location_from_message(message)
        if not rescued_location:
            return parsed_payload

        reconciled_payload = dict(parsed_payload)
        reconciled_payload["location"] = rescued_location
        logger.info(
            "Interpret payload accepted JSON but restored raw-message location evidence. location=%s specialties=%s",
            rescued_location,
            len(specialties),
        )
        return reconciled_payload

    def _reconcile_interpret_payload_specialties(
        self,
        parsed_payload: Dict[str, Any],
        message: str,
    ) -> Dict[str, Any]:
        if not isinstance(parsed_payload, dict):
            return parsed_payload

        if self._has_explicit_procedure_code_intent(message):
            return parsed_payload

        family_ids = self._specialty_family_ids_from_message(message)
        if len(family_ids) > 1 or self._has_child_allergy_ambiguity(message):
            reconciled_payload = dict(parsed_payload)
            reconciled_payload["specialties"] = []
            reconciled_payload["needs_clarification"] = True
            reconciled_payload["follow_up_focus"] = self._append_follow_up_focus(
                parsed_payload.get("follow_up_focus"),
                _SPECIALTY_CLARIFICATION_FOCUS,
            )
            logger.info(
                "Interpret payload abstained from specialty resolution due to competing specialty families. families=%s",
                family_ids,
            )
            return reconciled_payload

        if self._ensure_list(parsed_payload.get("specialties")):
            return parsed_payload

        rescued_specialties = self._specialties_from_family_ids(family_ids)
        if not rescued_specialties:
            return parsed_payload

        reconciled_payload = dict(parsed_payload)
        reconciled_payload["specialties"] = rescued_specialties
        logger.info(
            "Interpret payload accepted JSON but restored raw-message specialty evidence. specialties=%s",
            tuple(rescued_specialties),
        )
        return reconciled_payload

    def _has_child_allergy_ambiguity(self, message: str) -> bool:
        normalized_message = self._normalized_query_text(message)
        if not normalized_message:
            return False

        return (
            self._contains_any(normalized_message, _CHILD_CARE_AMBIGUITY_PATTERNS)
            and self._contains_any(
                normalized_message,
                _GENERIC_ALLERGY_AMBIGUITY_PATTERNS,
            )
            and not self._contains_any(
                normalized_message,
                _EXPLICIT_ALLERGY_SPECIALTY_PATTERNS,
            )
        )

    def _sanitize_interpret_payload_trust_boundary(
        self,
        parsed_payload: Dict[str, Any],
        message: str,
    ) -> Dict[str, Any]:
        if not isinstance(parsed_payload, dict):
            return parsed_payload

        sanitized_payload = dict(parsed_payload)
        payload_changed = False

        if not self._has_explicit_procedure_code_intent(message):
            sanitized_payload, payload_changed = self._remove_untrusted_procedure_gloss(
                sanitized_payload,
                message,
            )

        parsed_location = self._clean_card_value(sanitized_payload.get("location"))
        parsed_zip_code = self._extract_zip_code(parsed_location)
        if not parsed_zip_code or parsed_location != parsed_zip_code:
            return sanitized_payload if payload_changed else parsed_payload

        if self._message_contains_zip_code(message, parsed_zip_code):
            return sanitized_payload if payload_changed else parsed_payload

        rescued_location = self._rescue_location_from_message(message)
        sanitized_payload["location"] = rescued_location
        logger.info(
            "Interpret payload rejected untrusted bare zip location. parsed_zip=%s trusted_location=%s",
            parsed_zip_code,
            rescued_location or "",
        )
        return sanitized_payload

    def _remove_untrusted_procedure_gloss(
        self,
        parsed_payload: Dict[str, Any],
        message: str,
    ) -> Tuple[Dict[str, Any], bool]:
        summary = self._clean_card_value(parsed_payload.get("summary"))
        keywords = self._ensure_list(parsed_payload.get("keywords"))
        follow_up_focus = self._ensure_list(parsed_payload.get("follow_up_focus"))

        summary_has_procedure_gloss = self._contains_procedure_code_gloss(summary)
        filtered_keywords = [
            keyword
            for keyword in keywords
            if not self._contains_procedure_code_gloss(keyword)
        ]
        filtered_follow_up_focus = [
            focus
            for focus in follow_up_focus
            if not self._contains_procedure_code_gloss(focus)
        ]

        if (
            not summary_has_procedure_gloss
            and len(filtered_keywords) == len(keywords)
            and len(filtered_follow_up_focus) == len(follow_up_focus)
        ):
            return parsed_payload, False

        sanitized_payload = dict(parsed_payload)
        if summary_has_procedure_gloss:
            sanitized_payload["summary"] = message
        sanitized_payload["keywords"] = filtered_keywords
        sanitized_payload["follow_up_focus"] = filtered_follow_up_focus
        logger.info(
            "Interpret payload removed untrusted procedure-code gloss. summary_changed=%s keywords_removed=%s focus_removed=%s",
            summary_has_procedure_gloss,
            len(keywords) - len(filtered_keywords),
            len(follow_up_focus) - len(filtered_follow_up_focus),
        )
        return sanitized_payload, True

    def _contains_procedure_code_gloss(self, value: object) -> bool:
        normalized_value = self._normalized_query_text(value)
        if not normalized_value:
            return False
        if self._contains_any(
            normalized_value,
            ("cpt", "procedure code", "billing code"),
        ):
            return True
        return bool(
            re.search(r"\bprocedure\b", normalized_value)
            and re.search(r"\b\d{5}(?:\s+\d{4})?\b", normalized_value)
        )

    def _specialties_from_message(self, message: str) -> List[str]:
        return self._specialties_from_family_ids(
            self._specialty_family_ids_from_message(message)
        )

    def _specialties_from_family_ids(
        self,
        family_ids: tuple[str, ...],
    ) -> List[str]:
        if len(family_ids) != 1:
            return []

        family = SPECIALTY_FAMILY_BY_ID.get(family_ids[0])
        if family is None:
            return []

        display_label = _INTERPRET_SPECIALTY_LABEL_OVERRIDES.get(
            family.family_id,
            family.label,
        )
        return [display_label]

    def _specialty_family_ids_from_message(self, message: str) -> tuple[str, ...]:
        tokens = self._specialty_message_tokens(message)
        if not tokens:
            return ()

        max_alias_tokens = max(
            (
                len(self._specialty_message_tokens(candidate))
                for aliases in QUERY_SPECIALTY_FAMILY_ALIASES_BY_ID.values()
                for candidate in aliases
                if self._specialty_message_tokens(candidate)
            ),
            default=0,
        )
        if max_alias_tokens <= 0:
            return ()

        family_ids: List[str] = []
        seen_family_ids: set[str] = set()
        for start_index in range(len(tokens)):
            max_length = min(max_alias_tokens, len(tokens) - start_index)
            for length in range(max_length, 0, -1):
                phrase = " ".join(tokens[start_index : start_index + length])
                family_id = normalize_query_specialty_family_id(phrase)
                if family_id is None:
                    continue
                if family_id in seen_family_ids:
                    break
                seen_family_ids.add(family_id)
                family_ids.append(family_id)
                break

        return tuple(family_ids)

    @staticmethod
    def _specialty_message_tokens(value: object) -> List[str]:
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        normalized = normalized.replace("&", " and ")
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", normalized)
            if token and not token.isdigit()
        ]
        return tokens

    def _trusted_location_for_procedure_intent(self, message: str) -> Optional[str]:
        best_candidate: Optional[Tuple[int, str]] = None
        for pattern in (self._city_state_code_regex, self._city_state_name_regex):
            for match in pattern.finditer(message):
                city = match.group("city").strip().strip(",")
                state_fragment = match.group("state").strip()

                state_code = None
                if pattern is self._city_state_code_regex:
                    state_code = state_fragment.upper()
                else:
                    state_code = _US_STATE_NAMES.get(state_fragment.lower())

                if not state_code or not self._rescue_city_token_is_valid(city):
                    continue

                location = f"{city} {state_code}"
                if not self._location_has_city(location):
                    continue

                trailing_zip_code = self._trusted_zip_after_city_state(message, match.end())
                trusted_location = (
                    f"{city}, {state_code} {trailing_zip_code}"
                    if trailing_zip_code
                    else f"{city}, {state_code}"
                )
                candidate_score = len(city.replace(" ", "")) * 10 + int(
                    bool(trailing_zip_code)
                )
                if best_candidate is None or candidate_score >= best_candidate[0]:
                    best_candidate = (candidate_score, trusted_location)

        if best_candidate is None:
            return None
        return best_candidate[1]

    def _trusted_zip_after_city_state(
        self,
        message: str,
        start_index: int,
    ) -> Optional[str]:
        trailing_text = message[start_index:]
        zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", trailing_text)
        if not zip_match:
            return None

        leading_fragment = trailing_text[: zip_match.start()]
        if re.search(r"[A-Za-z]", leading_fragment):
            return None

        return zip_match.group(1)

    @staticmethod
    def _message_contains_zip_code(message: str, zip_code: str) -> bool:
        return re.search(
            rf"\b{re.escape(zip_code)}(?:-\d{{4}})?\b",
            message,
        ) is not None

    def _log_local_debug_interpret(self, parsed_query: ParsedCareQuery) -> None:
        if not self._local_debug_enabled():
            return

        logger.info(
            "care_local_debug_interpret detected_language=%s response_language=%s medical_need=%s location_present=%s location_shape=%s specialties=%s insurance_count=%s preferred_language_count=%s keyword_count=%s needs_clarification=%s care_setting=%s urgency=%s summary_length=%s",
            parsed_query.detected_language,
            parsed_query.response_language,
            parsed_query.medical_need,
            bool(parsed_query.location),
            self._debug_location_shape(parsed_query.location),
            tuple(parsed_query.specialties),
            len(parsed_query.insurance),
            len(parsed_query.preferred_languages),
            len(parsed_query.keywords),
            parsed_query.needs_clarification,
            parsed_query.care_setting or "",
            parsed_query.urgency or "",
            len(parsed_query.summary or ""),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _default_interpret_payload(message: str) -> Dict[str, Any]:
        return {
            "detected_language": "unknown",
            "response_language": "English",
            "summary": message,
            "medical_need": True,
            "location": None,
            "specialties": [],
            "insurance": [],
            "preferred_languages": [],
            "keywords": [],
            "patient_context": None,
            "care_setting": None,
            "urgency": None,
            "needs_clarification": False,
            "follow_up_focus": [],
        }

    # ------------------------------------------------------------------
    def _rescue_location_from_message(self, message: str) -> Optional[str]:
        zip_code = self._extract_zip_code(message)
        city_state = self._rescue_city_state_from_message(message)

        if city_state and zip_code:
            city, state_code = city_state
            return f"{city}, {state_code} {zip_code}"
        if city_state:
            city, state_code = city_state
            return f"{city}, {state_code}"
        return zip_code

    # ------------------------------------------------------------------
    def _rescue_specialties_from_message(self, message: str) -> List[str]:
        return self._specialties_from_message(message)

    # ------------------------------------------------------------------
    def _has_explicit_procedure_code_intent(self, message: str) -> bool:
        normalized_message = str(message or "").lower()
        return self._contains_any(
            normalized_message,
            _PROCEDURE_CODE_INTENT_PATTERNS,
        )

    # ------------------------------------------------------------------
    def _rescue_city_state_from_message(self, message: str) -> Optional[Tuple[str, str]]:
        best_candidate: Optional[Tuple[int, Tuple[str, str]]] = None
        for pattern in (self._city_state_code_regex, self._city_state_name_regex):
            for match in pattern.finditer(message):
                city = match.group("city").strip().strip(",")
                state_fragment = match.group("state").strip()

                state_code = None
                if pattern is self._city_state_code_regex:
                    state_code = state_fragment.upper()
                else:
                    state_code = _US_STATE_NAMES.get(state_fragment.lower())

                if not state_code:
                    continue
                if not self._rescue_city_token_is_valid(city):
                    continue

                location = f"{city} {state_code}"
                if not self._location_has_city(location):
                    continue

                candidate_score = len(city.replace(" ", ""))
                if best_candidate is None or candidate_score >= best_candidate[0]:
                    best_candidate = (candidate_score, (city.strip(), state_code))

        if best_candidate is None:
            return None
        return best_candidate[1]

    # ------------------------------------------------------------------
    @staticmethod
    def _rescue_city_token_is_valid(city: str) -> bool:
        normalized_city = city.strip().lower()
        if not normalized_city:
            return False

        if normalized_city in _LOCATION_NOISE_TOKENS:
            return False

        specialty_like_terms = {
            normalized_candidate
            for family in SPECIALTY_FAMILY_BY_ID.values()
            for candidate in (family.label, *family.aliases)
            for normalized_candidate in [
                unicodedata.normalize("NFKC", candidate).casefold().strip()
            ]
            if normalized_candidate
            and " " not in normalized_candidate
            and "/" not in normalized_candidate
        }
        if normalized_city in specialty_like_terms:
            return False

        return True

    # ------------------------------------------------------------------
    def _apply_conversation_language_backstop(
        self,
        query: "ParsedCareQuery",
        message: str,
        history: List[Dict[str, str]],
    ) -> "ParsedCareQuery":
        """Deterministic conversation-language override for signal-less turns.

        The full-history parse can misread a conversation whose latest turn is
        a bare ZIP (F11). When (a) the conversation is multi-turn, (b) the
        latest message carries no language signal, and (c) the merged language
        resolves to English/unknown, character-script evidence from the user's
        own messages decides instead. Latin-script languages are left to the
        parse (scripts cannot distinguish them from English).
        """
        if not history:
            return query
        if _message_has_language_signal(message):
            return query
        merged_language = query.response_language or query.detected_language
        if not _is_unknown_response_language(merged_language):
            if _lookup_language_alias(_normalize_response_language(merged_language)) != "english":
                return query
        user_texts = [
            str(turn.get("content") or "")
            for turn in history
            if isinstance(turn, dict) and turn.get("role") == "user"
        ] + [message]
        dominant = _dominant_user_script_language(user_texts)
        if dominant is None:
            return query
        return replace(query, detected_language=dominant, response_language=dominant)

    # ------------------------------------------------------------------
    def _merge_parsed_queries(
        self, full: ParsedCareQuery, latest: ParsedCareQuery
    ) -> ParsedCareQuery:
        def _prefer(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
            return primary if primary not in (None, "", "unknown") else fallback

        def _prefer_language(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
            # Language fields judge absence via _is_unknown_response_language
            # (case/variant tolerant: "Unknown", "N/A", "undetected", ...) —
            # the model does not spell its sentinel consistently.
            return fallback if _is_unknown_response_language(primary) else primary

        def _merge_lists(primary: List[str], fallback: List[str]) -> List[str]:
            merged: List[str] = []
            for item in primary + fallback:
                if item not in merged:
                    merged.append(item)
            return merged

        # Language is a conversation-level attribute: the full-history parse
        # sees every turn (including a genuine language switch), while the
        # latest-only parse of a low-information turn ("98101", "ok") reads as
        # English and used to clobber the conversation language (the s15
        # multi-turn English-rendering failure).
        detected_language = _prefer_language(full.detected_language, latest.detected_language)
        response_language = _prefer_language(full.response_language, latest.response_language)

        latest_needs_specialty_clarification = self._requires_specialty_clarification(
            latest.follow_up_focus
        )
        full_needs_specialty_clarification = self._requires_specialty_clarification(
            full.follow_up_focus
        )
        latest_is_location_only_follow_up = (
            full_needs_specialty_clarification
            and self._is_location_only_follow_up_turn(latest)
        )

        if latest_needs_specialty_clarification:
            specialties = []
        elif latest.specialties:
            specialties = latest.specialties
        elif full_needs_specialty_clarification:
            specialties = []
        else:
            specialties = full.specialties
        location = latest.location or full.location
        insurance = latest.insurance or full.insurance
        preferred_languages = _merge_lists(latest.preferred_languages, full.preferred_languages)
        keywords = _merge_lists(latest.keywords, full.keywords)
        patient_context = latest.patient_context or full.patient_context
        summary = (
            full.summary
            if full_needs_specialty_clarification and latest_is_location_only_follow_up
            else (latest.summary or full.summary)
        )
        care_setting = latest.care_setting or full.care_setting
        urgency = latest.urgency or full.urgency
        follow_up_focus = _merge_lists(latest.follow_up_focus, full.follow_up_focus)
        specialty_clarification_needed = latest_needs_specialty_clarification or (
            full_needs_specialty_clarification and not specialties
        )
        if specialty_clarification_needed:
            follow_up_focus = self._append_follow_up_focus(
                follow_up_focus,
                _SPECIALTY_CLARIFICATION_FOCUS,
            )
        else:
            follow_up_focus = self._remove_follow_up_focus(
                follow_up_focus,
                _SPECIALTY_CLARIFICATION_FOCUS,
            )

        latest_need = latest.medical_need
        full_need = full.medical_need
        if latest_need is None and full_need is None:
            medical_need = False
        elif latest_need is None:
            medical_need = bool(full_need)
        elif full_need is None:
            medical_need = bool(latest_need)
        else:
            medical_need = bool(latest_need or full_need)

        if not medical_need and specialties:
            medical_need = True

        ambiguity_resolved_in_latest_turn = bool(
            specialties
            and not specialty_clarification_needed
            and full_needs_specialty_clarification
            and not latest_needs_specialty_clarification
        )
        if ambiguity_resolved_in_latest_turn:
            needs_clarification = bool(latest.needs_clarification or follow_up_focus)
        else:
            needs_clarification = bool(
                latest.needs_clarification
                or full.needs_clarification
                or specialty_clarification_needed
            )

        return ParsedCareQuery(
            detected_language=detected_language or full.detected_language,
            response_language=response_language or full.response_language,
            summary=summary,
            medical_need=medical_need,
            location=location,
            specialties=specialties,
            insurance=insurance,
            preferred_languages=preferred_languages,
            keywords=keywords,
            patient_context=patient_context,
            care_setting=care_setting,
            urgency=urgency,
            needs_clarification=needs_clarification,
            follow_up_focus=follow_up_focus,
        )

    def _is_location_only_follow_up_turn(self, query: ParsedCareQuery) -> bool:
        if not query.location or query.specialties:
            return False
        if (
            query.insurance
            or query.preferred_languages
            or query.keywords
            or query.patient_context
            or query.urgency
        ):
            return False
        if query.care_setting and str(query.care_setting).strip().casefold() != "specialist":
            return False

        normalized_summary = self._normalized_query_text(query.summary)
        if not normalized_summary:
            return True
        return normalized_summary == self._normalized_query_text(query.location)

    @staticmethod
    def _normalized_query_text(value: object) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _append_follow_up_focus(self, values: Any, focus: str) -> List[str]:
        return self._dedupe_preserve_order(
            self._ensure_list(values) + [focus]
        )

    def _remove_follow_up_focus(self, values: Any, focus: str) -> List[str]:
        normalized_focus = focus.strip().casefold()
        return [
            value
            for value in self._ensure_list(values)
            if value.strip().casefold() != normalized_focus
        ]

    def _requires_specialty_clarification(self, values: Any) -> bool:
        normalized_focus = _SPECIALTY_CLARIFICATION_FOCUS.casefold()
        return any(
            value.strip().casefold() == normalized_focus
            for value in self._ensure_list(values)
        )

    # ------------------------------------------------------------------
    def _location_has_city(self, value: str) -> bool:
        normalized = value.strip().lower()
        if not normalized:
            return False

        if normalized in _US_STATE_NAMES:
            return False

        if normalized.upper() in _US_STATE_CODES:
            return False

        stripped_state_candidate = re.sub(
            r"\b(state|province|region|area|of|the)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped_state_candidate = re.sub(r"\s+", " ", stripped_state_candidate).strip()
        if stripped_state_candidate in _US_STATE_NAMES:
            return False

        tokens = [token.strip() for token in re.split(r"[,\s]+", normalized) if token.strip()]
        for token in tokens:
            upper = token.upper()
            if upper in _US_STATE_CODES or token in _US_STATE_NAMES:
                continue
            if token in _LOCATION_NOISE_TOKENS:
                continue
            if token.isdigit():
                continue
            if len(token) < 3:
                continue
            if not re.search(r"[A-Za-z]", token):
                continue
            return True
        return False

    # ------------------------------------------------------------------
    def _extract_zip_code(self, value: str) -> Optional[str]:
        # Use digit lookarounds rather than \b word boundaries: Python's Unicode
        # regex treats CJK characters as word characters, so "\b\d{5}\b" fails to
        # match a ZIP glued to CJK text (e.g. "儿科10013"). Lookarounds still
        # reject digits that are part of a longer number (e.g. "100135").
        # Fold non-ASCII digit scripts first (Arabic-Indic ٩٤١١٠, fullwidth
        # １００１３): Unicode \d already MATCHED them, but the extracted value
        # must be ASCII for the English-only provider API (FINDINGS F9).
        match = re.search(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)", fold_digits(value))
        if match:
            return match.group(1)
        return None

    # ------------------------------------------------------------------
    def _extract_state_code(self, value: str) -> Optional[str]:
        tokens = re.split(r"[,\s]+", value)
        for token in tokens:
            upper = token.strip().upper()
            if upper in _US_STATE_CODES:
                return upper
        normalized = value.strip().lower()
        if normalized in _US_STATE_NAMES:
            return _US_STATE_NAMES[normalized]
        stripped = re.sub(
            r"\b(state|province|region|area|of|the)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped in _US_STATE_NAMES:
            return _US_STATE_NAMES[stripped]
        return None

    # ------------------------------------------------------------------
    def _match_city_state(self, text: str) -> Optional[Tuple[str, str]]:
        for pattern in (self._city_state_code_regex, self._city_state_name_regex):
            match = pattern.search(text)
            if not match:
                continue

            city = match.group("city").strip().strip(",")
            state_fragment = match.group("state").strip()

            state_code = None
            if pattern is self._city_state_code_regex:
                state_code = state_fragment.upper()
            else:
                state_code = _US_STATE_NAMES.get(state_fragment.lower())

            if not state_code:
                continue

            location = f"{city} {state_code}"
            if self._location_has_city(location):
                return city.strip(), state_code

        return None

    # ------------------------------------------------------------------
    def _safe_json_parse(self, content: str) -> Optional[dict]:
        if not content:
            return None

        trimmed = content.strip()
        if trimmed.startswith('```'):
            trimmed = trimmed[3:]
            if trimmed.lower().startswith('json'):
                trimmed = trimmed[4:]
            trimmed = trimmed.lstrip("\n").rstrip("`").strip()

        try:
            return json.loads(trimmed)
        except json.JSONDecodeError as exc:
            repaired = self._repair_json(trimmed)
            if repaired and repaired != trimmed:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.warning("Interpret payload repair failed", exc_info=True)
            logger.warning("Interpret payload JSON parse failed: %s", exc)
            logger.debug("Interpret payload raw text omitted. length=%s", len(trimmed))
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _repair_json(payload: str) -> str:
        if not payload:
            return payload

        repaired = re.sub(r",(\s*[}\]])", r"\1", payload)

        stack: List[str] = []
        for char in repaired:
            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in ("}", "]") and stack:
                if stack[-1] == char:
                    stack.pop()

        while stack:
            repaired += stack.pop()

        return repaired

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            normalized: List[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    if item.strip():
                        normalized.append(item.strip())
                    continue
                if isinstance(item, dict):
                    for key in (
                        "language",
                        "language_desc",
                        "desc",
                        "description",
                        "name",
                    ):
                        candidate = item.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            normalized.append(candidate.strip())
                            break
                else:
                    normalized.append(str(item))
            return normalized
        if isinstance(value, str):
            if not value.strip():
                return []
            return [value]
        return []

    # ------------------------------------------------------------------
    @staticmethod
    def _first_match(data: Dict[str, Any], keys: List[str], default: Optional[str] = None) -> Optional[str]:
        for key in keys:
            if key in data and data[key]:
                return str(data[key])
        return default
