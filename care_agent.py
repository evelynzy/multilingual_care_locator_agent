from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import logging

from huggingface_hub import InferenceClient

from config_loader import get_prompt, get_search_settings
from provider_search import (
    ProviderSearchRequest,
    ProviderSearchService,
    SQLiteProviderSearchCache,
)
from provider_search.sources import (
    ClinicalTablesDatasetConfig,
    ClinicalTablesSource,
    NPPESSource,
)
from provider_search.sources.clinicaltables import DEFAULT_DATASET_CONFIGS
from care.language import (
    LanguageMixin,
    _NORMALIZED_TRUST_GUIDANCE_ALIASES,
    _REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES,
    _UNKNOWN_LANGUAGE_MARKERS,
    _normalize_response_language,
    normalize_chat_messages,
)
from care.safety import (
    SafetyMixin,
    _EMERGENCY_PATTERNS,
    _EMERGENCY_URGENCY_VALUES,
    _REQUIRED_TRUST_GUIDANCE,
    _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE,
    _get_prewritten_required_trust_guidance,
)
from care.intent import (
    INTERPRET_MAX_TOKENS,
    INTERPRET_RESPONSE_FORMAT,
    IntentMixin,
    ParsedCareQuery,
    _CHILD_CARE_AMBIGUITY_PATTERNS,
    _EXPLICIT_ALLERGY_SPECIALTY_PATTERNS,
    _GENERIC_ALLERGY_AMBIGUITY_PATTERNS,
    _INTERPRET_SPECIALTY_LABEL_OVERRIDES,
    _LOCATION_NOISE_TOKENS,
    _PROCEDURE_CODE_INTENT_PATTERNS,
    _SPECIALTY_CLARIFICATION_FOCUS,
    _US_STATE_CODES,
    _US_STATE_NAMES,
)
from care.guidance import (
    GuidanceMixin,
    _GENERIC_SPECIALIST_PATTERNS,
    _INSURANCE_PATTERNS,
    _LANGUAGE_PATTERNS,
    _PLAN_TYPE_PATTERNS,
    _REFERRAL_PATTERNS,
    _ROUTINE_PATTERNS,
    _SPECIALIST_PATTERNS,
    _URGENT_PATTERNS,
)
from care.rendering import (
    RenderingMixin,
    _DETERMINISTIC_RENDER_COPY,
    _DETERMINISTIC_RENDER_TRANSLATIONS,
    _reply_localization_target,
    _resolved_supported_language_key,
)

logger = logging.getLogger(__name__)


class CareLocatorAgent(LanguageMixin, SafetyMixin, IntentMixin, GuidanceMixin, RenderingMixin):
    """Coordinates LLM reasoning, dataset search, and fallback lookups."""

    def __init__(
        self,
        provider_search_service: Optional[ProviderSearchService] = None,
    ) -> None:
        self.prompts = {
            "interpret": get_prompt("interpret_user_need"),
            "response_system": get_prompt("response_system"),
            "response_template": get_prompt("response_user_template"),
            "response_template_clarification_needed": get_prompt(
                "response_user_template_clarification_needed"
            ),
            "response_template_emergency": get_prompt(
                "response_user_template_emergency"
            ),
            "response_template_fallback_only": get_prompt(
                "response_user_template_fallback_only"
            ),
            "response_template_location_needed": get_prompt(
                "response_user_template_location_needed"
            ),
        }
        self.search_settings = get_search_settings()
        clinical = self.search_settings.get("clinicaltables", {})
        field_probe_terms = clinical.get("field_probe_terms", {}) or {}
        self.ctss_timeout = clinical.get("timeout", 6)
        self.ctss_max_results = clinical.get("max_results", 3)

        self.npi_registry_config = self.search_settings.get("npi_registry", {}) or {}
        self._npi_registry_enabled = bool(
            self.npi_registry_config.get("enabled", True)
            and self.npi_registry_config.get("lookup_url")
        )
        self.npi_registry_url = self.npi_registry_config.get("lookup_url")
        self.npi_registry_timeout = self.npi_registry_config.get(
            "timeout", self.ctss_timeout
        )
        self.npi_registry_version = str(
            self.npi_registry_config.get("version", "2.1")
        )

        # Keep only runtime overrides here; shared ClinicalTables defaults live in provider_search.
        self.ctss_dataset_overrides: Dict[str, Dict[str, Any]] = {
            "npi_idv": {
                "search_url": clinical.get("individual_search_url"),
                # The v3 ClinicalTables API no longer exposes dedicated values/fields endpoints
                # for NPI datasets. Keep the override keys optional and default them to None
                # so the agent skips those HTTP calls unless configured explicitly.
                "values_url": clinical.get("individual_values_url") or None,
                "fields_url": clinical.get("individual_fields_url") or None,
                "probe_term": field_probe_terms.get("npi_idv"),
            },
            "npi_org": {
                "search_url": clinical.get("organization_search_url"),
                "values_url": clinical.get("organization_values_url") or None,
                "fields_url": clinical.get("organization_fields_url") or None,
                "probe_term": field_probe_terms.get("npi_org"),
            },
        }
        merged_ctss_configs = self._provider_search_dataset_configs()
        self._ctss_dataset_priority: List[str] = [
            dataset
            for dataset, cfg in merged_ctss_configs.items()
            if cfg.search_url
        ]

        self.fallback_resources = self.search_settings.get("fallback_resources", [])

        state_code_pattern = "|".join(sorted(_US_STATE_CODES))
        state_name_pattern = "|".join(
            re.escape(name)
            for name in sorted(_US_STATE_NAMES.keys(), key=len, reverse=True)
        )
        city_pattern = r"(?P<city>(?-i:[A-Z])[A-Za-z'.-]{2,}(?:\s+(?-i:[A-Z])[A-Za-z'.-]{2,})*)"
        separator_pattern = r"\s*,?\s+"
        self._city_state_code_regex = re.compile(
            rf"{city_pattern}{separator_pattern}(?P<state>{state_code_pattern})\b",
            re.IGNORECASE,
        )
        self._city_state_name_regex = re.compile(
            rf"{city_pattern}{separator_pattern}(?P<state>{state_name_pattern})\b",
            re.IGNORECASE,
        )
        self.provider_search_service = (
            provider_search_service or self._build_provider_search_service()
        )

    # ------------------------------------------------------------------
    def handle_request(
        self,
        client: InferenceClient,
        message: str,
        history: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        logger.info("Handling request. message_length=%s history_turns=%s", len(message), len(history))
        parsed_query = self._interpret_user_need(client, message, history)

        if history:
            latest_only_query = self._interpret_user_need(client, message, [])
            parsed_query = self._merge_parsed_queries(parsed_query, latest_only_query)

        has_emergency_signal = self._contains_emergency_signal(message.lower())
        navigation_guidance = (
            self._build_navigation_guidance(parsed_query, message)
            if parsed_query.medical_need or has_emergency_signal
            else {
                "mode": "search",
                "care_setting_guidance": None,
                "follow_up_questions": [],
                "specialist_plan_guidance": None,
                "location_only": False,
            }
        )

        response_payload = {
            "query": {
                "detected_language": parsed_query.detected_language,
                "response_language": parsed_query.response_language,
                "summary": parsed_query.summary,
                "medical_need": parsed_query.medical_need or has_emergency_signal,
                "location": parsed_query.location,
                "specialties": parsed_query.specialties,
                "insurance": parsed_query.insurance,
                "preferred_languages": parsed_query.preferred_languages,
                "keywords": parsed_query.keywords,
                "patient_context": parsed_query.patient_context,
                "care_setting": parsed_query.care_setting,
                "urgency": parsed_query.urgency,
                "needs_clarification": parsed_query.needs_clarification,
                "follow_up_focus": parsed_query.follow_up_focus,
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": self._verification_guidance(),
        }

        if navigation_guidance.get("care_setting_guidance"):
            response_payload["care_setting_guidance"] = navigation_guidance["care_setting_guidance"]
        if navigation_guidance.get("follow_up_questions"):
            response_payload["follow_up_questions"] = navigation_guidance["follow_up_questions"]
        if navigation_guidance.get("specialist_plan_guidance"):
            response_payload["specialist_plan_guidance"] = navigation_guidance["specialist_plan_guidance"]

        if navigation_guidance.get("mode") == "emergency":
            response_payload["emergency_guidance"] = navigation_guidance.get("care_setting_guidance")
            return self._compose_response(
                client,
                response_payload,
                max_tokens,
                temperature,
                top_p,
                template_key="response_template_emergency",
            )

        if navigation_guidance.get("follow_up_questions"):
            template_key = (
                "response_template_location_needed"
                if navigation_guidance.get("location_only")
                else "response_template_clarification_needed"
            )
            return self._compose_response(
                client,
                response_payload,
                max_tokens,
                temperature,
                top_p,
                template_key=template_key,
            )

        local_results: List[dict] = []
        fallback_results: List[dict] = []

        missing_location_hint = None
        had_source_failures = False

        if parsed_query.medical_need:
            provider_request = ProviderSearchRequest(
                specialties=tuple(parsed_query.specialties),
                location=parsed_query.location,
                insurance=tuple(parsed_query.insurance),
                preferred_languages=tuple(parsed_query.preferred_languages),
                keywords=tuple(parsed_query.keywords),
            )
            self._log_local_debug_provider_search_handoff(
                parsed_query=parsed_query,
                provider_request=provider_request,
            )
            search_response = self.provider_search_service.search(
                provider_request,
                limit=self.ctss_max_results,
            )
            missing_location_hint = search_response.missing_location_hint
            had_source_failures = any(
                bool(trace.error)
                for trace in search_response.search_trace.source_traces
            )

            local_results = [
                self._normalize_result_trust_metadata(
                    self._provider_search_result_to_payload(result)
                )
                for result in search_response.provider_results
            ]
            logger.info(
                "ProviderSearchService results=%s cache_hit=%s candidates=%s",
                len(local_results),
                search_response.search_trace.cache_hit,
                search_response.search_trace.total_candidates,
            )

            unverified_languages = self._unverified_preferred_languages(
                parsed_query.preferred_languages, search_response.provider_results
            )
            if unverified_languages and local_results:
                response_payload["language_unverified"] = unverified_languages

            if search_response.fallback_resources:
                fallback_results = [
                    self._normalize_result_trust_metadata(
                        self._fallback_resource_to_payload(resource, parsed_query)
                    )
                    for resource in search_response.fallback_resources
                ]

            if not local_results and not fallback_results and not missing_location_hint:
                fallback_results = [
                    self._normalize_result_trust_metadata(result)
                    for result in self._trusted_resource_fallback(parsed_query)
                ]
                logger.info(
                    "Trusted resource fallback results=%s", len(fallback_results)
                )
            if self._local_debug_enabled():
                logger.info(
                    "care_agent_result_debug request_fingerprint=%s local_results=%s fallback_results=%s final_visible=%s had_source_failures=%s missing_location_hint=%s",
                    search_response.search_trace.request_fingerprint,
                    len(local_results),
                    len(fallback_results),
                    len(local_results) + len(fallback_results),
                    had_source_failures,
                    bool(missing_location_hint),
                )
        else:
            logger.info("Parsed request marked as non-medical; skipping provider search")

        if navigation_guidance.get("care_setting_guidance") and "care_setting_guidance" not in response_payload:
            response_payload["care_setting_guidance"] = navigation_guidance["care_setting_guidance"]
        if navigation_guidance.get("specialist_plan_guidance") and "specialist_plan_guidance" not in response_payload:
            response_payload["specialist_plan_guidance"] = navigation_guidance["specialist_plan_guidance"]

        no_results_found = (
            parsed_query.medical_need
            and not local_results
            and not fallback_results
            and not missing_location_hint
        )
        response_payload["local_results"] = local_results
        response_payload["fallback_results"] = fallback_results

        if missing_location_hint:
            response_payload.setdefault("notes", missing_location_hint)
        elif had_source_failures and not local_results:
            response_payload.setdefault(
                "notes",
                "Provider search sources were temporarily unavailable. Showing trusted fallback resources when available.",
            )
        elif no_results_found:
            response_payload.setdefault(
                "notes",
                "No providers were found via the configured provider search sources."
            )

        if parsed_query.medical_need and (local_results or fallback_results) and not missing_location_hint:
            card_reply = self._compose_result_card_response(response_payload)
            target_language = _reply_localization_target(
                parsed_query.response_language or parsed_query.detected_language
            )
            if target_language:
                card_reply = self._localize_reply_via_llm(
                    client, card_reply, target_language, max_tokens, temperature, top_p
                )
            return card_reply

        if missing_location_hint:
            template_key = "response_template_location_needed"
        else:
            template_key = "response_template_fallback_only"

        return self._compose_response(
            client,
            response_payload,
            max_tokens,
            temperature,
            top_p,
            template_key=template_key,
        )

    def _log_local_debug_provider_search_handoff(
        self,
        *,
        parsed_query: ParsedCareQuery,
        provider_request: ProviderSearchRequest,
    ) -> None:
        if not self._local_debug_enabled():
            return

        logger.info(
            "care_agent_local_debug_handoff medical_need=%s location_present=%s location_shape=%s specialties=%s insurance_count=%s preferred_language_count=%s keyword_count=%s",
            parsed_query.medical_need,
            bool(provider_request.location),
            self._debug_location_shape(provider_request.location),
            tuple(provider_request.specialties),
            len(provider_request.insurance),
            len(provider_request.preferred_languages),
            len(provider_request.keywords),
        )

    @staticmethod
    def _local_debug_enabled() -> bool:
        return (
            os.getenv("PROVIDER_SEARCH_DEBUG", "").strip() == "1"
            and os.getenv("CARE_LOCATOR_LOCAL_DEBUG", "").strip() == "1"
        )

    @staticmethod
    def _debug_location_shape(location: Optional[str]) -> str:
        if not location:
            return "missing"
        parts = []
        if re.search(r"\b\d{5}(?:-\d{4})?\b", location):
            parts.append("zip")
        if "," in location:
            parts.append("comma")
        if re.search(r"\b[A-Z]{2}\b", location):
            parts.append("state")
        if not parts:
            parts.append("freeform")
        return "+".join(parts)


    # ------------------------------------------------------------------
    def _build_provider_search_service(self) -> ProviderSearchService:
        nppes_source = NPPESSource(
            lookup_url=self.npi_registry_url or "https://npiregistry.cms.hhs.gov/api/",
            version=self.npi_registry_version,
            timeout=self.npi_registry_timeout,
        )
        clinicaltables_source = ClinicalTablesSource(
            timeout=self.ctss_timeout,
            dataset_configs=self._provider_search_dataset_configs(),
            nppes_source=nppes_source if self._npi_registry_enabled else None,
        )
        cache = SQLiteProviderSearchCache()
        return ProviderSearchService(
            clinicaltables_source=clinicaltables_source,
            cache=cache,
            datasets=tuple(self._ctss_dataset_priority),
            per_dataset_limit=self.ctss_max_results,
        )

    # ------------------------------------------------------------------
    def _provider_search_dataset_configs(self) -> dict[str, ClinicalTablesDatasetConfig]:
        configs: dict[str, ClinicalTablesDatasetConfig] = {}
        for dataset, default_config in DEFAULT_DATASET_CONFIGS.items():
            overrides = self.ctss_dataset_overrides.get(dataset, {})
            search_url = (
                overrides["search_url"]
                if "search_url" in overrides
                else default_config.search_url
            )
            if not search_url:
                continue
            configs[dataset] = ClinicalTablesDatasetConfig(
                search_url=search_url,
                values_url=(
                    overrides["values_url"]
                    if "values_url" in overrides
                    else default_config.values_url
                ),
                fields_url=(
                    overrides["fields_url"]
                    if "fields_url" in overrides
                    else default_config.fields_url
                ),
                probe_term=(
                    overrides["probe_term"]
                    if "probe_term" in overrides
                    else default_config.probe_term
                ),
                source_label=default_config.source_label,
                result_fields=list(default_config.result_fields),
            )
        return configs

    # ------------------------------------------------------------------
    def _compose_response(
        self,
        client: InferenceClient,
        payload: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        template_key: str = "response_template",
    ) -> str:
        query = payload.get("query", {})
        response_language = query.get("response_language") or query.get("detected_language") or "English"

        if self._should_use_deterministic_numeric_clarification(payload, template_key):
            logger.info(
                "Using deterministic clarification response for numeric-token trust boundary. template_key=%s",
                template_key,
            )
            return self._compose_safe_fallback_response(
                payload,
                response_language,
                template_key,
                finish_reason="numeric_trust_boundary",
            )

        summary_prompt = self.prompts.get("response_system") or (
            "You are a care navigation assistant responding to users in their preferred language."
        )

        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        template = self.prompts.get(template_key) or (
            "Here is the structured data you must convert into a helpful message:\n"
            "```json\n{payload_json}\n```\n"
            "Requirements:\n"
            "- Write in the language specified by query.response_language.\n"
            "- Briefly confirm the understood need.\n"
            "- Present providers as a numbered list with key details.\n"
            "- Mention if the results came from a public API or curated resource list.\n"
            "- Offer next steps or safety guidance when appropriate."
        )
        user_instructions = template.format(payload_json=payload_json)

        messages = normalize_chat_messages(
            [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": user_instructions},
            ]
        )

        logger.debug("Compose prompt prepared. system_length=%s", len(summary_prompt))
        logger.debug("Compose payload prepared. json_length=%s", len(payload_json))

        completion = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        choices = getattr(completion, "choices", None) or []
        first_choice = choices[0] if choices else None
        finish_reason = (
            getattr(first_choice, "finish_reason", None)
            if first_choice is not None
            else "unknown"
        )
        logger.info("Response generation finish_reason=%s", finish_reason)

        if first_choice is None:
            logger.warning(
                "Response generation returned no choices; using deterministic fallback. template_key=%s",
                template_key,
            )
            return self._compose_safe_fallback_response(
                payload,
                response_language,
                template_key,
                finish_reason=finish_reason,
            )

        content = self._content_from_completion_choice(first_choice)
        normalized_content = content.strip() if isinstance(content, str) else ""

        if finish_reason == "length" or not normalized_content:
            logger.warning(
                "Response generation returned incomplete or empty content; using deterministic fallback. finish_reason=%s template_key=%s",
                finish_reason,
                template_key,
            )
            return self._compose_safe_fallback_response(
                payload,
                response_language,
                template_key,
                finish_reason=finish_reason,
            )

        return self._append_required_trust_guidance(
            normalized_content,
            response_language,
        )

    def _should_use_deterministic_numeric_clarification(
        self,
        payload: Dict[str, Any],
        template_key: str,
    ) -> bool:
        if template_key != "response_template_clarification_needed":
            return False

        query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
        follow_up_questions = self._ensure_list(payload.get("follow_up_questions"))
        if not follow_up_questions:
            return False

        follow_up_focus = self._ensure_list(query.get("follow_up_focus"))
        trust_boundary_values: List[Any] = [query.get("summary")]
        trust_boundary_values.extend(self._ensure_list(query.get("keywords")))
        trust_boundary_values.extend(follow_up_focus)
        if any(
            self._contains_procedure_code_gloss(value)
            for value in trust_boundary_values
        ):
            return False

        summary = self._clean_card_value(query.get("summary"))
        location = self._clean_card_value(query.get("location"))
        if not summary or not location or summary == location:
            return False

        summary_zip = self._extract_zip_code(summary)
        location_zip = self._extract_zip_code(location)
        if not summary_zip or not location_zip or summary_zip != location_zip:
            return False

        return any(
            question.startswith("What kind of care do you need")
            for question in follow_up_questions
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _content_from_completion_choice(first_choice: Any) -> Optional[str]:
        message = getattr(first_choice, "message", None)
        content = None

        if isinstance(message, dict):
            content = message.get("content")
        elif message is not None:
            content = getattr(message, "content", None)

        if content is None and hasattr(first_choice, "text"):
            content = getattr(first_choice, "text")

        if isinstance(content, list):
            text_chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    text_chunks.append(item)
                elif isinstance(item, dict):
                    text_chunks.append(str(item.get("text", "")))
            content = "".join(text_chunks)

        if content is not None and not isinstance(content, str):
            content = str(content)

        return content

    # ------------------------------------------------------------------
    def _compose_safe_fallback_response(
        self,
        payload: Dict[str, Any],
        response_language: Optional[str],
        template_key: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        query = payload.get("query", {})
        language_key = _resolved_supported_language_key(response_language)
        lines: List[str] = []

        primary_guidance = ""
        if template_key == "response_template_emergency":
            primary_guidance = (
                payload.get("emergency_guidance")
                or payload.get("care_setting_guidance")
                or ""
            )
        else:
            primary_guidance = payload.get("care_setting_guidance") or ""

        translated_primary_guidance = self._translate_deterministic_text(
            primary_guidance,
            language_key,
        )
        if translated_primary_guidance:
            lines.append(
                f"**{self._render_copy(language_key, 'care_route_label')}:** {translated_primary_guidance}"
            )

        translated_specialist_guidance = self._translate_deterministic_text(
            payload.get("specialist_plan_guidance") or "",
            language_key,
        )
        if translated_specialist_guidance:
            if lines:
                lines.append("")
            lines.append(
                f"**{self._render_copy(language_key, 'referral_note_label')}:** {translated_specialist_guidance}"
            )

        translated_notes = self._translate_deterministic_text(
            self._clean_card_value(payload.get("notes")),
            language_key,
        )
        if translated_notes:
            if lines:
                lines.append("")
            lines.append(
                f"**{self._render_copy(language_key, 'note_label')}:** {translated_notes}"
            )

        follow_up_questions = [
            self._translate_deterministic_text(question, language_key)
            for question in self._ensure_list(payload.get("follow_up_questions"))
        ]
        follow_up_questions = [question for question in follow_up_questions if question]
        if follow_up_questions:
            if lines:
                lines.append("")
            lines.extend(f"- {question}" for question in follow_up_questions)

        verification_guidance = self._translate_deterministic_text(
            payload.get("verification_guidance") or "",
            language_key,
        )
        if verification_guidance:
            if lines:
                lines.append("")
            lines.append(
                f"**{self._render_copy(language_key, 'before_contact_label')}:** {verification_guidance}"
            )

        if not lines:
            summary = self._clean_card_value(query.get("summary")) or "your care search"
            lines.append(
                self._render_copy(language_key, "results_intro", summary=summary)
            )
            if finish_reason == "length":
                lines.append(
                    f"**{self._render_copy(language_key, 'note_label')}:** "
                    f"{self._render_copy(language_key, 'verification_reminder')}"
                )

        return self._append_required_trust_guidance(
            "\n".join(lines).strip(),
            response_language,
        )


__all__ = ["CareLocatorAgent", "ParsedCareQuery"]
