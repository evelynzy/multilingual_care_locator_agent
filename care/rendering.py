"""Deterministic card rendering, copy tables, disclosure notes, curated fallbacks."""
from __future__ import annotations

import re
from dataclasses import asdict
from html import escape
from typing import Any, Dict, List, Optional

from provider_search import normalize_search_result
from care.language import (
    _is_unknown_response_language,
    _lookup_language_alias,
    _normalize_response_language,
)
from care.locales_loader import load_locales as _load_locales

_DETERMINISTIC_RENDER_COPY = {
    "english": {
        "results_intro": "Here are care navigation results for {summary}.",
        "language_unverified_note": "⚠️ We could not confirm that any of these providers speak {languages}. Please verify language availability when you contact them.",
        "result_title_fallback": "Result {index}",
        "care_route_label": "Care route",
        "referral_note_label": "Referral note",
        "fallback_resources_label": "Trusted resources and fallback options",
        "resource_region_label": "Region",
        "resource_description_label": "Details",
        "note_label": "Note",
        "before_contact_label": "Before you contact a provider",
        "specialty_type_label": "Specialty/type",
        "address_label": "Address",
        "phone_label": "Phone",
        "source_label": "Source",
        "website_label": "Website",
        "why_matched_label": "Why matched",
        "listed_insurance_label": "Listed insurance",
        "insurance_verification_label": "Insurance/network verification",
        "accepting_patients_label": "Accepting new patients",
        "appointment_availability_label": "Appointment availability",
        "next_step_label": "Next step",
        "care_directory_result": "Care directory result",
        "not_listed": "Not listed",
        "unknown_source": "Unknown source",
        "listed_insurance_suffix": "reported only; network participation is not verified here",
        "appointment_availability_value": "Not verified; call the provider to confirm.",
        "verification_reminder": "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
        "verification_reminder_short": "Call to confirm network status, referral needs, new-patient status, and appointment availability.",
        "matched_search_summary": "Relevant to your search for {summary}.",
        "listed_provider_type": "Listed under {taxonomy}.",
        "matched_available_result": "Matched available care directory result for the search criteria.",
        "informational_badge": "Informational",
        "network_unverified_badge": "Network unverified",
        "new_patients_unknown_badge": "New patients unknown",
        "appointments_unverified_badge": "Appointments unverified",
        "status_unverified": "unverified",
        "status_unknown": "unknown",
        "trust_label_source": "Source: {value}",
        "trust_label_insurance": "Insurance/network: {value}",
        "trust_label_new_patients": "New patients: {value}",
        "trust_label_medicare_opt_out": "Medicare opt-out: {value}",
        "medicare_opted_out": "opted out",
        "medicare_no_record": "no opt-out record found",
        "medicare_unknown": "unknown",
        "phi_notice": "🔒 I removed what looked like {types} from your message — I don't need it to help you find care.",
        "phi_type_labels": {
            "ssn": "a Social Security number",
            "phone": "a phone number",
            "email": "an email address",
            "date": "a date of birth",
            "id_number": "an ID number",
        },
    },
}

_DETERMINISTIC_RENDER_TRANSLATIONS = {
    "For same-day, non-emergency care, urgent care is usually the best fit.": {},
    "For routine or ongoing care, primary care is usually the best fit.": {},
    "For a known specialty or referral need, a specialist is usually the right route.": {},
    "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.": {},
    "Insurance/network participation is not confirmed by source data.": {},
    "Source data does not confirm new-patient availability.": {},
    "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.": {},
    "Provider search sources were temporarily unavailable. Showing trusted fallback resources when available.": {},
    "What city and state or ZIP code should I search?": {},
    "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?": {},
    "Which insurance plan should I use when tailoring listed-insurance guidance?": {},
    "Do you want a provider who speaks a specific language?": {},
    "What plan type do you have, if you want me to tailor referral guidance (for example HMO, PPO, or POS)?": {},
    "If symptoms are severe or life-threatening, call emergency services now or go to the nearest emergency room.": {},
}

# Non-English strings live in committed locale files (care/locales/*.json),
# generated from the English masters above by care.generate_locales. English
# is the single hardcoded source; edit English, re-run the generator.
_LOADED_LOCALES = _load_locales()
for _language_key, _locale in _LOADED_LOCALES.items():
    _DETERMINISTIC_RENDER_COPY[_language_key] = _locale["copy"]
    for _sentence, _translated in _locale["sentences"].items():
        _DETERMINISTIC_RENDER_TRANSLATIONS.setdefault(_sentence, {})[_language_key] = _translated


def _resolved_supported_language_key(response_language: Optional[str]) -> str:
    if _is_unknown_response_language(response_language):
        return "english"

    language_key = _lookup_language_alias(_normalize_response_language(response_language))
    if language_key not in _DETERMINISTIC_RENDER_COPY:
        return "english"
    return language_key


def _phi_notice_line(phi_types, language_key):
    """One-line notice naming the redacted PHI types, in the resolved copy language."""
    copy = _DETERMINISTIC_RENDER_COPY.get(language_key) or _DETERMINISTIC_RENDER_COPY["english"]
    labels_map = copy.get("phi_type_labels") or _DETERMINISTIC_RENDER_COPY["english"]["phi_type_labels"]
    seen = []
    for phi_type in phi_types:
        label = labels_map.get(phi_type) or _DETERMINISTIC_RENDER_COPY["english"]["phi_type_labels"].get(phi_type, phi_type)
        if label not in seen:
            seen.append(label)
    template = copy.get("phi_notice") or _DETERMINISTIC_RENDER_COPY["english"]["phi_notice"]
    return template.format(types=", ".join(seen))


def _reply_localization_target(response_language: Optional[str]) -> Optional[str]:
    """The display language to LLM-localize a results reply into, or None if not needed.

    All seven known languages (English natively; es/zh/ar/ko/vi/tl from committed
    locale files) render deterministically and need nothing. Only long-tail
    languages (Czech, ...) get the LLM wrapper-translation pass, so the reply
    still reaches the user in their language while provider data stays verbatim.
    """
    if _is_unknown_response_language(response_language):
        return None
    if _resolved_supported_language_key(response_language) != "english":
        return None
    if _lookup_language_alias(_normalize_response_language(response_language)) == "english":
        return None
    return str(response_language).strip() or None


class RenderingMixin:
    # ------------------------------------------------------------------
    @staticmethod
    def _unverified_preferred_languages(
        requested_languages: Any, provider_results: Any
    ) -> List[str]:
        """Requested preferred languages that no returned provider is confirmed to speak.

        NPI records rarely carry language data, so a requested language usually cannot
        be confirmed. Returning the unconfirmed languages lets the reply disclose the
        unmet need instead of silently presenting non-matching providers (finding F6).
        """
        requested = [
            str(lang).strip() for lang in (requested_languages or []) if str(lang).strip()
        ]
        if not requested:
            return []
        spoken = set()
        for result in provider_results or []:
            provider = getattr(result, "provider", None)
            for lang in getattr(provider, "languages", None) or ():
                spoken.add(str(lang).strip().casefold())
        return [lang for lang in requested if lang.casefold() not in spoken]

    # ------------------------------------------------------------------
    def _compose_result_card_response(self, payload: Dict[str, Any]) -> str:
        query = payload.get("query", {})
        response_language = (
            query.get("response_language")
            or query.get("detected_language")
            or "English"
        )
        language_key = _resolved_supported_language_key(response_language)
        summary = self._clean_card_value(query.get("summary")) or "your care search"
        local_results = list(payload.get("local_results") or [])
        fallback_results = list(payload.get("fallback_results") or [])

        lines = [self._render_copy(language_key, "results_intro", summary=summary)]

        unverified_languages = payload.get("language_unverified")
        if unverified_languages:
            languages_text = ", ".join(
                str(lang).strip().title()
                for lang in unverified_languages
                if str(lang).strip()
            )
            if languages_text:
                lines.extend(
                    [
                        "",
                        self._render_copy(
                            language_key,
                            "language_unverified_note",
                            languages=languages_text,
                        ),
                    ]
                )

        care_setting_guidance = self._translate_deterministic_text(
            payload.get("care_setting_guidance")
            or "",
            language_key,
        )
        if care_setting_guidance:
            lines.extend(
                [
                    "",
                    f"**{self._render_copy(language_key, 'care_route_label')}:** {care_setting_guidance}",
                ]
            )

        specialist_guidance = self._translate_deterministic_text(
            payload.get("specialist_plan_guidance")
            or "",
            language_key,
        )
        if specialist_guidance:
            lines.extend(
                [
                    "",
                    f"**{self._render_copy(language_key, 'referral_note_label')}:** {specialist_guidance}",
                ]
            )

        notes = self._clean_card_value(payload.get("notes"))
        if notes:
            lines.extend(
                ["", f"**{self._render_copy(language_key, 'note_label')}:** {notes}"]
            )

        for index, result in enumerate(local_results, start=1):
            if isinstance(result, dict):
                lines.extend(
                    [
                        "",
                        self._format_provider_result_card(
                            result,
                            index,
                            query,
                            language_key=language_key,
                        ),
                    ]
                )

        if fallback_results:
            lines.extend(
                [
                    "",
                    f"**{self._render_copy(language_key, 'fallback_resources_label')}:**",
                ]
            )
            for index, result in enumerate(fallback_results, start=1):
                if isinstance(result, dict):
                    lines.extend(
                        [
                            "",
                            self._format_fallback_resource_entry(
                                result,
                                index,
                                language_key=language_key,
                            ),
                        ]
                    )

        verification_guidance = self._translate_deterministic_text(
            payload.get("verification_guidance") or "",
            language_key,
        )
        if verification_guidance and local_results:
            lines.extend(
                [
                    "",
                    f"**{self._render_copy(language_key, 'before_contact_label')}:** {verification_guidance}",
                ]
            )

        return self._append_required_trust_guidance(
            "\n".join(lines).strip(),
            response_language,
        )

    # ------------------------------------------------------------------
    def _format_fallback_resource_entry(
        self,
        result: Dict[str, Any],
        index: int,
        language_key: str = "english",
    ) -> str:
        name = self._escape_markdown_text(
            self._clean_card_value(result.get("name")) or self._render_copy(
                language_key,
                "result_title_fallback",
                index=index,
            )
        )
        location = self._escape_markdown_text(self._clean_card_value(result.get("location")))
        website = self._format_visible_website_value(result.get("website"))
        description = self._escape_markdown_text(self._clean_card_value(result.get("description")))
        provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
        source = self._escape_markdown_text(
            self._clean_card_value(provenance.get("source"))
            or self._clean_card_value(result.get("source"))
            or self._render_copy(language_key, "unknown_source")
        )

        details = []
        if location:
            details.append(
                f"{self._render_copy(language_key, 'resource_region_label')}: {location}"
            )
        details.append(f"{self._render_copy(language_key, 'source_label')}: {source}")
        if website:
            details.append(f"{self._render_copy(language_key, 'website_label')}: {website}")
        if description:
            details.append(
                f"{self._render_copy(language_key, 'resource_description_label')}: {description}"
            )

        resource_entry = f"{index}. **{name}**"
        if details:
            resource_entry = f"{resource_entry}: {'; '.join(details)}"
        return resource_entry

    # ------------------------------------------------------------------
    def _format_visible_website_value(self, value: object) -> str:
        cleaned_value = self._clean_card_value(value)
        if not cleaned_value:
            return ""

        if re.match(r"^https?://\S+$", cleaned_value, re.IGNORECASE):
            return cleaned_value

        return self._escape_markdown_text(cleaned_value)

    # ------------------------------------------------------------------
    def _format_provider_result_card(
        self,
        result: Dict[str, Any],
        index: int,
        query: Dict[str, Any],
        language_key: str = "english",
    ) -> str:
        name = self._clean_card_value(result.get("name")) or self._render_copy(
            language_key,
            "result_title_fallback",
            index=index,
        )
        specialty = self._clean_subtitle_fragment(
            self._result_specialty_label(result),
            kind="specialty",
        )
        location = self._clean_subtitle_fragment(
            self._result_location_label(result),
            kind="location",
        )
        phone = self._clean_card_value(result.get("phone"))
        provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
        source = (
            self._clean_card_value(provenance.get("source"))
            or self._clean_card_value(result.get("source"))
            or self._render_copy(language_key, "unknown_source")
        )
        insurance = self._ensure_list(result.get("insurance_reported"))
        why_matched = self._result_match_reason(result, query, language_key)
        listed_insurance = (
            f"{', '.join(insurance)} ({self._render_copy(language_key, 'listed_insurance_suffix')})"
            if insurance
            else ""
        )
        insurance_verification = result.get("insurance_network_verification")
        insurance_status = self._verification_status_label(
            insurance_verification,
            "unverified",
            language_key,
        )
        new_patient_verification = result.get("accepting_new_patients_status")
        new_patient_status = self._verification_status_label(
            new_patient_verification,
            "unknown",
            language_key,
        )
        appointment_status = self._render_copy(
            language_key,
            "appointment_availability_value",
        )
        verification_reminder = self._render_copy(language_key, "verification_reminder_short")

        subtitle_parts = self._dedupe_preserve_order(
            [part for part in [specialty, location] if part]
        )
        phone_source_parts = [
            self._render_card_meta_item(
                self._render_copy(language_key, "phone_label"),
                phone or self._render_copy(language_key, "not_listed"),
            ),
            self._render_card_meta_item(
                self._render_copy(language_key, "source_label"),
                source,
            ),
        ]

        explicit_trust_badges = [
            self._render_copy(language_key, "informational_badge"),
            self._render_copy(language_key, "network_unverified_badge"),
            self._render_copy(language_key, "new_patients_unknown_badge"),
            self._render_copy(language_key, "appointments_unverified_badge"),
        ]
        dynamic_trust_badges = [
            self._translate_trust_label(label, language_key)
            for label in self._ensure_list(result.get("trust_labels"))
            if self._translate_trust_label(label, language_key)
        ]
        dynamic_trust_badges = [
            label
            for label in dynamic_trust_badges
            if label
            and label not in {
                self._render_copy(language_key, "trust_label_source", value=source),
                self._render_copy(
                    language_key,
                    "trust_label_insurance",
                    value=self._render_copy(language_key, "status_unverified"),
                ),
                self._render_copy(
                    language_key,
                    "trust_label_new_patients",
                    value=self._render_copy(language_key, "status_unknown"),
                ),
                self._render_copy(
                    language_key,
                    "trust_label_medicare_opt_out",
                    value=self._render_copy(language_key, "medicare_unknown"),
                ),
            }
        ]
        trust_badges = self._dedupe_preserve_order(
            explicit_trust_badges + dynamic_trust_badges
        )
        subtitle_html = (
            f'    <div class="provider-card__subtitle">{escape(" • ".join(subtitle_parts))}</div>'
            if subtitle_parts
            else ""
        )

        card_lines = [
            '<div class="provider-card">',
            '  <div class="provider-card__header">',
            f'    <div class="provider-card__title">{escape(f"{index}. {name}")}</div>',
        ]
        if subtitle_html:
            card_lines.append(subtitle_html)
        card_detail_lines = [
            f'    <div class="provider-card__detail"><span class="provider-card__label">{escape(self._render_copy(language_key, "why_matched_label"))}</span><span class="provider-card__value">{escape(why_matched)}</span></div>',
        ]
        if listed_insurance:
            card_detail_lines.append(
                f'    <div class="provider-card__detail"><span class="provider-card__label">{escape(self._render_copy(language_key, "listed_insurance_label"))}</span><span class="provider-card__value">{escape(listed_insurance)}</span></div>'
            )
        if self._should_show_verification_detail(
            insurance_verification,
            default_status="unverified",
            default_basis="Insurance/network participation is not confirmed by source data.",
        ):
            card_detail_lines.append(
                f'    <div class="provider-card__detail"><span class="provider-card__label">{escape(self._render_copy(language_key, "insurance_verification_label"))}</span><span class="provider-card__value">{escape(insurance_status)}</span></div>'
            )
        if self._should_show_verification_detail(
            new_patient_verification,
            default_status="unknown",
            default_basis="Source data does not confirm new-patient availability.",
        ):
            card_detail_lines.append(
                f'    <div class="provider-card__detail"><span class="provider-card__label">{escape(self._render_copy(language_key, "accepting_patients_label"))}</span><span class="provider-card__value">{escape(new_patient_status)}</span></div>'
            )
        card_detail_lines.append(
            f'    <div class="provider-card__detail"><span class="provider-card__label">{escape(self._render_copy(language_key, "appointment_availability_label"))}</span><span class="provider-card__value">{escape(appointment_status)}</span></div>'
        )
        card_lines.extend(
            [
                "  </div>",
                f'  <div class="provider-card__meta">{"".join(phone_source_parts)}</div>',
                f'  <div class="provider-card__trust-row">{"".join(self._render_card_badge(label) for label in trust_badges)}</div>',
                '  <div class="provider-card__body">',
                *card_detail_lines,
                "  </div>",
                f'  <div class="provider-card__footer"><span class="provider-card__label">{escape(self._render_copy(language_key, "next_step_label"))}</span><span class="provider-card__value">{escape(verification_reminder)}</span></div>',
                "</div>",
            ]
        )
        return "\n".join(card_lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _render_card_meta_item(label: str, value: str) -> str:
        return (
            '<span class="provider-card__meta-item">'
            f'<span class="provider-card__meta-label">{escape(label)}</span>'
            f'<span class="provider-card__meta-value">{escape(value)}</span>'
            "</span>"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _render_card_badge(label: str) -> str:
        return f'<span class="provider-card__badge">{escape(label)}</span>'

    # ------------------------------------------------------------------
    def _result_specialty_label(self, result: Dict[str, Any]) -> str:
        specialties = self._ensure_list(result.get("specialties"))
        if specialties:
            return ", ".join(specialties)
        for key in ("taxonomy", "provider_type", "description"):
            value = self._clean_card_value(result.get(key))
            if value:
                return value
        return ""

    # ------------------------------------------------------------------
    def _result_location_label(self, result: Dict[str, Any]) -> str:
        return self._clean_card_value(result.get("location")) or self._clean_card_value(
            result.get("address")
        )

    # ------------------------------------------------------------------
    def _clean_subtitle_fragment(self, value: Any, kind: str) -> str:
        cleaned_value = self._clean_card_value(value)
        if not cleaned_value:
            return ""

        normalized = re.sub(r"\s+", " ", cleaned_value).strip()
        if len(normalized) <= 1:
            return ""

        if kind == "location":
            has_structural_detail = (
                "," in normalized
                or " " in normalized
                or any(character.isdigit() for character in normalized)
            )
            if not has_structural_detail and normalized.isupper() and len(normalized) <= 4:
                return ""
            return normalized

        if kind == "specialty":
            if re.fullmatch(r"[A-Za-z]\.?", normalized):
                return ""
            return normalized

        return normalized

    # ------------------------------------------------------------------
    def _result_match_reason(
        self,
        result: Dict[str, Any],
        query: Dict[str, Any],
        language_key: str = "english",
    ) -> str:
        summary = self._clean_card_value(query.get("summary"))
        if summary:
            return self._render_copy(
                language_key,
                "matched_search_summary",
                summary=summary,
            )

        taxonomy = self._clean_card_value(result.get("taxonomy"))
        if taxonomy:
            return self._render_copy(
                language_key,
                "listed_provider_type",
                taxonomy=taxonomy,
            )

        description = self._clean_card_value(result.get("description"))
        if description:
            return description

        return self._render_copy(language_key, "matched_available_result")

    # ------------------------------------------------------------------
    def _verification_status_label(
        self,
        value: Any,
        default: str,
        language_key: str = "english",
    ) -> str:
        if isinstance(value, dict):
            status = str(value.get("status") or default)
            basis = value.get("basis")
            localized_status = self._translate_status_value(
                status,
                language_key,
            )
            if basis:
                return f"{localized_status} ({self._translate_deterministic_text(str(basis), language_key)})"
            return localized_status
        return self._translate_status_value(str(default), language_key)

    # ------------------------------------------------------------------
    def _translate_status_value(self, status: str, language_key: str) -> str:
        normalized_status = str(status).strip().lower().replace("_", " ")
        if normalized_status == "unverified":
            return self._render_copy(language_key, "status_unverified")
        if normalized_status == "unknown":
            return self._render_copy(language_key, "status_unknown")
        return str(status)

    # ------------------------------------------------------------------
    @staticmethod
    def _should_show_verification_detail(
        value: Any,
        default_status: str,
        default_basis: str,
    ) -> bool:
        if not isinstance(value, dict):
            return False

        normalized_status = str(value.get("status") or default_status).strip().lower()
        normalized_status = normalized_status.replace("_", " ")
        expected_status = str(default_status).strip().lower().replace("_", " ")
        if normalized_status != expected_status:
            return True

        basis = str(value.get("basis") or "").strip()
        return bool(basis and basis != default_basis)

    # ------------------------------------------------------------------
    @staticmethod
    def _render_copy(language_key: str, key: str, **kwargs: Any) -> str:
        copy = _DETERMINISTIC_RENDER_COPY.get(language_key) or _DETERMINISTIC_RENDER_COPY["english"]
        template = copy.get(key) or _DETERMINISTIC_RENDER_COPY["english"].get(key, key)
        return template.format(**kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _translate_deterministic_text(text: str, language_key: str) -> str:
        cleaned_text = str(text).strip()
        if not cleaned_text:
            return ""
        translations = _DETERMINISTIC_RENDER_TRANSLATIONS.get(cleaned_text, {})
        return translations.get(language_key, cleaned_text)

    # ------------------------------------------------------------------
    def _translate_trust_label(self, label: str, language_key: str) -> str:
        cleaned_label = str(label).strip()
        if not cleaned_label:
            return ""
        if cleaned_label.startswith("Source: "):
            source_value = cleaned_label[len("Source: ") :]
            return self._render_copy(
                language_key,
                "trust_label_source",
                value=source_value,
            )
        if cleaned_label.startswith("Insurance/network: "):
            insurance_value = cleaned_label[len("Insurance/network: ") :]
            return self._render_copy(
                language_key,
                "trust_label_insurance",
                value=self._translate_status_value(
                    insurance_value,
                    language_key,
                ),
            )
        if cleaned_label.startswith("New patients: "):
            new_patient_value = cleaned_label[len("New patients: ") :]
            return self._render_copy(
                language_key,
                "trust_label_new_patients",
                value=self._translate_status_value(
                    new_patient_value,
                    language_key,
                ),
            )
        if cleaned_label.startswith("Medicare opt-out: "):
            medicare_value = cleaned_label[len("Medicare opt-out: ") :]
            localized_value = {
                "opted out": self._render_copy(language_key, "medicare_opted_out"),
                "no opt-out record found": self._render_copy(language_key, "medicare_no_record"),
                "unknown": self._render_copy(language_key, "medicare_unknown"),
            }.get(medicare_value, medicare_value)
            return self._render_copy(
                language_key,
                "trust_label_medicare_opt_out",
                value=localized_value,
            )
        return self._translate_deterministic_text(cleaned_label, language_key)

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_card_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _escape_markdown_text(value: str) -> str:
        if not value:
            return ""
        escaped = value.replace("\\", "\\\\")
        return re.sub(r"([`*_{}\[\]()#+.!|>~-])", r"\\\1", escaped)

    # ------------------------------------------------------------------
    def _trusted_resource_fallback(self, query: ParsedCareQuery) -> List[dict]:
        region_hint = query.location or "international"
        if not self.fallback_resources:
            return []

        care_setting = self._classify_care_setting(
            query,
            self._combined_request_text(query, query.summary),
        )
        query_terms = {
            term.lower()
            for term in (query.specialties + query.keywords)
            if isinstance(term, str)
        }
        if care_setting:
            query_terms.add(care_setting)
        region_contexts = self._fallback_region_contexts(query)

        suggestions: List[dict] = []
        for resource in self.fallback_resources:
            name = resource.get("name")
            url = resource.get("url")
            description = resource.get("description")
            if not name or not url:
                continue

            specialty_filters = [
                str(item).lower() for item in resource.get("specialties", []) if item
            ]
            care_setting_filters = [
                str(item).lower() for item in resource.get("care_settings", []) if item
            ]
            region_filters = [
                str(item).lower() for item in resource.get("regions", []) if item
            ]

            if specialty_filters:
                if not query_terms:
                    continue
                if not any(filter_term in query_terms for filter_term in specialty_filters):
                    continue

            if care_setting_filters and care_setting not in care_setting_filters:
                continue

            if region_filters:
                if region_contexts:
                    if not any(
                        filter_term in context
                        for filter_term in region_filters
                        for context in region_contexts
                    ):
                        continue
                else:
                    # Location is unknown: include national/US-scoped resources
                    # as well as globally-scoped ones so a US user who did not
                    # provide a parseable location still sees national links.
                    _UNKNOWN_LOCATION_INCLUDED = {
                        "international", "global", "united states", "usa", "us"
                    }
                    if not any(
                        filter_term in _UNKNOWN_LOCATION_INCLUDED
                        for filter_term in region_filters
                    ):
                        continue

            suggestions.append(
                {
                    "name": name,
                    "location": region_hint,
                    "website": url,
                    "description": description,
                    "source": "Trusted public directories",
                }
            )

        return suggestions

    # ------------------------------------------------------------------
    def _fallback_region_contexts(self, query: ParsedCareQuery) -> set[str]:
        location_text = (query.location or "").strip().lower()
        if not location_text:
            return set()

        contexts = {location_text}
        if self._extract_state_code(query.location) or self._extract_zip_code(query.location):
            contexts.update({"united states", "usa", "us"})
        return contexts

    # ------------------------------------------------------------------
    @staticmethod
    def _provider_search_result_to_payload(result: Any) -> dict:
        normalized_result = normalize_search_result(asdict(result))
        provider_payload = asdict(normalized_result.provider)
        provider_payload["provider_id"] = normalized_result.provider.provider_id
        provider_payload["source"] = normalized_result.source or normalized_result.provider.source
        provider_payload["location"] = normalized_result.provider.location_summary
        if normalized_result.score is not None:
            provider_payload["score"] = normalized_result.score
        if normalized_result.retriever_metadata:
            provider_payload["retriever_metadata"] = dict(normalized_result.retriever_metadata)
        return provider_payload

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_resource_to_payload(resource: Any, query: ParsedCareQuery) -> dict:
        resource_dict = asdict(resource)
        return {
            "name": resource_dict.get("name"),
            "location": query.location or "international",
            "website": resource_dict.get("url"),
            "description": resource_dict.get("description"),
            "source": resource_dict.get("source") or "Trusted public directories",
            "provenance": {
                "source": resource_dict.get("source") or "Trusted public directories",
            },
        }

    # ------------------------------------------------------------------
    def _normalize_result_trust_metadata(self, result: dict) -> dict:
        if not isinstance(result, dict):
            return result

        normalized = dict(result)

        provenance = normalized.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        source_value = normalized.get("source")
        if isinstance(source_value, str) and source_value.strip():
            provenance.setdefault("source", source_value.strip())
        else:
            provenance.setdefault("source", "Unknown source")

        normalized["provenance"] = provenance

        reported_insurance = normalized.pop("insurance", None)
        if reported_insurance is None:
            reported_insurance = normalized.pop("accepted_insurance", None)
        if reported_insurance is None:
            reported_insurance = normalized.get("insurance_reported")
        normalized["insurance_reported"] = self._ensure_list(reported_insurance)

        insurance_verification = normalized.get("insurance_network_verification")
        if not isinstance(insurance_verification, dict):
            insurance_verification = {}
        insurance_verification.setdefault("status", "unverified")
        insurance_verification.setdefault("verified", False)
        insurance_verification.setdefault(
            "basis",
            "Insurance/network participation is not confirmed by source data.",
        )
        insurance_verification.setdefault("source", provenance["source"])
        normalized["insurance_network_verification"] = insurance_verification

        new_patient_status = normalized.get("accepting_new_patients_status")
        if not isinstance(new_patient_status, dict):
            new_patient_status = {}
        new_patient_status.setdefault("status", "unknown")
        new_patient_status.setdefault("verified", False)
        new_patient_status.setdefault(
            "basis",
            "Source data does not confirm new-patient availability.",
        )
        new_patient_status.setdefault("source", provenance["source"])
        normalized["accepting_new_patients_status"] = new_patient_status

        normalized.setdefault(
            "insurance_display",
            "Reported/listed insurance (not verified).",
        )
        normalized["trust_labels"] = self._build_result_trust_labels(normalized)

        return normalized

    # ------------------------------------------------------------------
    @staticmethod
    def _build_result_trust_labels(result: dict) -> List[str]:
        result_data = result if isinstance(result, dict) else {}
        provenance = result_data.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        source = provenance.get("source") or result_data.get("source") or "Unknown source"
        labels = [f"Source: {source}"]

        insurance_verification = result_data.get("insurance_network_verification")
        if not isinstance(insurance_verification, dict):
            insurance_verification = {}
        insurance_status = str(
            insurance_verification.get("status") or "unverified"
        ).replace("_", " ")
        labels.append(f"Insurance/network: {insurance_status}")

        new_patient_status = result_data.get("accepting_new_patients_status")
        if not isinstance(new_patient_status, dict):
            new_patient_status = {}
        accepting_status = str(new_patient_status.get("status") or "unknown").replace(
            "_", " "
        )
        labels.append(f"New patients: {accepting_status}")

        medicare_opt_out = result_data.get("medicare_opt_out")
        if isinstance(medicare_opt_out, dict):
            opted_out = medicare_opt_out.get("opted_out")
            if opted_out is True:
                labels.append("Medicare opt-out: opted out")
            elif opted_out is False:
                labels.append("Medicare opt-out: no opt-out record found")
            else:
                labels.append("Medicare opt-out: unknown")
        else:
            labels.append("Medicare opt-out: unknown")

        return labels

    # ------------------------------------------------------------------
    @staticmethod
    def _verification_guidance() -> str:
        return (
            "Call the provider and insurer to confirm network status, accepted insurance plan, "
            "referral requirements, new-patient availability, location, and "
            "appointment availability."
        )
