from __future__ import annotations

import json
import os
import re
from html import escape
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import logging

from huggingface_hub import InferenceClient

from config_loader import get_prompt, get_search_settings
from provider_search import (
    ProviderSearchRequest,
    ProviderSearchService,
    SQLiteProviderSearchCache,
    normalize_search_result,
)
from provider_search.sources import (
    ClinicalTablesDatasetConfig,
    ClinicalTablesSource,
    NPPESSource,
)
from provider_search.sources.clinicaltables import DEFAULT_DATASET_CONFIGS
from provider_search.specialty_families import (
    normalize_query_specialty_family_id,
)

from care.language import (
    LanguageMixin,
    _NORMALIZED_TRUST_GUIDANCE_ALIASES,
    _REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES,
    _UNKNOWN_LANGUAGE_MARKERS,
    _is_unknown_response_language,
    _lookup_language_alias,
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

logger = logging.getLogger(__name__)


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
    },
    "spanish": {
        "results_intro": "Aquí están los resultados de navegación de atención para {summary}.",
        "language_unverified_note": "⚠️ No pudimos confirmar que alguno de estos proveedores hable {languages}. Verifique la disponibilidad de idioma al contactarlos.",
        "result_title_fallback": "Resultado {index}",
        "care_route_label": "Ruta de atención",
        "referral_note_label": "Nota sobre remisión",
        "fallback_resources_label": "Recursos confiables y opciones de respaldo",
        "resource_region_label": "Región",
        "resource_description_label": "Detalles",
        "note_label": "Nota",
        "before_contact_label": "Antes de contactar a un proveedor",
        "specialty_type_label": "Especialidad/tipo",
        "address_label": "Dirección",
        "phone_label": "Teléfono",
        "source_label": "Fuente",
        "website_label": "Sitio web",
        "why_matched_label": "Por qué coincide",
        "listed_insurance_label": "Seguro listado",
        "insurance_verification_label": "Verificación de seguro/red",
        "accepting_patients_label": "Acepta pacientes nuevos",
        "appointment_availability_label": "Disponibilidad de citas",
        "next_step_label": "Siguiente paso",
        "care_directory_result": "Resultado del directorio de atención",
        "not_listed": "No figura",
        "unknown_source": "Fuente desconocida",
        "listed_insurance_suffix": "solo informado; la participación en la red no está verificada aquí",
        "appointment_availability_value": "No verificada; llame al proveedor para confirmarla.",
        "verification_reminder": "Llame al proveedor y a la aseguradora para confirmar el estado de la red, el plan de seguro aceptado, los requisitos de remisión, la disponibilidad para pacientes nuevos, la ubicación y la disponibilidad de citas.",
        "verification_reminder_short": "Llame para confirmar la red, la necesidad de remisión, si aceptan pacientes nuevos y la disponibilidad de citas.",
        "matched_search_summary": "Relacionado con su búsqueda de {summary}.",
        "listed_provider_type": "Figura como {taxonomy}.",
        "matched_available_result": "Coincidió con un resultado disponible del directorio de atención para los criterios de búsqueda.",
        "informational_badge": "Informativo",
        "network_unverified_badge": "Red no verificada",
        "new_patients_unknown_badge": "Pacientes nuevos desconocido",
        "appointments_unverified_badge": "Citas sin verificar",
        "status_unverified": "no verificado",
        "status_unknown": "desconocido",
        "trust_label_source": "Fuente: {value}",
        "trust_label_insurance": "Seguro/red: {value}",
        "trust_label_new_patients": "Pacientes nuevos: {value}",
        "trust_label_medicare_opt_out": "Exclusión de Medicare: {value}",
        "medicare_opted_out": "excluido",
        "medicare_no_record": "sin registro de exclusión",
        "medicare_unknown": "desconocido",
    },
    "simplified_chinese": {
        "results_intro": "{summary}的护理导航结果如下。",
        "language_unverified_note": "⚠️ 我们无法确认这些医疗服务者会说{languages}。请在联系时自行确认语言服务。",
        "result_title_fallback": "结果{index}",
        "care_route_label": "就医路线",
        "referral_note_label": "转诊提示",
        "fallback_resources_label": "可信资源与备用选项",
        "resource_region_label": "适用地区",
        "resource_description_label": "说明",
        "note_label": "说明",
        "before_contact_label": "联系机构前",
        "specialty_type_label": "专科/类型",
        "address_label": "地址",
        "phone_label": "电话",
        "source_label": "来源",
        "website_label": "网站",
        "why_matched_label": "匹配原因",
        "listed_insurance_label": "列出的保险",
        "insurance_verification_label": "保险/网络验证",
        "accepting_patients_label": "是否接收新患者",
        "appointment_availability_label": "预约可用性",
        "next_step_label": "下一步",
        "care_directory_result": "护理目录结果",
        "not_listed": "未列出",
        "unknown_source": "来源未知",
        "listed_insurance_suffix": "仅为列出信息；此处未验证网络参与情况",
        "appointment_availability_value": "尚未验证；请致电服务提供者确认。",
        "verification_reminder": "请致电服务提供者和保险公司，确认网络状态、接受的保险计划、转诊要求、新患者接收情况、地点和预约可用性。",
        "verification_reminder_short": "请致电确认网络状态、转诊要求、新患者接收情况和预约可用性。",
        "matched_search_summary": "与您搜索的{summary}相关。",
        "listed_provider_type": "列为{taxonomy}。",
        "matched_available_result": "根据搜索条件匹配到可用的护理目录结果。",
        "informational_badge": "信息性匹配",
        "network_unverified_badge": "网络未验证",
        "new_patients_unknown_badge": "新患者情况未知",
        "appointments_unverified_badge": "预约未验证",
        "status_unverified": "未验证",
        "status_unknown": "未知",
        "trust_label_source": "来源：{value}",
        "trust_label_insurance": "保险/网络：{value}",
        "trust_label_new_patients": "新患者：{value}",
        "trust_label_medicare_opt_out": "Medicare退出：{value}",
        "medicare_opted_out": "已退出",
        "medicare_no_record": "未找到退出记录",
        "medicare_unknown": "未知",
    },
}

_DETERMINISTIC_RENDER_TRANSLATIONS = {
    "For same-day, non-emergency care, urgent care is usually the best fit.": {
        "spanish": "Para atención el mismo día que no sea una emergencia, urgent care suele ser la mejor opción.",
        "simplified_chinese": "对于当天且非紧急的就医需求，急诊门诊通常更合适。",
    },
    "For routine or ongoing care, primary care is usually the best fit.": {
        "spanish": "Para la atención rutinaria o continua, la atención primaria suele ser la mejor opción.",
        "simplified_chinese": "对于常规或持续性的就医需求，初级保健通常更合适。",
    },
    "For a known specialty or referral need, a specialist is usually the right route.": {
        "spanish": "Para una necesidad conocida de especialista o remisión, un especialista suele ser la ruta correcta.",
        "simplified_chinese": "如果已经明确需要某个专科或需要转诊，专科医生通常是合适的路线。",
    },
    "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.": {
        "spanish": "Para buscar especialistas, los planes HMO y POS suelen requerir una remisión de atención primaria; los PPO pueden no requerirla, pero debe confirmarlo con su aseguradora y los documentos del plan.",
        "simplified_chinese": "查找专科医生时，HMO 和 POS 计划通常需要初级保健医生转诊；PPO 计划可能不需要，但仍应与保险公司和计划文件确认。",
    },
    "Insurance/network participation is not confirmed by source data.": {
        "spanish": "Los datos de la fuente no confirman la participación en la red o el seguro.",
        "simplified_chinese": "来源数据未确认保险或网络参与情况。",
    },
    "Source data does not confirm new-patient availability.": {
        "spanish": "Los datos de la fuente no confirman la disponibilidad para pacientes nuevos.",
        "simplified_chinese": "来源数据未确认是否接收新患者。",
    },
    "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.": {
        "spanish": "Llame al proveedor y a la aseguradora para confirmar el estado de la red, el plan de seguro aceptado, los requisitos de remisión, la disponibilidad para pacientes nuevos, la ubicación y la disponibilidad de citas.",
        "simplified_chinese": "请致电服务提供者和保险公司，确认网络状态、接受的保险计划、转诊要求、新患者接收情况、地点和预约可用性。",
    },
    "Provider search sources were temporarily unavailable. Showing trusted fallback resources when available.": {
        "spanish": "Las fuentes de búsqueda de proveedores no estuvieron disponibles temporalmente. Se mostrarán recursos de respaldo confiables cuando estén disponibles.",
        "simplified_chinese": "提供者搜索来源暂时不可用。如有可信的备用资源，将优先显示。",
    },
    "What city and state or ZIP code should I search?": {
        "spanish": "¿Qué ciudad y estado o código postal debo buscar?",
        "simplified_chinese": "我应该搜索哪个城市和州，或哪个邮政编码？",
    },
    "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?": {
        "spanish": "¿Qué tipo de atención necesita (por ejemplo atención primaria, pediatría, dermatología, otorrinolaringología o atención urgente)?",
        "simplified_chinese": "您需要哪种类型的医疗服务（例如初级保健、儿科、皮肤科、耳鼻喉科或急诊门诊）？",
    },
    "Which insurance plan should I use when tailoring listed-insurance guidance?": {
        "spanish": "¿Qué plan de seguro debo usar para adaptar la orientación sobre el seguro listado?",
        "simplified_chinese": "在调整列出保险的说明时，我应该使用哪个保险计划？",
    },
    "Do you want a provider who speaks a specific language?": {
        "spanish": "¿Desea un proveedor que hable un idioma específico?",
        "simplified_chinese": "您是否希望服务提供者会说某种特定语言？",
    },
    "What plan type do you have, if you want me to tailor referral guidance (for example HMO, PPO, or POS)?": {
        "spanish": "Si desea que adapte la orientación sobre remisiones, ¿qué tipo de plan tiene (por ejemplo HMO, PPO o POS)?",
        "simplified_chinese": "如果您希望我调整转诊说明，您的计划类型是什么（例如 HMO、PPO 或 POS）？",
    },
    "If symptoms are severe or life-threatening, call emergency services now or go to the nearest emergency room.": {
        "spanish": "Si los síntomas son graves o ponen en peligro la vida, llame a los servicios de emergencia ahora o vaya a la sala de emergencias más cercana.",
        "simplified_chinese": "如果症状严重或危及生命，请立即拨打急救电话或前往最近的急诊室。",
    },
}


def _resolved_supported_language_key(response_language: Optional[str]) -> str:
    if _is_unknown_response_language(response_language):
        return "english"

    language_key = _lookup_language_alias(_normalize_response_language(response_language))
    if language_key not in _DETERMINISTIC_RENDER_COPY:
        return "english"
    return language_key


def _reply_localization_target(response_language: Optional[str]) -> Optional[str]:
    """The display language to LLM-localize a results reply into, or None if not needed.

    en/es/zh are rendered natively by the deterministic copy, and English needs nothing.
    Any other detected language (Czech, Korean, Arabic, ...) currently falls back to
    English copy — those get an LLM wrapper-translation pass so the reply reaches the
    user in their language while the provider data stays verbatim. Restores the
    any-language localization lost when provider cards became deterministic (df2362c).
    """
    if _is_unknown_response_language(response_language):
        return None
    if _resolved_supported_language_key(response_language) != "english":
        return None
    if _lookup_language_alias(_normalize_response_language(response_language)) == "english":
        return None
    return str(response_language).strip() or None



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

class CareLocatorAgent(LanguageMixin, SafetyMixin, IntentMixin):
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
