from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import logging

import requests
from huggingface_hub import InferenceClient

from config_loader import get_prompt, get_search_settings
from retriever import ProviderRepository, SearchCriteria

logger = logging.getLogger(__name__)


_REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE = {
    "english": (
        "Important safety and trust notes:\n"
        "- This tool supports care navigation only and does not diagnose, prescribe, or replace licensed medical advice.\n"
        "- Directory matches are informational, not referrals, endorsements, or guarantees of clinical fit.\n"
        "- Insurance/network participation, referral requirements, new-patient availability, location, and appointment availability are not verified unless the source explicitly says so. Call the provider and insurer to confirm before seeking care.\n"
        "- Do not share personal health information such as full names, addresses, Social Security numbers, or medical record numbers.\n"
        "- If symptoms are severe or life-threatening, call emergency services (911 in the U.S.) or go to the nearest emergency room."
    ),
    "spanish": (
        "Notas importantes de seguridad y confianza:\n"
        "- Esta herramienta solo apoya la navegación de atención y no diagnostica, receta ni reemplaza el consejo médico de profesionales con licencia.\n"
        "- Las coincidencias del directorio son informativas, no remisiones, endosos ni garantías de ajuste clínico.\n"
        "- La participación en la red del seguro, los requisitos de remisión, la disponibilidad para pacientes nuevos, la ubicación y la disponibilidad de citas no están verificadas a menos que la fuente lo indique explícitamente. Llame al proveedor y a la aseguradora para confirmar antes de buscar atención.\n"
        "- No comparta información personal de salud, como nombres completos, direcciones, números de Seguro Social o números de expediente médico.\n"
        "- Si los síntomas son graves o potencialmente mortales, llame a los servicios de emergencia (911 en EE. UU.) o vaya a la sala de emergencias más cercana."
    ),
    "simplified_chinese": (
        "重要的安全和信任提示：\n"
        "- 此工具仅支持护理导航，不会诊断、开药或替代持证医疗专业人员的建议。\n"
        "- 目录匹配结果仅供参考，不是转诊、背书或临床适配性的保证。\n"
        "- 除非来源明确说明，否则保险网络参与情况、转诊要求、新患者接收情况、地点和预约可用性均未经过验证。就医前请致电服务提供者和保险公司确认。\n"
        "- 请勿分享个人健康信息，例如全名、地址、社会安全号码或病历号码。\n"
        "- 如果症状严重或危及生命，请拨打紧急服务电话（美国为 911）或前往最近的急诊室。"
    ),
    "vietnamese": (
        "Ghi chú quan trọng về an toàn và độ tin cậy:\n"
        "- Công cụ này chỉ hỗ trợ điều hướng chăm sóc và không chẩn đoán, kê đơn hoặc thay thế lời khuyên y tế từ chuyên gia có giấy phép.\n"
        "- Các kết quả khớp trong danh bạ chỉ mang tính thông tin, không phải là giới thiệu, chứng thực hoặc bảo đảm phù hợp lâm sàng.\n"
        "- Việc tham gia mạng lưới bảo hiểm, yêu cầu giấy giới thiệu, tình trạng nhận bệnh nhân mới, địa điểm và lịch hẹn chưa được xác minh trừ khi nguồn nêu rõ. Hãy gọi cho nhà cung cấp và công ty bảo hiểm để xác nhận trước khi tìm kiếm dịch vụ chăm sóc.\n"
        "- Không chia sẻ thông tin sức khỏe cá nhân như họ tên đầy đủ, địa chỉ, số An sinh Xã hội hoặc số hồ sơ y tế.\n"
        "- Nếu triệu chứng nghiêm trọng hoặc đe dọa tính mạng, hãy gọi dịch vụ khẩn cấp (911 tại Hoa Kỳ) hoặc đến phòng cấp cứu gần nhất."
    ),
    "tagalog": (
        "Mahahalagang tala sa kaligtasan at pagtitiwala:\n"
        "- Ang tool na ito ay sumusuporta lamang sa paggabay sa paghahanap ng pangangalaga at hindi nagdi-diagnose, nagrereseta, o pumapalit sa payong medikal ng lisensiyadong propesyonal.\n"
        "- Ang mga tugma sa direktoryo ay para sa impormasyon lamang, hindi referral, pag-endorso, o garantiya ng klinikal na pagiging angkop.\n"
        "- Hindi beripikado ang paglahok sa insurance/network, mga kinakailangan sa referral, pagtanggap ng bagong pasyente, lokasyon, at availability ng appointment maliban kung tahasang sinasabi ng pinagmulan. Tawagan ang provider at insurer upang kumpirmahin bago humingi ng pangangalaga.\n"
        "- Huwag magbahagi ng personal na impormasyong pangkalusugan tulad ng buong pangalan, address, Social Security number, o medical record number.\n"
        "- Kung malubha o nagbabanta sa buhay ang mga sintomas, tumawag sa emergency services (911 sa U.S.) o pumunta sa pinakamalapit na emergency room."
    ),
    "arabic": (
        "ملاحظات مهمة حول السلامة والثقة:\n"
        "- تدعم هذه الأداة التنقل في الرعاية فقط ولا تشخص أو تصف أدوية أو تحل محل المشورة الطبية من مختصين مرخصين.\n"
        "- نتائج الدليل معلوماتية فقط، وليست إحالات أو تزكيات أو ضمانات للملاءمة السريرية.\n"
        "- لا يتم التحقق من المشاركة في شبكة التأمين، أو متطلبات الإحالة، أو توفر استقبال مرضى جدد، أو الموقع، أو توفر المواعيد ما لم يذكر المصدر ذلك صراحة. اتصل بمقدم الخدمة وشركة التأمين للتأكيد قبل طلب الرعاية.\n"
        "- لا تشارك معلومات صحية شخصية مثل الأسماء الكاملة أو العناوين أو أرقام الضمان الاجتماعي أو أرقام السجلات الطبية.\n"
        "- إذا كانت الأعراض شديدة أو مهددة للحياة، فاتصل بخدمات الطوارئ (911 في الولايات المتحدة) أو اذهب إلى أقرب غرفة طوارئ."
    ),
    "korean": (
        "중요한 안전 및 신뢰 안내:\n"
        "- 이 도구는 진료 탐색만 지원하며 진단, 처방 또는 면허가 있는 의료 전문가의 조언을 대체하지 않습니다.\n"
        "- 디렉터리 일치 결과는 정보 제공용일 뿐이며, 의뢰, 보증 또는 임상적 적합성 보장이 아닙니다.\n"
        "- 출처가 명시적으로 밝히지 않는 한 보험 네트워크 참여, 의뢰 요건, 신규 환자 접수 여부, 위치 및 예약 가능 여부는 확인된 것이 아닙니다. 진료를 받기 전에 제공자와 보험사에 전화해 확인하세요.\n"
        "- 전체 이름, 주소, 사회보장번호 또는 의무기록 번호와 같은 개인 건강 정보를 공유하지 마세요.\n"
        "- 증상이 심각하거나 생명을 위협하는 경우 응급 서비스(미국에서는 911)에 전화하거나 가장 가까운 응급실로 가세요."
    ),
}

_REQUIRED_TRUST_GUIDANCE = _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE["english"]

_REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "esp": "spanish",
    "espanol": "spanish",
    "spanish": "spanish",
    "zh": "simplified_chinese",
    "zh-cn": "simplified_chinese",
    "zh-hans": "simplified_chinese",
    "chinese": "simplified_chinese",
    "simplified chinese": "simplified_chinese",
    "mandarin": "simplified_chinese",
    "mandarin chinese": "simplified_chinese",
    "中文": "simplified_chinese",
    "简体中文": "simplified_chinese",
    "普通话": "simplified_chinese",
    "vi": "vietnamese",
    "vie": "vietnamese",
    "vietnamese": "vietnamese",
    "tiếng việt": "vietnamese",
    "tagalog": "tagalog",
    "filipino": "tagalog",
    "tl": "tagalog",
    "fil": "tagalog",
    "ar": "arabic",
    "ara": "arabic",
    "arabic": "arabic",
    "العربية": "arabic",
    "ko": "korean",
    "kor": "korean",
    "korean": "korean",
    "한국어": "korean",
}

_UNKNOWN_LANGUAGE_MARKERS = {
    "unknown",
    "undetected",
    "undetermined",
    "unspecified",
    "none",
    "null",
    "n/a",
}


def _normalize_response_language(response_language: Optional[str]) -> str:
    if not response_language:
        return ""

    normalized_language = unicodedata.normalize("NFKD", str(response_language).strip().lower())
    normalized_language = "".join(
        character for character in normalized_language if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized_language)


def _is_unknown_response_language(response_language: Optional[str]) -> bool:
    normalized_language = _normalize_response_language(response_language)
    return not normalized_language or normalized_language in _UNKNOWN_LANGUAGE_MARKERS


def _get_prewritten_required_trust_guidance(response_language: Optional[str]) -> Optional[str]:
    if _is_unknown_response_language(response_language):
        return _REQUIRED_TRUST_GUIDANCE

    normalized_language = _normalize_response_language(response_language)
    language_key = _REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES.get(normalized_language)

    if language_key is None:
        for alias, alias_language_key in _REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES.items():
            if (
                normalized_language.startswith(f"{alias} ")
                or normalized_language.startswith(f"{alias}-")
                or normalized_language.startswith(f"{alias} (")
            ):
                language_key = alias_language_key
                break

    if language_key is None:
        return None

    return _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[language_key]


def normalize_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "role" not in message or "content" not in message:
            continue
        normalized_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )
    return normalized_messages


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


class CareLocatorAgent:
    """Coordinates LLM reasoning, dataset search, and fallback lookups."""

    def __init__(self, provider_repository: Optional[ProviderRepository] = None) -> None:
        self.provider_repository = provider_repository or ProviderRepository()
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
        self.ctss_values_max_results = clinical.get("values_max_results", 10)

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
        self._npi_registry_cache: Dict[str, Optional[dict]] = {}

        self.ctss_dataset_configs: Dict[str, Dict[str, Any]] = {
            "npi_idv": {
                "search_url": clinical.get(
                    "individual_search_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/search",
                ),
                # The v3 ClinicalTables API no longer exposes dedicated values/fields endpoints
                # for NPI datasets. Keep the configuration keys optional and default them to None
                # so the agent skips those HTTP calls.
                "values_url": clinical.get("individual_values_url") or None,
                "fields_url": clinical.get("individual_fields_url") or None,
                "probe_term": field_probe_terms.get("npi_idv", "urology"),
                "source_label": "NPI Registry (individual)",
                "result_fields": [
                    "name.full",
                    "name.first",
                    "name.middle",
                    "name.last",
                    "name.prefix",
                    "name.suffix",
                    "NPI",
                    "provider_type",
                    "addr_practice.full",
                    "addr_practice.address_1",
                    "addr_practice.address_2",
                    "addr_practice.city",
                    "addr_practice.state",
                    "addr_practice.zip",
                    "addr_practice.country_name",
                    "addr_practice.phone",
                    "languages",
                ],
            },
            "npi_org": {
                "search_url": clinical.get(
                    "organization_search_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/search",
                ),
                "values_url": clinical.get("organization_values_url") or None,
                "fields_url": clinical.get("organization_fields_url") or None,
                "probe_term": field_probe_terms.get("npi_org", "clinic"),
                "source_label": "NPI Registry (organization)",
                "result_fields": [
                    "name.full",
                    "NPI",
                    "provider_type",
                    "addr_practice.full",
                    "addr_practice.address_1",
                    "addr_practice.address_2",
                    "addr_practice.city",
                    "addr_practice.state",
                    "addr_practice.zip",
                    "addr_practice.country_name",
                    "addr_practice.phone",
                    "languages",
                ],
            },
        }
        self._ctss_dataset_priority: List[str] = [
            dataset
            for dataset, cfg in self.ctss_dataset_configs.items()
            if cfg.get("search_url")
        ]

        self.fallback_resources = self.search_settings.get("fallback_resources", [])
        self._ctss_field_map: Dict[str, Dict[str, int]] = {}
        self._ctss_result_field_order: Dict[str, List[str]] = {}
        self._ctss_suggest_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._ctss_taxonomy_fields: List[str] = [
            "provider_type",
            "taxonomies[0].desc",
        ]
        self._ctss_location_fields: List[str] = [
            "addr_practice.city",
            "addr_practice.state",
            "addr_practice.zip",
            "addr_practice.country_name",
        ]
        self._location_aliases: Dict[str, str] = {
            "bay area": "San Francisco CA",
            "sf bay area": "San Francisco CA",
            "silicon valley": "San Jose CA",
            "south bay": "San Jose CA",
            "east bay": "Oakland CA",
            "la": "Los Angeles CA",
            "greater los angeles": "Los Angeles CA",
            "dallas fort worth": "Dallas TX",
        }

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

        self._initialize_clinicaltables_field_maps()

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

        if parsed_query.medical_need:
            search_criteria = SearchCriteria(
                specialties=parsed_query.specialties,
                location=parsed_query.location,
                insurance=parsed_query.insurance,
                preferred_languages=parsed_query.preferred_languages,
                keywords=parsed_query.keywords,
            )

            local_results = [
                self._normalize_result_trust_metadata(result)
                for result in self.provider_repository.search(search_criteria)
            ]
            logger.info("Local semantic search results=%s", len(local_results))

            if not local_results:
                fallback_results, missing_location_hint = self._search_clinicaltables(
                    parsed_query,
                    limit=self.ctss_max_results,
                )
                fallback_results = [
                    self._normalize_result_trust_metadata(result)
                    for result in fallback_results
                ]
                logger.info(
                    "ClinicalTables fallback results=%s", len(fallback_results)
                )
                if not fallback_results and not missing_location_hint:
                    fallback_results = [
                        self._normalize_result_trust_metadata(result)
                        for result in self._trusted_resource_fallback(parsed_query)
                    ]
                    logger.info(
                        "Trusted resource fallback results=%s", len(fallback_results)
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

        fallback_only_mode = self._should_use_fallback_only_template()

        if missing_location_hint:
            response_payload.setdefault("notes", missing_location_hint)
        elif no_results_found:
            response_payload.setdefault(
                "notes",
                "No providers were found in the local dataset or via the ClinicalTables API."
            )

        if self.provider_repository.load_error and not fallback_only_mode:
            response_payload["notes"] = (
                "Local dataset fallback was used because loading the configured Hugging Face "
                f"dataset failed: {self.provider_repository.load_error}"
            )

        if parsed_query.medical_need and (local_results or fallback_results) and not missing_location_hint:
            return self._compose_result_card_response(response_payload)

        if missing_location_hint:
            template_key = "response_template_location_needed"
            fallback_only_mode = False
        else:
            template_key = (
                "response_template_fallback_only"
                if fallback_only_mode
                else "response_template"
            )

        return self._compose_response(
            client,
            response_payload,
            max_tokens,
            temperature,
            top_p,
            template_key=template_key,
        )

    # ------------------------------------------------------------------
    def _compose_result_card_response(self, payload: Dict[str, Any]) -> str:
        query = payload.get("query", {})
        response_language = (
            query.get("response_language")
            or query.get("detected_language")
            or "English"
        )
        summary = self._clean_card_value(query.get("summary")) or "your care search"
        results = list(payload.get("local_results") or []) + list(
            payload.get("fallback_results") or []
        )

        lines = [f"Here are care navigation results for {summary}."]

        care_setting_guidance = self._clean_card_value(
            payload.get("care_setting_guidance")
        )
        if care_setting_guidance:
            lines.extend(["", f"**Care route:** {care_setting_guidance}"])

        specialist_guidance = self._clean_card_value(
            payload.get("specialist_plan_guidance")
        )
        if specialist_guidance:
            lines.extend(["", f"**Referral note:** {specialist_guidance}"])

        notes = self._clean_card_value(payload.get("notes"))
        if notes:
            lines.extend(["", f"**Note:** {notes}"])

        for index, result in enumerate(results, start=1):
            if isinstance(result, dict):
                lines.extend(
                    ["", self._format_provider_result_card(result, index, query)]
                )

        verification_guidance = self._clean_card_value(
            payload.get("verification_guidance")
        )
        if verification_guidance:
            lines.extend(
                ["", f"**Before you contact a provider:** {verification_guidance}"]
            )

        return self._append_required_trust_guidance(
            "\n".join(lines).strip(),
            response_language,
        )

    # ------------------------------------------------------------------
    def _format_provider_result_card(
        self,
        result: Dict[str, Any],
        index: int,
        query: Dict[str, Any],
    ) -> str:
        name = self._clean_card_value(result.get("name")) or f"Result {index}"
        specialty = self._result_specialty_label(result)
        address = self._clean_card_value(result.get("address")) or self._clean_card_value(
            result.get("location")
        )
        phone = self._clean_card_value(result.get("phone"))
        website = self._clean_card_value(result.get("website"))
        provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
        source = (
            self._clean_card_value(provenance.get("source"))
            or self._clean_card_value(result.get("source"))
            or "Unknown source"
        )
        insurance = self._ensure_list(result.get("insurance_reported"))
        trust_labels = self._ensure_list(result.get("trust_labels"))

        card_lines = [
            f"### {index}. {name}",
            f"- **Specialty/type:** {specialty or 'Not listed'}",
            f"- **Address:** {address or 'Not listed'}",
            f"- **Phone:** {phone or 'Not listed'}",
        ]
        if website:
            card_lines.append(f"- **Website:** {website}")

        card_lines.extend(
            [
                f"- **Source:** {source}",
                f"- **Why matched:** {self._result_match_reason(result, query)}",
                f"- **Listed insurance:** {', '.join(insurance) if insurance else 'Not listed'} (reported only; network participation is not verified here)",
                f"- **Insurance/network verification:** {self._verification_status_label(result.get('insurance_network_verification'), 'unverified')}",
                f"- **Accepting new patients:** {self._verification_status_label(result.get('accepting_new_patients_status'), 'unknown')}",
                "- **Appointment availability:** Not verified; call the provider to confirm.",
                f"- **Trust labels:** {', '.join(trust_labels) if trust_labels else 'Source and verification details not listed'}",
                "- **Referral/verification reminder:** Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
            ]
        )
        return "\n".join(card_lines)

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
    def _result_match_reason(
        self,
        result: Dict[str, Any],
        query: Dict[str, Any],
    ) -> str:
        query_parts = self._ensure_list(query.get("specialties")) + self._ensure_list(
            query.get("keywords")
        )
        if query_parts:
            return "Matched requested care terms: " + ", ".join(query_parts)

        taxonomy = self._clean_card_value(result.get("taxonomy"))
        if taxonomy:
            return f"Listed provider type: {taxonomy}"

        description = self._clean_card_value(result.get("description"))
        if description:
            return description

        return "Matched available care directory result for the search criteria."

    # ------------------------------------------------------------------
    @staticmethod
    def _verification_status_label(value: Any, default: str) -> str:
        if isinstance(value, dict):
            status = value.get("status") or default
            basis = value.get("basis")
            if basis:
                return f"{status} ({basis})"
            return str(status)
        return default

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_card_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    # ------------------------------------------------------------------
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

        completion = client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.0,
            top_p=0.1,
        )

        content = completion.choices[0].message["content"] if completion.choices else ""
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
            retry_completion = client.chat_completion(
                retry_messages,
                max_tokens=512,
                temperature=0.0,
                top_p=0.1,
            )
            retry_content = (
                retry_completion.choices[0].message["content"]
                if retry_completion.choices
                else ""
            )
            logger.debug("Interpret retry response received. length=%s", len(retry_content))
            parsed_payload = self._safe_json_parse(retry_content)

        if not parsed_payload:
            parsed_payload = {
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

        detected_language = str(parsed_payload.get("detected_language", "unknown"))
        response_language = parsed_payload.get("response_language") or detected_language or "English"

        if (
            response_language
            and detected_language
            and response_language.lower().strip() == "english"
            and detected_language.lower().strip() != "english"
        ):
            response_language = detected_language

        return ParsedCareQuery(
            detected_language=detected_language,
            response_language=str(response_language or "English"),
            summary=str(parsed_payload.get("summary", "")),
            medical_need=bool(parsed_payload.get("medical_need", True)),
            location=parsed_payload.get("location"),
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

    # ------------------------------------------------------------------
    def _merge_parsed_queries(
        self, full: ParsedCareQuery, latest: ParsedCareQuery
    ) -> ParsedCareQuery:
        def _prefer(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
            return primary if primary not in (None, "", "unknown") else fallback

        def _merge_lists(primary: List[str], fallback: List[str]) -> List[str]:
            merged: List[str] = []
            for item in primary + fallback:
                if item not in merged:
                    merged.append(item)
            return merged

        detected_language = _prefer(latest.detected_language, full.detected_language)
        response_language = _prefer(latest.response_language, full.response_language)

        specialties = latest.specialties or full.specialties
        location = latest.location or full.location
        insurance = latest.insurance or full.insurance
        preferred_languages = _merge_lists(latest.preferred_languages, full.preferred_languages)
        keywords = _merge_lists(latest.keywords, full.keywords)
        patient_context = latest.patient_context or full.patient_context
        summary = latest.summary or full.summary
        care_setting = latest.care_setting or full.care_setting
        urgency = latest.urgency or full.urgency
        follow_up_focus = _merge_lists(latest.follow_up_focus, full.follow_up_focus)

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

        needs_clarification = bool(latest.needs_clarification or full.needs_clarification)

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

    # ------------------------------------------------------------------
    def _build_navigation_guidance(
        self, query: ParsedCareQuery, message: str
    ) -> Dict[str, Any]:
        request_text = self._combined_request_text(query, message)
        care_setting = self._classify_care_setting(query, request_text)

        follow_up_questions: List[str] = []

        if query.medical_need and not self._has_specific_location(query, request_text):
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
    @staticmethod
    def _contains_any(text: str, patterns: Tuple[str, ...]) -> bool:
        normalized_text = text.lower()
        return any(
            CareLocatorAgent._contains_phrase(normalized_text, pattern)
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
    @staticmethod
    def _contains_emergency_signal(text: str) -> bool:
        if re.search(r"(?<!\d)911(?!\d)", text):
            return True
        if re.search(r"\b9[\s-]*1[\s-]*1\b", text):
            return True
        return CareLocatorAgent._contains_any(text, _EMERGENCY_PATTERNS)

    # ------------------------------------------------------------------
    def _has_plan_type(self, query: ParsedCareQuery, text: str) -> bool:
        if self._contains_any(text, _PLAN_TYPE_PATTERNS):
            return True
        return any(self._contains_any(item.lower(), _PLAN_TYPE_PATTERNS) for item in query.insurance)

    # ------------------------------------------------------------------
    def _has_specific_location(self, query: ParsedCareQuery, text: str) -> bool:
        if query.location and self._location_has_city(query.location):
            return True
        return bool(self._match_city_state(text)) or bool(self._extract_zip_code(text))

    # ------------------------------------------------------------------
    def _has_clear_care_need(self, query: ParsedCareQuery, text: str) -> bool:
        if query.specialties:
            return True
        if self._contains_any(
            text, _SPECIALIST_PATTERNS + _ROUTINE_PATTERNS + _URGENT_PATTERNS
        ):
            return True
        return False

    # ------------------------------------------------------------------
    def _classify_care_setting(self, query: ParsedCareQuery, text: str) -> str:
        if self._contains_emergency_signal(text):
            return "emergency"
        if self._contains_any(text, _URGENT_PATTERNS):
            return "urgent_care"
        if self._contains_any(text, _ROUTINE_PATTERNS):
            return "pcp"
        if query.specialties or self._contains_any(text, _SPECIALIST_PATTERNS):
            return "specialist"
        return "unclear"

    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    # ------------------------------------------------------------------
    def _search_clinicaltables(
        self, query: ParsedCareQuery, limit: int = 3
    ) -> tuple[List[dict], Optional[str]]:
        (
            search_terms,
            query_filter,
            location_was_specific,
            city_hint,
            zip_hint,
            state_hint,
        ) = self._compose_clinicaltables_query_parts(query)
        logger.debug(
            "ClinicalTables combined query prepared. search_terms_length=%s has_filter=%s",
            len(search_terms),
            bool(query_filter),
        )

        if not search_terms.strip():
            return [], None

        if not location_was_specific:
            return [], self._location_hint()

        per_dataset_limit = max(limit, self.ctss_max_results)

        search_variants: List[tuple[str, Optional[str], str]] = [
            (search_terms, query_filter, "primary"),
        ]

        location_only_terms = " ".join(
            part for part in [zip_hint, city_hint, state_hint] if part
        )
        if (
            location_only_terms
            and location_only_terms.strip()
            and location_only_terms.strip() != search_terms.strip()
        ):
            search_variants.append(
                (location_only_terms.strip(), query_filter, "location-only")
            )

        for variant_terms, variant_filter, variant_label in search_variants:
            variant_results: List[dict] = []
            logger.debug(
                "ClinicalTables search variant=%s terms_length=%s has_filter=%s",
                variant_label,
                len(variant_terms),
                bool(variant_filter),
            )
            for dataset in self._ctss_dataset_priority:
                dataset_results = self._clinicaltables_search_dataset(
                    dataset,
                    variant_terms,
                    variant_filter,
                    per_dataset_limit,
                    city_hint=city_hint,
                    zip_hint=zip_hint,
                    state_hint=state_hint,
                )
                for record in dataset_results:
                    variant_results.append(record)
                    if len(variant_results) >= limit:
                        return variant_results[:limit], None

            if variant_results:
                return variant_results[:limit], None

        return [], None

    # ------------------------------------------------------------------
    def _clinicaltables_search_dataset(
        self,
        dataset: str,
        search_terms: str,
        query_filter: Optional[str],
        limit: int,
        *,
        city_hint: Optional[str] = None,
        zip_hint: Optional[str] = None,
        state_hint: Optional[str] = None,
    ) -> List[dict]:
        config = self.ctss_dataset_configs.get(dataset, {})
        url = config.get("search_url")
        if not url:
            return []

        params = {"terms": search_terms, "maxList": str(limit)}
        if query_filter:
            params["q"] = query_filter
        field_names = self._ctss_result_field_order.get(dataset)
        if field_names:
            params["df"] = ",".join(field_names)

        try:
            response = requests.get(url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables %s search failed. terms_length=%s error=%s",
                dataset,
                len(search_terms),
                exc,
            )
            return []

        try:
            payload = response.json()
        except ValueError:
            payload = None

        logger.debug(
            "ClinicalTables %s raw response status=%s text=%s",
            dataset,
            response.status_code,
            response.text[:500],
        )
        fields, entries = self._parse_clinicaltables_payload(dataset, payload)
        logger.debug(
            "ClinicalTables %s parsed fields_count=%s entries_count=%s sample=%s",
            dataset,
            len(fields) if fields else 0,
            len(entries) if entries else 0,
            entries[:1] if entries else entries,
        )
        if not entries:
            return []

        results: List[dict] = []

        for row in entries[:limit]:
            if not isinstance(row, (list, tuple)):
                continue
            data = {
                str(fields[idx]): row[idx]
                for idx in range(min(len(fields), len(row)))
            }

            if not data:
                fallback_record = self._clinicaltables_build_fallback_record(dataset, row, config)
                if fallback_record:
                    logger.debug(
                        "ClinicalTables %s fallback record constructed: %s",
                        dataset,
                        fallback_record,
                    )
                    enhanced = self._enhance_with_npi_registry(
                        dataset, fallback_record
                    )
                    results.append(enhanced)
                continue

            name = self._first_match(
                data,
                [
                    "name.full",
                    "name.last",
                    "name.first",
                ],
                default="Healthcare Provider",
            )

            if dataset == "npi_idv":
                first = data.get("name.first")
                last = data.get("name.last")
                middle = data.get("name.middle")
                combined = " ".join(
                    chunk
                    for chunk in [first, middle, last]
                    if isinstance(chunk, str) and chunk.strip()
                )
                if combined.strip():
                    name = combined.strip()

            full_address = self._first_match(data, ["addr_practice.full"])
            city = self._first_match(
                data,
                ["addr_practice.city"],
            )

            state = self._first_match(
                data,
                ["addr_practice.state"],
            )

            postal_code = self._first_match(
                data,
                ["addr_practice.zip"],
            )

            street = self._first_match(
                data,
                ["addr_practice.address_1"],
            )

            country = self._first_match(
                data,
                ["addr_practice.country_name"],
            )

            phone = self._first_match(
                data,
                ["addr_practice.phone"],
            )

            taxonomy = self._first_match(
                data,
                [
                    "provider_type",
                    "taxonomies[0].desc",
                    "taxonomies[0].code",
                ],
            )

            languages = self._ensure_list(data.get("languages"))

            npi = data.get("NPI")

            if state_hint:
                state_str = str(state).strip().upper() if state else ""
                if state_str and state_str != state_hint.upper():
                    continue

            if zip_hint:
                postal_str = str(postal_code).strip() if postal_code else ""
                if not postal_str or not postal_str.startswith(zip_hint):
                    continue

            if city_hint:
                city_str = str(city).strip().lower() if city else ""
                if city_str != city_hint.strip().lower():
                    continue

            location_parts: List[str] = []
            if full_address and isinstance(full_address, str) and full_address.strip():
                location_parts.append(full_address.strip())
            else:
                if street:
                    location_parts.append(str(street))

                city_state = ", ".join(
                    chunk for chunk in [city, state] if isinstance(chunk, str) and chunk.strip()
                )
                if city_state:
                    if postal_code and isinstance(postal_code, str) and postal_code.strip():
                        city_state = f"{city_state} {postal_code.strip()}"
                    location_parts.append(city_state)
                elif postal_code:
                    location_parts.append(str(postal_code))

                if country and isinstance(country, str) and country.strip() and country.strip().upper() not in {"US", "USA"}:
                    location_parts.append(country.strip())

            location_text = ", ".join(location_parts)

            record = {
                "name": name,
                "location": location_text
                or ", ".join(
                    chunk
                    for chunk in [city, state]
                    if isinstance(chunk, str) and chunk.strip()
                ),
                "phone": phone,
                "npi": npi,
                "languages": languages,
                "taxonomy": taxonomy,
                "source": config.get(
                    "source_label", "NPI Registry via clinicaltables.nlm.nih.gov"
                ),
                "dataset": dataset,
                "raw": data,
            }
            results.append(self._enhance_with_npi_registry(dataset, record))

        return results

    # ------------------------------------------------------------------
    def _clinicaltables_build_fallback_record(
        self, dataset: str, row: Tuple[Any, ...], config: Dict[str, Any]
    ) -> Optional[dict]:
        if not isinstance(row, (list, tuple)) or not row:
            return None

        def _get(index: int) -> Optional[Any]:
            return row[index] if len(row) > index else None

        raw_name = _get(0)
        name = str(raw_name).strip() if raw_name else "Healthcare Provider"
        if dataset == "npi_idv" and "," in name:
            parts = [part.strip() for part in name.split(",", 1)]
            if len(parts) == 2:
                formatted = f"{parts[1]} {parts[0]}".strip()
                if formatted:
                    name = formatted

        npi = _get(1)
        taxonomy = _get(2)
        address = _get(3)

        phone = None
        if len(row) > 4:
            phone_candidate = _get(4)
            if isinstance(phone_candidate, str) and phone_candidate.strip():
                phone = phone_candidate.strip()

        location = str(address).strip() if address else ""

        return {
            "name": name,
            "location": location,
            "phone": phone,
            "npi": npi,
            "languages": [],
            "taxonomy": taxonomy,
            "source": config.get(
                "source_label", "NPI Registry via clinicaltables.nlm.nih.gov"
            ),
            "dataset": dataset,
            "raw": {"fallback_row": list(row)},
        }

    # ------------------------------------------------------------------
    def _enhance_with_npi_registry(self, dataset: str, record: dict) -> dict:
        if not self._npi_registry_enabled:
            return record
        if not isinstance(record, dict):
            return record

        npi = record.get("npi")
        if not npi:
            return record

        npi_str = str(npi).strip()
        if not npi_str.isdigit():
            return record

        lookup = self._lookup_npi_registry_entry(npi_str)
        if not lookup:
            return record

        practice_address = lookup.get("practice_address")
        mailing_address = lookup.get("mailing_address")
        location_override = self._format_npi_registry_location(practice_address)
        if location_override:
            record["location"] = location_override

        phone_value = None
        if isinstance(practice_address, dict):
            phone_value = practice_address.get("telephone_number")
        if not phone_value and isinstance(mailing_address, dict):
            phone_value = mailing_address.get("telephone_number")
        if isinstance(phone_value, str) and phone_value.strip():
            record["phone"] = phone_value.strip()

        if not record.get("taxonomy") and lookup.get("taxonomies"):
            taxonomy = self._first_non_empty_taxonomy_description(
                lookup.get("taxonomies")
            )
            if taxonomy:
                record["taxonomy"] = taxonomy

        record.setdefault("raw", {})["npi_registry"] = {
            "dataset": dataset,
            "lookup": lookup,
        }
        # check CMS opt‑out status and attach the result
        try:
            opt_out_info = self._check_medicare_opt_out(str(record["npi"]))
            if opt_out_info:
                record["medicare_opt_out"] = opt_out_info
        except Exception:
            # do not break provider enrichment if opt‑out lookup fails
            logger.warning("Error checking opt‑out status for %s", record.get("npi"))

        return record

    # ------------------------------------------------------------------
    def _lookup_npi_registry_entry(self, npi: str) -> Optional[dict]:
        cached = self._npi_registry_cache.get(npi)
        if cached is not None:
            return cached

        if not self._npi_registry_enabled or not self.npi_registry_url:
            self._npi_registry_cache[npi] = None
            return None

        params = {
            "number": npi,
            "version": self.npi_registry_version,
            "limit": 1,
        }

        try:
            response = requests.get(
                self.npi_registry_url,
                params=params,
                timeout=self.npi_registry_timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("NPI registry lookup failed for %s: %s", npi, exc)
            self._npi_registry_cache[npi] = None
            return None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        entry: Optional[dict] = None
        if isinstance(payload, dict):
            results = payload.get("results") or []
            if results:
                candidate = results[0]
                if isinstance(candidate, dict):
                    entry = candidate

        if not entry:
            self._npi_registry_cache[npi] = None
            return None

        addresses = entry.get("addresses")
        practice = self._select_npi_registry_address(addresses, target="LOCATION")
        mailing = self._select_npi_registry_address(addresses, target="MAILING")

        normalized = {
            "npi": npi,
            "practice_address": practice,
            "mailing_address": mailing,
            "taxonomies": entry.get("taxonomies"),
            "basic": entry.get("basic"),
            "enumeration_type": entry.get("enumeration_type"),
            "created_epoch": entry.get("created_epoch"),
            "last_updated_epoch": entry.get("last_updated_epoch"),
            "raw_entry": entry,
        }

        self._npi_registry_cache[npi] = normalized
        return normalized

        # ------------------------------------------------------------------
    def _check_medicare_opt_out(self, npi: str) -> Optional[dict]:
        """
        Look up a provider's Medicare opt‑out status using CMS's Opt‑Out Affidavits API.
        Returns a dict with opted_out (bool), optout_effective_date, optout_end_date,
        or None if no record is found or the request fails.
        """
        # Dataset ID for the Opt‑Out Affidavits (as of 2025-09-28).
        # Fields include NPI, Optout Effective Date, and Optout End Date:contentReference[oaicite:0]{index=0}.
        dataset_id = "9887a515-7552-4693-bf58-735c77af46d7"
        base_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"

        try:
            # Build query: filter by exact NPI
            resp = requests.get(
                base_url,
                params={
                    "filter[NPI]": str(npi),
                    # limit results to one record; not strictly necessary but avoids large payloads
                    "size": "1",
                },
                timeout=6,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Opt‑out lookup failed for %s: %s", npi, exc)
            return None

        # API returns an array of records; if empty, provider has no opt‑out on file
        if not isinstance(data, list) or not data:
            return {"opted_out": False}

        record = data[0]
        effective = record.get("Optout Effective Date")
        end = record.get("Optout End Date")

        # Determine if the opt‑out is currently in effect.  An open‑ended (null) end date
        # or end date in the future means the provider is opted out.
        opted_out = False
        if effective:
            # End date might be empty or a future date (YYYY/MM/DD format)
            if not end:
                opted_out = True
            else:
                try:
                    from datetime import date
                    end_date = date.fromisoformat(end.replace("/", "-"))
                    opted_out = end_date >= date.today()
                except Exception:
                    opted_out = True

        return {
            "opted_out": opted_out,
            "optout_effective_date": effective,
            "optout_end_date": end,
        }
    
    # ------------------------------------------------------------------
    @staticmethod
    def _select_npi_registry_address(addresses: Any, *, target: str) -> Optional[dict]:
        if not isinstance(addresses, list):
            return None

        fallback: Optional[dict] = None
        for entry in addresses:
            if not isinstance(entry, dict):
                continue
            if fallback is None:
                fallback = entry
            purpose = entry.get("address_purpose")
            if isinstance(purpose, str) and purpose.strip().upper() == target.upper():
                return entry

        return fallback

    # ------------------------------------------------------------------
    @staticmethod
    def _format_npi_registry_location(address: Optional[dict]) -> str:
        if not isinstance(address, dict):
            return ""

        parts: List[str] = []
        for key in ("address_1", "address_2"):
            value = address.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())

        city = address.get("city")
        state = address.get("state")
        locality_parts: List[str] = []
        if isinstance(city, str) and city.strip():
            locality_parts.append(city.strip())
        if isinstance(state, str) and state.strip():
            locality_parts.append(state.strip())

        locality = ", ".join(locality_parts)
        postal = address.get("postal_code")
        if isinstance(postal, str) and postal.strip():
            postal_clean = postal.strip()
            if locality:
                locality = f"{locality} {postal_clean}"
            else:
                locality = postal_clean

        if locality:
            parts.append(locality)

        country = address.get("country_name") or address.get("country_code")
        if isinstance(country, str):
            country_clean = country.strip()
            if country_clean and country_clean.upper() not in {"US", "USA", "UNITED STATES"}:
                parts.append(country_clean)

        return ", ".join(parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _first_non_empty_taxonomy_description(taxonomies: Any) -> Optional[str]:
        if not isinstance(taxonomies, list):
            return None

        for entry in taxonomies:
            if not isinstance(entry, dict):
                continue
            for key in ("desc", "code"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    # ------------------------------------------------------------------
    def _compose_clinicaltables_query_parts(
        self, query: ParsedCareQuery
    ) -> Tuple[str, Optional[str], bool]:
        parts: List[str] = []
        filters: List[str] = []
        location_was_specific = False

        # Specialty: pick the first reasonable suggestion
        for specialty in query.specialties:
            best = self._best_suggestion_across_datasets(
                specialty, self._ctss_taxonomy_fields
            )
            if best:
                parts.append(best)
                break

        # Location: reduce to one normalized chunk (city/state/zip)
        alias_used = False

        location_inputs: List[str] = []
        city_hint: Optional[str] = None
        zip_hint: Optional[str] = None
        state_hint: Optional[str] = None

        if query.location:
            location_inputs.append(query.location)

        inferred_location = self._infer_location_hint(query)
        if inferred_location and inferred_location not in location_inputs:
            location_inputs.append(inferred_location)

        for location_input in location_inputs:
            normalized_location, alias_used = self._normalize_location_alias(
                location_input
            )
            for chunk in self._split_location(normalized_location):
                user_chunk = chunk.strip()
                if not user_chunk:
                    continue

                match = self._match_city_state(user_chunk)
                chunk_city: Optional[str] = None
                chunk_state: Optional[str] = None
                if match:
                    chunk_city, chunk_state = match
                else:
                    chunk_state = self._extract_state_code(user_chunk)

                chunk_zip = self._extract_zip_code(user_chunk)

                if chunk_city and not city_hint:
                    city_hint = chunk_city
                if chunk_state and not state_hint:
                    state_hint = chunk_state
                if chunk_zip and not zip_hint:
                    zip_hint = chunk_zip

                if user_chunk not in parts:
                    parts.append(user_chunk)

                suggestion = self._best_suggestion_across_datasets(
                    user_chunk, self._ctss_location_fields
                )

                suggestion_for_filters: Optional[str] = None
                suggestion_for_terms: Optional[str] = None

                if suggestion and suggestion.strip() and suggestion != user_chunk:
                    user_state = self._extract_state_code(user_chunk)
                    suggestion_state = self._extract_state_code(suggestion)
                    if user_state and suggestion_state and user_state != suggestion_state:
                        suggestion_for_filters = None
                        suggestion_for_terms = None
                    else:
                        suggestion_for_filters = suggestion
                        suggestion_for_terms = suggestion

                if suggestion_for_terms and suggestion_for_terms not in parts:
                    parts.append(suggestion_for_terms)

                chunk_filters = self._build_query_filters(
                    user_chunk,
                    suggestion_for_filters,
                    city_hint=chunk_city,
                    zip_hint=chunk_zip,
                    state_hint=chunk_state,
                )
                for filter_value in chunk_filters:
                    if filter_value not in filters:
                        filters.append(filter_value)
                if self._location_filters_are_specific(
                    chunk_filters, user_chunk, alias_used
                ):
                    location_was_specific = True

                should_stop = bool(chunk_filters) or bool(chunk_zip) or bool(chunk_city)
                if location_was_specific or should_stop:
                    break

            if location_was_specific or city_hint or zip_hint:
                break

        if not zip_hint and query.location:
            explicit_zip = self._extract_zip_code(query.location)
            if explicit_zip:
                zip_hint = explicit_zip
                filter_value = f"addr_practice.zip:{explicit_zip}"
                if filter_value not in filters:
                    filters.append(filter_value)
                location_was_specific = True

        if not state_hint:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.state:"):
                    state_hint = filter_value.split(":", 1)[1]
                    break

        if not zip_hint:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.zip:"):
                    zip_hint = filter_value.split(":", 1)[1].strip()
                    break

        if not city_hint and filters:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.city:"):
                    value = filter_value.split(":", 1)[1].strip()
                    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                        value = value[1:-1]
                    city_hint = value
                    break

        query_filter = " AND ".join(filters) if filters else None

        return (
            " ".join(part for part in parts if part),
            query_filter,
            location_was_specific,
            city_hint,
            zip_hint,
            state_hint,
        )

    # ------------------------------------------------------------------
    def _build_query_filters(
        self,
        user_location: str,
        suggestion: Optional[str] = None,
        *,
        city_hint: Optional[str] = None,
        zip_hint: Optional[str] = None,
        state_hint: Optional[str] = None,
    ) -> List[str]:
        filters: List[str] = []

        user_state = self._extract_state_code(user_location)
        suggestion_state = (
            self._extract_state_code(suggestion) if suggestion else None
        )

        state_code = state_hint or user_state or suggestion_state
        if user_state and suggestion_state and user_state != suggestion_state:
            state_code = user_state

        if state_code:
            filters.append(f"addr_practice.state:{state_code}")

        if zip_hint:
            filters.append(f"addr_practice.zip:{zip_hint}")
        elif city_hint and state_code:
            escaped_city = city_hint.replace('"', '\"')
            filters.append(f'addr_practice.city:"{escaped_city}"')

        return list(dict.fromkeys(filters))

    # ------------------------------------------------------------------
    def _location_filters_are_specific(
        self, filters: List[str], user_location: str, alias_used: bool
    ) -> bool:
        if self._extract_zip_code(user_location):
            return True

        if alias_used:
            return False

        has_state = any(filter_str.startswith("addr_practice.state") for filter_str in filters)
        if not has_state:
            return False

        return self._location_has_city(user_location)

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
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", value)
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
    def _normalize_location_alias(self, location: str) -> Tuple[str, bool]:
        key = location.strip().lower()
        if key in self._location_aliases:
            return self._location_aliases[key], True
        return location, False

    # ------------------------------------------------------------------
    def _location_hint(self) -> str:
        return (
            "Please share a specific city, state, or ZIP code so I can refine the search."
        )

    # ------------------------------------------------------------------
    def _infer_location_hint(self, query: ParsedCareQuery) -> Optional[str]:
        texts: List[str] = []
        if query.summary:
            texts.append(query.summary)
        texts.extend(query.keywords)

        for text in texts:
            if not text or not isinstance(text, str):
                continue
            match = self._match_city_state(text)
            if match:
                city, state_code = match
                return f"{city} {state_code}"
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
    def _parse_clinicaltables_payload(
        self,
        dataset: str,
        payload: Any,
    ) -> Tuple[List[str], List[List[Any]]]:
        fields: List[str] = []
        entries: List[List[Any]] = []

        if not isinstance(payload, list):
            return fields, entries

        if len(payload) >= 4 and isinstance(payload[2], list) and isinstance(payload[3], list):
            potential_fields = payload[2]
            potential_entries = payload[3]
        elif len(payload) >= 3 and isinstance(payload[1], list) and isinstance(payload[2], list):
            potential_fields = payload[1]
            potential_entries = payload[2]
        else:
            potential_fields = (
                payload[2] if len(payload) > 2 and isinstance(payload[2], list) else []
            )
            potential_entries = (
                payload[3] if len(payload) > 3 and isinstance(payload[3], list) else []
            )

        if all(isinstance(item, str) for item in potential_fields):
            fields = [str(item) for item in potential_fields]
        elif all(isinstance(item, int) for item in potential_fields):
            reverse_map = {
                index: name
                for name, index in self._ctss_field_map.get(dataset, {}).items()
            }
            fields = [reverse_map.get(int(item), str(item)) for item in potential_fields]
        else:
            configured_fields = self._ctss_result_field_order.get(dataset, [])
            if not configured_fields:
                configured_fields = (
                    self.ctss_dataset_configs.get(dataset, {}).get("result_fields")
                    or []
                )
            if configured_fields:
                fields = list(configured_fields)
            else:
                fields = []

        filtered_entries: List[List[Any]] = []
        for row in potential_entries:
            if isinstance(row, (list, tuple)):
                filtered_entries.append(list(row))
        entries = filtered_entries

        return fields, entries

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_clinicaltables_fields_payload(payload: Any) -> Dict[str, int]:
        mapping: Dict[str, int] = {}

        if not isinstance(payload, list):
            return mapping

        if payload and isinstance(payload[0], list) and len(payload[0]) >= 2:
            entries = payload
        else:
            entries = payload[1:] if len(payload) > 1 else []

        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 2:
                continue

            field_index: Optional[int] = None
            index_candidate = entry[0]
            if isinstance(index_candidate, (int, str)):
                try:
                    field_index = int(index_candidate)
                except (TypeError, ValueError):
                    field_index = None

            field_name_candidate = entry[1]
            if not isinstance(field_name_candidate, str) or not field_name_candidate:
                continue

            if field_index is None:
                field_index = len(mapping)

            mapping[field_name_candidate] = field_index

        return mapping

    # ------------------------------------------------------------------
    def _best_suggestion_across_datasets(
        self, term: Optional[str], candidate_fields: List[str]
    ) -> str:
        if not term:
            return ""

        cleaned = term.strip()
        if not cleaned:
            return ""

        for dataset in self._ctss_dataset_priority:
            suggestions = self._clinicaltables_suggest_for_dataset(
                dataset, cleaned, candidate_fields
            )
            if suggestions:
                return suggestions[0]

        return cleaned

    # ------------------------------------------------------------------
    def _clinicaltables_suggest_for_dataset(
        self, dataset: str, term: str, candidate_fields: List[str]
    ) -> List[str]:
        config = self.ctss_dataset_configs.get(dataset, {})
        values_url = config.get("values_url")
        if not values_url:
            return []

        field_map = self._ctss_field_map.get(dataset)
        if not field_map:
            return []

        aggregated: List[str] = []
        for field in candidate_fields:
            field_index = field_map.get(field)
            if field_index is None:
                continue

            cache_key = (dataset, term, field)
            cached = self._ctss_suggest_cache.get(cache_key)
            if cached is not None:
                aggregated.extend(cached)
                continue

            suggestions = self._clinicaltables_request_values(
                dataset, term, field_index, values_url
            )
            self._ctss_suggest_cache[cache_key] = suggestions
            aggregated.extend(suggestions)

        return list(dict.fromkeys(aggregated))

    # ------------------------------------------------------------------
    def _clinicaltables_request_values(
        self, dataset: str, term: str, field_index: int, values_url: str
    ) -> List[str]:
        params = {
            "terms": term,
            "df": str(field_index),
            "maxList": str(self.ctss_values_max_results),
        }

        logger.debug(
            "ClinicalTables values request dataset=%s term_length=%s field_index=%s",
            dataset,
            len(term),
            field_index,
        )

        try:
            response = requests.get(values_url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables values failed dataset=%s term_length=%s field_index=%s error=%s",
                dataset,
                len(term),
                field_index,
                exc,
            )
            return []

        try:
            payload = response.json()
        except ValueError:
            payload = None

        suggestions: List[str] = []

        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    suggestions.extend(self._flatten_suggestions(value))
                elif isinstance(value, str):
                    suggestions.append(value)
        elif isinstance(payload, list):
            if payload and isinstance(payload[0], int):
                for item in payload[1:]:
                    if isinstance(item, list):
                        suggestions.extend(self._flatten_suggestions(item))
                    elif isinstance(item, str):
                        suggestions.append(item)
            else:
                suggestions.extend(self._flatten_suggestions(payload))

        return list(dict.fromkeys(suggestions))

    # ------------------------------------------------------------------
    def _initialize_clinicaltables_field_maps(self) -> None:
        for dataset, config in self.ctss_dataset_configs.items():
            requested_fields = config.get("result_fields") or []
            alias_map: Dict[str, List[str]] = config.get("field_aliases", {}) or {}

            if requested_fields:
                field_map = {
                    field: index for index, field in enumerate(requested_fields)
                }
            else:
                field_map = self._fetch_field_map_for_dataset(dataset, config)

            self._ctss_field_map[dataset] = field_map

            resolved_fields: List[str] = list(requested_fields) or list(field_map.keys())
            if not resolved_fields:
                continue

            if alias_map and field_map:
                filtered_fields: List[str] = []
                for canonical_field in resolved_fields:
                    candidates = [canonical_field] + alias_map.get(canonical_field, [])
                    if any(candidate in field_map for candidate in candidates):
                        filtered_fields.append(canonical_field)
                    else:
                        logger.warning(
                            "ClinicalTables %s missing requested field '%s'",
                            dataset,
                            canonical_field,
                        )
                if filtered_fields:
                    resolved_fields = filtered_fields

            self._ctss_result_field_order[dataset] = list(dict.fromkeys(resolved_fields))

    # ------------------------------------------------------------------
    def _fetch_field_map_for_dataset(
        self, dataset: str, config: Dict[str, Any]
    ) -> Dict[str, int]:
        fields_url = config.get("fields_url")
        if fields_url:
            try:
                response = requests.get(fields_url, timeout=self.ctss_timeout)
                response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "ClinicalTables %s fields fetch failed: %s",
                    dataset,
                    exc,
                )
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                else:
                    mapping = self._parse_clinicaltables_fields_payload(payload)
                    if mapping:
                        logger.debug(
                            "ClinicalTables %s field map loaded (sample=%s)",
                            dataset,
                            list(mapping.keys())[:6],
                        )
                        return mapping

        # Fallback: probe the search endpoint for at least one result and infer fields
        url = config.get("search_url")
        if not url:
            return {}

        probe_term = config.get("probe_term") or "urology"
        params = {"terms": probe_term, "maxList": "1"}

        try:
            response = requests.get(url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables %s field map probe failed: %s",
                dataset,
                exc,
            )
            return {}

        try:
            payload = response.json()
        except ValueError:
            payload = None

        fields, _ = self._parse_clinicaltables_payload(dataset, payload)
        if not fields:
            return {}

        mapping = {str(field): idx for idx, field in enumerate(fields)}
        logger.debug(
            "ClinicalTables %s fallback field map loaded (sample=%s)",
            dataset,
            list(mapping.keys())[:6],
        )
        return mapping

    # ------------------------------------------------------------------
    def _flatten_suggestions(self, values: List[Any]) -> List[str]:
        flattened: List[str] = []
        for value in values:
            if isinstance(value, str):
                flattened.append(value)
            elif isinstance(value, list):
                flattened.extend(
                    [item for item in value if isinstance(item, str) and item]
                )
        return flattened

    # ------------------------------------------------------------------
    @staticmethod
    def _split_location(location: str) -> List[str]:
        parts = [part.strip() for part in re.split(r"[|/,]", location) if part.strip()]
        if parts:
            return parts
        stripped = location.strip()
        return [stripped] if stripped else [location]

    # ------------------------------------------------------------------
    def _trusted_resource_fallback(self, query: ParsedCareQuery) -> List[dict]:
        region_hint = query.location or "international"
        if not self.fallback_resources:
            return []

        query_terms = {
            term.lower()
            for term in (query.specialties + query.keywords)
            if isinstance(term, str)
        }
        location_text = (query.location or "").lower()

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
            region_filters = [
                str(item).lower() for item in resource.get("regions", []) if item
            ]

            if specialty_filters:
                if not query_terms:
                    continue
                if not any(filter_term in query_terms for filter_term in specialty_filters):
                    continue

            if region_filters:
                if location_text:
                    if not any(filter_term in location_text for filter_term in region_filters):
                        continue
                else:
                    if not any(filter_term in {"international", "global"} for filter_term in region_filters):
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
            raise RuntimeError("Model did not return any choices")

        content = self._content_from_completion_choice(first_choice)

        if not content:
            raise RuntimeError("Model response missing content")

        if not isinstance(content, str):
            content = str(content)

        return self._append_required_trust_guidance(content.strip(), response_language)

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

    # ------------------------------------------------------------------
    def _should_use_fallback_only_template(self) -> bool:
        repository = self.provider_repository

        if repository is None:
            return True

        has_dataset_id = bool(getattr(repository, "dataset_id", None))
        has_local_records = bool(getattr(repository, "providers", []))

        return (not has_dataset_id) and (not has_local_records)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_parse(content: str) -> Optional[dict]:
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
            repaired = CareLocatorAgent._repair_json(trimmed)
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
        if isinstance(value, list):
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


__all__ = ["CareLocatorAgent", "ParsedCareQuery"]
