from __future__ import annotations

import json
import re
import unicodedata
from html import escape
from dataclasses import asdict, dataclass, field
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

_DETERMINISTIC_RENDER_COPY = {
    "english": {
        "results_intro": "Here are care navigation results for {summary}.",
        "result_title_fallback": "Result {index}",
        "care_route_label": "Care route",
        "referral_note_label": "Referral note",
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
        "result_title_fallback": "Resultado {index}",
        "care_route_label": "Ruta de atención",
        "referral_note_label": "Nota sobre remisión",
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
        "result_title_fallback": "结果{index}",
        "care_route_label": "就医路线",
        "referral_note_label": "转诊提示",
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


def _resolved_supported_language_key(response_language: Optional[str]) -> str:
    if _is_unknown_response_language(response_language):
        return "english"

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

    if language_key not in _DETERMINISTIC_RENDER_COPY:
        return "english"
    return language_key or "english"


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

        if parsed_query.medical_need:
            provider_request = ProviderSearchRequest(
                specialties=tuple(parsed_query.specialties),
                location=parsed_query.location,
                insurance=tuple(parsed_query.insurance),
                preferred_languages=tuple(parsed_query.preferred_languages),
                keywords=tuple(parsed_query.keywords),
            )
            search_response = self.provider_search_service.search(
                provider_request,
                limit=self.ctss_max_results,
            )
            missing_location_hint = search_response.missing_location_hint

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
        elif no_results_found:
            response_payload.setdefault(
                "notes",
                "No providers were found via the configured provider search sources."
            )

        if parsed_query.medical_need and (local_results or fallback_results) and not missing_location_hint:
            return self._compose_result_card_response(response_payload)

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
    def _compose_result_card_response(self, payload: Dict[str, Any]) -> str:
        query = payload.get("query", {})
        response_language = (
            query.get("response_language")
            or query.get("detected_language")
            or "English"
        )
        language_key = _resolved_supported_language_key(response_language)
        summary = self._clean_card_value(query.get("summary")) or "your care search"
        results = list(payload.get("local_results") or []) + list(
            payload.get("fallback_results") or []
        )

        lines = [self._render_copy(language_key, "results_intro", summary=summary)]

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

        for index, result in enumerate(results, start=1):
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

        verification_guidance = self._translate_deterministic_text(
            payload.get("verification_guidance") or "",
            language_key,
        )
        if verification_guidance:
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
    @staticmethod
    def _clean_subtitle_fragment(value: Any, kind: str) -> str:
        cleaned_value = CareLocatorAgent._clean_card_value(value)
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
    @staticmethod
    def _verification_status_label(
        value: Any,
        default: str,
        language_key: str = "english",
    ) -> str:
        if isinstance(value, dict):
            status = str(value.get("status") or default)
            basis = value.get("basis")
            localized_status = CareLocatorAgent._translate_status_value(
                status,
                language_key,
            )
            if basis:
                return f"{localized_status} ({CareLocatorAgent._translate_deterministic_text(str(basis), language_key)})"
            return localized_status
        return CareLocatorAgent._translate_status_value(str(default), language_key)

    # ------------------------------------------------------------------
    @staticmethod
    def _translate_status_value(status: str, language_key: str) -> str:
        normalized_status = str(status).strip().lower().replace("_", " ")
        if normalized_status == "unverified":
            return CareLocatorAgent._render_copy(language_key, "status_unverified")
        if normalized_status == "unknown":
            return CareLocatorAgent._render_copy(language_key, "status_unknown")
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
    @staticmethod
    def _translate_trust_label(label: str, language_key: str) -> str:
        cleaned_label = str(label).strip()
        if not cleaned_label:
            return ""
        if cleaned_label.startswith("Source: "):
            source_value = cleaned_label[len("Source: ") :]
            return CareLocatorAgent._render_copy(
                language_key,
                "trust_label_source",
                value=source_value,
            )
        if cleaned_label.startswith("Insurance/network: "):
            insurance_value = cleaned_label[len("Insurance/network: ") :]
            return CareLocatorAgent._render_copy(
                language_key,
                "trust_label_insurance",
                value=CareLocatorAgent._translate_status_value(
                    insurance_value,
                    language_key,
                ),
            )
        if cleaned_label.startswith("New patients: "):
            new_patient_value = cleaned_label[len("New patients: ") :]
            return CareLocatorAgent._render_copy(
                language_key,
                "trust_label_new_patients",
                value=CareLocatorAgent._translate_status_value(
                    new_patient_value,
                    language_key,
                ),
            )
        if cleaned_label.startswith("Medicare opt-out: "):
            medicare_value = cleaned_label[len("Medicare opt-out: ") :]
            localized_value = {
                "opted out": CareLocatorAgent._render_copy(language_key, "medicare_opted_out"),
                "no opt-out record found": CareLocatorAgent._render_copy(language_key, "medicare_no_record"),
                "unknown": CareLocatorAgent._render_copy(language_key, "medicare_unknown"),
            }.get(medicare_value, medicare_value)
            return CareLocatorAgent._render_copy(
                language_key,
                "trust_label_medicare_opt_out",
                value=localized_value,
            )
        return CareLocatorAgent._translate_deterministic_text(cleaned_label, language_key)

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
        for dataset, config in self.ctss_dataset_configs.items():
            search_url = config.get("search_url")
            if not search_url:
                continue
            configs[dataset] = ClinicalTablesDatasetConfig(
                search_url=search_url,
                values_url=config.get("values_url"),
                fields_url=config.get("fields_url"),
                probe_term=config.get("probe_term"),
                source_label=str(
                    config.get("source_label") or "NPI Registry via clinicaltables.nlm.nih.gov"
                ),
                result_fields=list(config.get("result_fields") or []),
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


__all__ = ["CareLocatorAgent", "ParsedCareQuery"]
