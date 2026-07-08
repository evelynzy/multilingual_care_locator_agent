"""Required trust/safety guidance (7 languages) and emergency signal detection."""
from __future__ import annotations

import re
from typing import Optional

from care.language import (
    _is_unknown_response_language,
    _lookup_language_alias,
    _normalize_response_language,
)

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
