"""Language detection aliases, response-language resolution, and reply localization."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


# Normalize alias keys the SAME way inputs are normalized, so native-script
# aliases (Korean 한국어 decomposes under NFKD; Vietnamese tiếng việt loses its
# diacritics) match. Language-agnostic: any future native-script alias works.
_NORMALIZED_TRUST_GUIDANCE_ALIASES = {
    _normalize_response_language(alias): language_key
    for alias, language_key in _REQUIRED_TRUST_GUIDANCE_LANGUAGE_ALIASES.items()
}


def _lookup_language_alias(normalized_language: str) -> Optional[str]:
    language_key = _NORMALIZED_TRUST_GUIDANCE_ALIASES.get(normalized_language)
    if language_key is None:
        for alias, alias_language_key in _NORMALIZED_TRUST_GUIDANCE_ALIASES.items():
            if (
                normalized_language.startswith(f"{alias} ")
                or normalized_language.startswith(f"{alias}-")
                or normalized_language.startswith(f"{alias} (")
            ):
                language_key = alias_language_key
                break
    return language_key


def _message_has_language_signal(text: str) -> bool:
    """A message with no letters in any script (bare ZIP, digits, punctuation)
    carries no language signal and must not shift the conversation language."""
    return any(unicodedata.category(ch).startswith("L") for ch in str(text or ""))


_SCRIPT_NAME_PREFIXES = (
    ("CJK UNIFIED", "Chinese"),
    # Compatibility ideographs (U+F900-FAFF) are also Han: without this
    # prefix they would count toward the letter total but credit no bucket,
    # diluting the strict-majority threshold.
    ("CJK COMPATIBILITY IDEOGRAPH", "Chinese"),
    ("HANGUL", "Korean"),
    ("ARABIC", "Arabic"),
)


def _dominant_user_script_language(texts) -> Optional[str]:
    """Deterministic conversation-language evidence from character scripts.

    Counts letter characters across the given user texts; if one non-Latin
    script (Han/Hangul/Arabic) holds a strict majority of ALL letters, return
    its language name. Latin-script languages are indistinguishable from
    English at this layer, so Latin/mixed input returns None.
    """
    counts = {label: 0 for _, label in _SCRIPT_NAME_PREFIXES}
    total_letters = 0
    for text in texts or ():
        for ch in str(text or ""):
            if not unicodedata.category(ch).startswith("L"):
                continue
            total_letters += 1
            try:
                char_name = unicodedata.name(ch)
            except ValueError:
                continue
            for prefix, label in _SCRIPT_NAME_PREFIXES:
                if char_name.startswith(prefix):
                    counts[label] += 1
                    break
    if not total_letters:
        return None
    label, top = max(counts.items(), key=lambda item: item[1])
    if top * 2 > total_letters:
        return label
    return None


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


class LanguageMixin:
    # ------------------------------------------------------------------
    def _localize_reply_via_llm(
        self,
        client,
        reply_text: str,
        target_language: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Translate the wrapper copy of a rendered results reply into target_language,
        keeping provider data (names, addresses, phones, ZIPs, URLs) verbatim.

        Returns the original reply unchanged on any failure, so an unsupported language
        is never worse off than the current English fallback.
        """
        source = str(reply_text or "").strip()
        if not source or not str(target_language).strip():
            return reply_text

        system = "You are a professional translator for a healthcare navigation assistant."
        instructions = (
            "Translate the care-navigation reply below into {language}.\n"
            "Keep every provider name, street address, phone number, ZIP code, URL, and "
            "email address EXACTLY as written — do not translate or transliterate them.\n"
            "Translate all other text: headings, labels, guidance, and safety notes.\n"
            "Preserve the Markdown structure and line breaks.\n"
            "Return ONLY the translated reply, with no preamble.\n\n"
            "Reply:\n{reply}"
        ).format(language=str(target_language).strip(), reply=source)

        try:
            completion = client.chat_completion(
                normalize_chat_messages(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": instructions},
                    ]
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choices = getattr(completion, "choices", None) or []
            if not choices:
                return reply_text
            message = getattr(choices[0], "message", None)
            translated = (getattr(message, "content", "") or "").strip() if message else ""
            return translated or reply_text
        except Exception as exc:  # noqa: BLE001 - never make the reply worse than English
            logger.warning("Reply localization to %s failed: %s", target_language, exc)
            return reply_text
