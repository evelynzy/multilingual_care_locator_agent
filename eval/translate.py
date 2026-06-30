from __future__ import annotations

import json
from typing import List

LANGUAGE_NAMES = {"zh": "Chinese", "es": "Spanish", "ar": "Arabic", "ko": "Korean"}

_SYSTEM = (
    "You are a professional medical translator. Translate the user's healthcare "
    "search query into {language}. Preserve all digits and US ZIP codes exactly. "
    "Translate medical terms naturally as a layperson in that language would phrase them. "
    "Return ONLY the translated text, with no quotes, labels, or explanation."
)


def _extract_text(completion) -> str:
    if not getattr(completion, "choices", None):
        return ""
    message = getattr(completion.choices[0], "message", None)
    if message is None:
        return ""
    return (getattr(message, "content", "") or "").strip()


def translate_turns(turns: List[str], language: str, client, model: str) -> List[str]:
    language_name = LANGUAGE_NAMES[language]
    translated: List[str] = []
    for turn in turns:
        messages = [
            {"role": "system", "content": _SYSTEM.format(language=language_name)},
            {"role": "user", "content": "Text:\n{0}".format(turn)},
        ]
        completion = client.chat_completion(messages=messages, model=model, max_tokens=200)
        text = _extract_text(completion)
        translated.append(text if text else turn)
    return translated


def fill_missing_variants(path: str, client, model: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    added = 0
    for scenario in payload["scenarios"]:
        en_turns = scenario["variants"]["en"]["turns"]
        for language in LANGUAGE_NAMES:
            if language in scenario["variants"]:
                continue
            translated = translate_turns(en_turns, language, client, model)
            scenario["variants"][language] = {
                "turns": translated,
                "source": "mt",
                "verification_status": "mt_only",
                "verified_by": None,
                "notes": "machine-translated; not verified by a {0} speaker".format(LANGUAGE_NAMES[language]),
            }
            added += 1

    if added:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    return added
