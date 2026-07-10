"""Offline locale generator: translate the English master strings once per
known language and write committed locale files (care/locales/<key>.json).

Run manually (never at app runtime):
    .venv/bin/python -m care.generate_locales            # all six languages
    .venv/bin/python -m care.generate_locales korean     # one language

English is the single source of every translation (always English -> X). When
English copy changes, edit the master in code and re-run this script.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

TRANSLATION_MODEL = "Qwen/Qwen2.5-72B-Instruct"

LOCALE_LANGUAGE_NAMES = {
    "spanish": "Spanish",
    "simplified_chinese": "Simplified Chinese",
    "arabic": "Arabic",
    "korean": "Korean",
    "vietnamese": "Vietnamese",
    "tagalog": "Tagalog",
}

AUTO_TRANSLATED_MARK_EN = "Safety notes auto-translated from English."

LOCALES_DIR = Path(__file__).parent / "locales"

_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")

_SYSTEM = (
    "You are a professional translator for a healthcare navigation product. "
    "Translate the given UI string into {language}. Keep every placeholder "
    "token such as {{summary}} or {{value}} EXACTLY as written (curly braces "
    "and the word inside unchanged). Preserve line breaks and any leading "
    "symbols. Return ONLY the translated string."
)


def build_english_master() -> dict:
    from care.rendering import _DETERMINISTIC_RENDER_COPY, _DETERMINISTIC_RENDER_TRANSLATIONS
    from care.safety import _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE

    return {
        "copy": _DETERMINISTIC_RENDER_COPY["english"],
        "sentences": list(_DETERMINISTIC_RENDER_TRANSLATIONS.keys()),
        "trust_guidance": _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE["english"],
        "auto_translated_mark": AUTO_TRANSLATED_MARK_EN,
    }


def _extract_text(completion) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return (getattr(message, "content", "") or "").strip() if message else ""


def _translate_string(text: str, language_name: str, client, model: str) -> str:
    completion = client.chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM.format(language=language_name)},
            {"role": "user", "content": text},
        ],
        model=model,
        max_tokens=900,
        temperature=0.0,
    )
    translated = _extract_text(completion)
    if not translated:
        raise ValueError("empty translation for: {0!r}".format(text[:60]))
    # Multiset comparison: a duplicate placeholder dropped from the
    # translation must fail even when the set of distinct tokens matches.
    if sorted(_PLACEHOLDER_RE.findall(text)) != sorted(_PLACEHOLDER_RE.findall(translated)):
        raise ValueError("placeholder mismatch for: {0!r}".format(text[:60]))
    if text.count("\n") != translated.count("\n"):
        raise ValueError("line-structure mismatch for: {0!r}".format(text[:60]))
    return translated


def generate_locale(language_key: str, client, model: str) -> dict:
    master = build_english_master()
    language_name = LOCALE_LANGUAGE_NAMES[language_key]

    copy: Dict[str, object] = {}
    for key, value in master["copy"].items():
        if isinstance(value, dict):
            copy[key] = {
                sub_key: _translate_string(sub_value, language_name, client, model)
                for sub_key, sub_value in value.items()
            }
        else:
            copy[key] = _translate_string(value, language_name, client, model)

    sentences = {
        sentence: _translate_string(sentence, language_name, client, model)
        for sentence in master["sentences"]
    }

    return {
        "language_key": language_key,
        "source": "machine-translated from the English masters; not reviewed by a native speaker",
        "copy": copy,
        "sentences": sentences,
        "trust_guidance": _translate_string(master["trust_guidance"], language_name, client, model),
        "auto_translated_mark": _translate_string(
            master["auto_translated_mark"], language_name, client, model
        ),
    }


def main(argv: List[str]) -> int:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import InferenceClient

    load_dotenv()
    token = os.environ["HF_TOKEN"]
    client = InferenceClient(model=TRANSLATION_MODEL, token=token)

    targets = argv or list(LOCALE_LANGUAGE_NAMES)
    LOCALES_DIR.mkdir(exist_ok=True)
    for language_key in targets:
        locale = generate_locale(language_key, client, TRANSLATION_MODEL)
        path = LOCALES_DIR / "{0}.json".format(language_key)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(locale, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
