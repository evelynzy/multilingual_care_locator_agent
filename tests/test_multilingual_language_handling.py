"""Deterministic tests for the language-handling plumbing (all 7 languages).

The real LLM language *detection* step is an integration concern and stays out of
scope here. These tests exercise the pure, deterministic functions that decide,
given a (possibly messy) response-language string: which trust-guidance language
to use and which render-copy locale to use. The repo already tests render copy
and per-language guidance through _compose_result_card_response; this file adds
direct coverage of the alias/normalization layer across every supported language,
plus the unsupported-language fallback.
"""

import sys
import types
import unittest


# Match the existing test convention: stub the heavy deps so care imports.
if "huggingface_hub" not in sys.modules:
    _hf = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InferenceClient stub should not be used in tests")

    _hf.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = _hf


from care import (
    _REQUIRED_TRUST_GUIDANCE,
    _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE,
    _get_prewritten_required_trust_guidance,
    _normalize_response_language,
    _resolved_supported_language_key,
)

G = _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE


class LanguageNormalizationTests(unittest.TestCase):
    def test_strips_case_accents_and_whitespace(self):
        self.assertEqual(_normalize_response_language("Español"), "espanol")
        self.assertEqual(_normalize_response_language("  MANDARIN  "), "mandarin")
        self.assertEqual(_normalize_response_language("ZH-CN"), "zh-cn")

    def test_empty_and_none_normalize_to_empty(self):
        self.assertEqual(_normalize_response_language(None), "")
        self.assertEqual(_normalize_response_language(""), "")
        self.assertEqual(_normalize_response_language("   "), "")


class TrustGuidanceResolutionTests(unittest.TestCase):
    # Aliases that DO resolve, one or more per supported language.
    WORKING_ALIASES = [
        ("en", "english"),
        ("English", "english"),
        ("ENG", "english"),
        ("es", "spanish"),
        ("Español", "spanish"),
        ("spanish", "spanish"),
        ("zh", "simplified_chinese"),
        ("zh-cn", "simplified_chinese"),
        ("中文", "simplified_chinese"),
        ("简体中文", "simplified_chinese"),
        ("Mandarin", "simplified_chinese"),
        ("Mandarin Chinese", "simplified_chinese"),
        ("vi", "vietnamese"),
        ("vietnamese", "vietnamese"),
        ("tl", "tagalog"),
        ("Tagalog", "tagalog"),
        ("filipino", "tagalog"),
        ("ar", "arabic"),
        ("Arabic", "arabic"),
        ("العربية", "arabic"),
        ("ko", "korean"),
        ("Korean", "korean"),
    ]

    def test_supported_aliases_resolve_to_their_language_guidance(self):
        for alias, expected_key in self.WORKING_ALIASES:
            with self.subTest(alias=alias):
                self.assertEqual(_get_prewritten_required_trust_guidance(alias), G[expected_key])

    def test_unknown_markers_use_english_guidance(self):
        for marker in ("unknown", "n/a", "none", None):
            with self.subTest(marker=marker):
                self.assertEqual(_get_prewritten_required_trust_guidance(marker), _REQUIRED_TRUST_GUIDANCE)

    def test_unsupported_language_has_no_prewritten_guidance(self):
        # A genuinely unsupported language yields None so the caller can fall back.
        self.assertIsNone(_get_prewritten_required_trust_guidance("russian"))

    def test_locale_tag_prefixes_resolve(self):
        self.assertEqual(_get_prewritten_required_trust_guidance("English (US)"), G["english"])
        self.assertEqual(_get_prewritten_required_trust_guidance("es-mx"), G["spanish"])


class RenderLocaleResolutionTests(unittest.TestCase):
    def test_render_copy_locales_resolve(self):
        self.assertEqual(_resolved_supported_language_key("en"), "english")
        self.assertEqual(_resolved_supported_language_key("Español"), "spanish")
        self.assertEqual(_resolved_supported_language_key("中文"), "simplified_chinese")

    def test_all_known_languages_resolve_to_native_render(self):
        # Every product-recognized language renders natively from the committed
        # locale files (previously vi/tl/ar/ko fell back to the English render).
        for alias, expected_key in (
            ("vietnamese", "vietnamese"),
            ("Tagalog", "tagalog"),
            ("Arabic", "arabic"),
            ("Korean", "korean"),
        ):
            with self.subTest(alias=alias):
                self.assertEqual(_resolved_supported_language_key(alias), expected_key)

    def test_unsupported_language_renders_in_english(self):
        self.assertEqual(_resolved_supported_language_key("russian"), "english")


class NativeScriptAliasGapTests(unittest.TestCase):
    """Native-script aliases must resolve to their language (regression guard)."""

    def test_korean_native_script_should_resolve_to_korean(self):
        self.assertEqual(_get_prewritten_required_trust_guidance("한국어"), G["korean"])

    def test_vietnamese_native_script_should_resolve_to_vietnamese(self):
        self.assertEqual(_get_prewritten_required_trust_guidance("tiếng việt"), G["vietnamese"])

    def test_native_script_aliases_resolve_for_all_distinct_scripts(self):
        cases = [
            ("中文", "simplified_chinese"),
            ("简体中文", "simplified_chinese"),
            ("한국어", "korean"),
            ("tiếng việt", "vietnamese"),
            ("العربية", "arabic"),
        ]
        for alias, expected_key in cases:
            with self.subTest(alias=alias):
                self.assertEqual(_get_prewritten_required_trust_guidance(alias), G[expected_key])


if __name__ == "__main__":
    unittest.main()
