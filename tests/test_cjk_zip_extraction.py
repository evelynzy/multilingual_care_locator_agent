"""Regression tests: ZIP extraction must work when a 5-digit ZIP is glued to
CJK characters (e.g. a Chinese query typed without a space: "儿科10013").

Root cause: ``_extract_zip_code`` used ``\\b(\\d{5})\\b``. Python's Unicode regex
classifies CJK characters as word characters, so there is no ``\\b`` boundary
between e.g. "科" and "1", and the ZIP was never extracted. With the LLM intent
JSON frequently failing for Chinese input, the deterministic fallback then had
no location to search with, collapsing the result set to zero.
"""

import sys
import types
import unittest
from unittest.mock import Mock


if "huggingface_hub" not in sys.modules:
    _hf = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InferenceClient stub should not be used in tests")

    _hf.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = _hf


from care_agent import CareLocatorAgent


class CjkZipExtractionTests(unittest.TestCase):
    def setUp(self):
        self.agent = CareLocatorAgent(provider_search_service=Mock())

    def test_extract_zip_from_cjk_glued_digits(self):
        # The reported failing case: Chinese specialty term glued to the ZIP.
        self.assertEqual(self.agent._extract_zip_code("儿科10013"), "10013")
        self.assertEqual(self.agent._extract_zip_code("心脏科95051"), "95051")

    def test_rescue_location_from_cjk_glued_message(self):
        self.assertEqual(self.agent._rescue_location_from_message("儿科10013"), "10013")

    # --- regression guards: existing behavior must be preserved ---
    def test_spaced_and_english_zip_still_extracted(self):
        self.assertEqual(self.agent._extract_zip_code("儿科 10013"), "10013")
        self.assertEqual(self.agent._extract_zip_code("pediatrics 10013"), "10013")

    def test_zip_plus_four_still_extracted(self):
        self.assertEqual(self.agent._extract_zip_code("zip 10013-1234"), "10013")

    def test_six_digit_number_not_treated_as_zip(self):
        # A longer run of digits must not be mistaken for a 5-digit ZIP.
        self.assertIsNone(self.agent._extract_zip_code("code 100135"))

    # --- latin-glued ZIP: documents/locks the widened regex behavior ---

    def test_extract_zip_glued_to_latin_letters(self):
        # The widened regex (no \\b) also extracts when digits are glued to Latin text.
        self.assertEqual(self.agent._extract_zip_code("abc10013"), "10013")

    def test_seven_digit_run_glued_to_latin_not_extracted(self):
        # The 6+-digit guard must still hold for Latin-glued numbers.
        self.assertIsNone(self.agent._extract_zip_code("x1234567y"))


if __name__ == "__main__":
    unittest.main()
