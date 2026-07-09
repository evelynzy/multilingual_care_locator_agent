"""Unit tests for the deterministic PHI input guard (care/privacy.py)."""

import unittest

from care.privacy import PHIMatch, RedactionResult, fold_digits, redact_phi, scan_phi


class ScanPhiDetectorTests(unittest.TestCase):
    def _types(self, text):
        return [m.phi_type for m in scan_phi(text)]

    # --- positives -------------------------------------------------------
    def test_ssn_hyphenated(self):
        self.assertEqual(self._types("my ssn is 123-45-6789"), ["ssn"])

    def test_ssn_spaced_and_dotted(self):
        self.assertEqual(self._types("123 45 6789"), ["ssn"])
        self.assertEqual(self._types("123.45.6789"), ["ssn"])

    def test_phone_us_formats(self):
        self.assertEqual(self._types("call me at (415) 555-0123"), ["phone"])
        self.assertEqual(self._types("415-555-0123"), ["phone"])
        self.assertEqual(self._types("+1 415 555 0123"), ["phone"])

    def test_phone_bare_10_and_11_digit_runs(self):
        self.assertEqual(self._types("4155550123"), ["phone"])
        self.assertEqual(self._types("13812345678"), ["phone"])

    def test_phone_3_4_4_grouping(self):
        # Chinese/Korean mobile grouping (138-1234-5678, 010-1234-5678).
        self.assertEqual(self._types("138-1234-5678"), ["phone"])
        self.assertEqual(self._types("010-1234-5678"), ["phone"])

    def test_email(self):
        self.assertEqual(self._types("reach me at jane.doe+x@example.org please"), ["email"])

    def test_date_formats(self):
        self.assertEqual(self._types("dob 01/02/1985"), ["date"])
        self.assertEqual(self._types("born 1985-01-02"), ["date"])
        self.assertEqual(self._types("02-01-1985"), ["date"])

    def test_id_number_six_plus_digit_run(self):
        self.assertEqual(self._types("member id 123456"), ["id_number"])
        self.assertEqual(self._types("mrn 12345678"), ["id_number"])

    def test_bare_nine_digit_run_is_id_number(self):
        # Unseparated SSN is the likelier/riskier reading than unseparated ZIP+4.
        self.assertEqual(self._types("123456789"), ["id_number"])

    # --- ZIP protection (hard rule) ---------------------------------------
    def test_zip_five_digits_never_flagged(self):
        self.assertEqual(scan_phi("primary care 10001"), ())
        self.assertEqual(scan_phi("pediatrician 94110"), ())

    def test_zip_plus_four_never_flagged(self):
        self.assertEqual(scan_phi("dermatologist 94110-1234"), ())

    def test_cjk_glued_zip_never_flagged(self):
        self.assertEqual(scan_phi("儿科10013"), ())

    # --- precedence and overlap -------------------------------------------
    def test_email_wins_over_digit_rules_inside_address(self):
        matches = scan_phi("a1234567@example.com")
        self.assertEqual([m.phi_type for m in matches], ["email"])

    def test_multiple_types_in_one_message(self):
        text = "ssn 123-45-6789 phone (415) 555-0123 member 987654"
        self.assertEqual(self._types(text), ["ssn", "phone", "id_number"])


class RedactPhiTests(unittest.TestCase):
    def test_replaces_with_placeholders_and_reports_matches(self):
        result = redact_phi("my ssn is 123-45-6789, find cardiology 94110")
        self.assertIsInstance(result, RedactionResult)
        self.assertEqual(result.text, "my ssn is [REDACTED: SSN], find cardiology 94110")
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].phi_type, "ssn")
        self.assertEqual(result.matches[0].matched_text, "123-45-6789")

    def test_clean_text_is_byte_identical_passthrough(self):
        text = "primary care 10001"
        result = redact_phi(text)
        self.assertEqual(result.text, text)
        self.assertEqual(result.matches, ())

    def test_idempotent_on_already_redacted_text(self):
        once = redact_phi("call 415-555-0123").text
        twice = redact_phi(once)
        self.assertEqual(twice.text, once)
        self.assertEqual(twice.matches, ())

    def test_empty_and_none_safe(self):
        self.assertEqual(redact_phi("").text, "")


class FoldDigitsTests(unittest.TestCase):
    def test_arabic_indic_digits_fold(self):
        self.assertEqual(fold_digits("١٢٣-٤٥-٦٧٨٩"), "123-45-6789")

    def test_fullwidth_digits_fold(self):
        self.assertEqual(fold_digits("１３８１２３４５６７８"), "13812345678")

    def test_ascii_and_non_digits_untouched(self):
        text = "primary care 10001 — 儿科"
        self.assertEqual(fold_digits(text), text)

    def test_length_preserved(self):
        source = "رقم ٩٨٧٦٥٤٣٢١"
        self.assertEqual(len(fold_digits(source)), len(source))


class ScanPhiFoldedDigitTests(unittest.TestCase):
    def test_arabic_indic_date_detected(self):
        # The date pattern's year anchor is the ASCII literal (?:19|20) — the
        # one place Unicode \d does not save us (FINDINGS F9).
        matches = scan_phi("تاريخ الميلاد ١٩٨٥-٠١-٠٢")
        self.assertEqual([m.phi_type for m in matches], ["date"])
        # matched_text preserves the ORIGINAL script:
        self.assertEqual(matches[0].matched_text, "١٩٨٥-٠١-٠٢")

    def test_arabic_indic_ssn_detected(self):
        matches = scan_phi("رقم الضمان ١٢٣-٤٥-٦٧٨٩")
        self.assertEqual([m.phi_type for m in matches], ["ssn"])
        self.assertEqual(matches[0].matched_text, "١٢٣-٤٥-٦٧٨٩")

    def test_arabic_indic_zip_still_exempt(self):
        self.assertEqual(scan_phi("طبيب أطفال ٩٤١١٠"), ())

    def test_fullwidth_phone_detected(self):
        self.assertEqual([m.phi_type for m in scan_phi("１３８１２３４５６７８")], ["phone"])


if __name__ == "__main__":
    unittest.main()
