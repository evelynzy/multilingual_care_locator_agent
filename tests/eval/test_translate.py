from __future__ import annotations

import json
import os
import tempfile
import unittest

from eval.translate import LANGUAGE_NAMES, fill_missing_variants


class _StubClient:
    """Returns a deterministic fake translation so the writer can be tested offline."""

    def __init__(self):
        self.calls = []

    def chat_completion(self, messages, model=None, max_tokens=None, temperature=None):
        self.calls.append((messages, model))
        user = messages[-1]["content"]
        text = "[t]" + user.split("Text:\n", 1)[-1].strip()

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class TranslateWriterTests(unittest.TestCase):
    def test_language_names_cover_non_english(self):
        self.assertEqual(set(LANGUAGE_NAMES), {"zh", "es", "ar", "ko"})

    def test_fill_writes_four_variants_per_english_only_scenario(self):
        data = {
            "languages": ["en", "zh", "es", "ar", "ko"],
            "scenarios": [{
                "id": "s01", "category": "c", "dimension": "d",
                "gold": {"expected_specialty": "cardiology", "expected_state": "CA",
                         "expect_followup": False, "expect_nonzero_providers": True,
                         "expect_emergency_routing": False, "expected_preferred_language": None},
                "variants": {"en": {"turns": ["cardiology 98101"], "source": "english_seed",
                                     "verification_status": "human_verified", "verified_by": "author", "notes": None}},
            }],
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

            added = fill_missing_variants(path, _StubClient(), model="fake-model")
            self.assertEqual(added, 4)

            with open(path, "r", encoding="utf-8") as fh:
                out = json.load(fh)
            variants = out["scenarios"][0]["variants"]
            self.assertEqual(set(variants), {"en", "zh", "es", "ar", "ko"})
            self.assertEqual(variants["zh"]["source"], "mt")
            self.assertEqual(variants["zh"]["verification_status"], "mt_only")
            self.assertEqual(len(variants["zh"]["turns"]), 1)

    def test_fill_is_idempotent(self):
        data = {
            "languages": ["en", "zh", "es", "ar", "ko"],
            "scenarios": [{
                "id": "s01", "category": "c", "dimension": "d",
                "gold": {"expected_specialty": None, "expected_state": None,
                         "expect_followup": True, "expect_nonzero_providers": False,
                         "expect_emergency_routing": False, "expected_preferred_language": None},
                "variants": {"en": {"turns": ["mental health 98101"], "source": "english_seed",
                                     "verification_status": "human_verified", "verified_by": "author", "notes": None}},
            }],
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            self.assertEqual(fill_missing_variants(path, _StubClient(), model="fake-model"), 4)
            self.assertEqual(fill_missing_variants(path, _StubClient(), model="fake-model"), 0)

    def test_translate_turns_falls_back_when_choice_has_no_message(self):
        from eval.translate import translate_turns

        class _NoMessageClient:
            def chat_completion(self, messages, model=None, max_tokens=None, temperature=None):
                class _Choice:
                    pass  # malformed: no .message attribute

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        result = translate_turns(["cardiology 98101"], "zh", _NoMessageClient(), model="m")
        self.assertEqual(result, ["cardiology 98101"])


if __name__ == "__main__":
    unittest.main()
