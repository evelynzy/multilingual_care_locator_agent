import unittest

from eval.phi_guard_eval import evaluate_corpus


class PhiGuardEvalTests(unittest.TestCase):
    def setUp(self):
        self.stats = evaluate_corpus()

    def test_corpus_loads_and_covers_all_languages(self):
        languages = {lang for lang, _ in self.stats}
        self.assertEqual(languages, {"en", "zh", "es", "ar", "ko"})

    def test_english_detection_is_complete(self):
        for phi_type in ("ssn", "phone", "email", "date", "id_number"):
            cell = self.stats[("en", phi_type)]
            self.assertEqual(cell["detected"], cell["total"], phi_type)

    def test_no_false_positives_in_any_language(self):
        for (language, key), cell in self.stats.items():
            if key == "negatives":
                self.assertEqual(cell["false_positives"], 0, language)


if __name__ == "__main__":
    unittest.main()
