import unittest

from eval.phi_guard_eval import evaluate_corpus


class PhiGuardEvalTests(unittest.TestCase):
    def setUp(self):
        self.stats = evaluate_corpus()

    def test_corpus_loads_and_covers_all_languages(self):
        languages = {lang for lang, _ in self.stats}
        self.assertEqual(languages, {"en", "zh", "es", "ar", "ko"})

    def test_detection_is_complete_for_every_language_and_type(self):
        # Parity pin for FINDINGS F9: post-fold, every positive corpus entry in
        # every language must detect. A regression here is a fairness bug.
        for (language, key), cell in self.stats.items():
            if key == "negatives":
                continue
            self.assertEqual(cell["detected"], cell["total"], "{0}/{1}".format(language, key))

    def test_no_false_positives_in_any_language(self):
        for (language, key), cell in self.stats.items():
            if key == "negatives":
                self.assertEqual(cell["false_positives"], 0, language)


if __name__ == "__main__":
    unittest.main()
