import unittest

from provider_search.sources.clinicaltables import ClinicalTablesSource


class UmbrellaSpecialtyTermTests(unittest.TestCase):
    """The NPI registry has no 'primary care' taxonomy, so searching that
    umbrella term literally returns zero providers. When NPI offers no
    suggestion of its own, the umbrella term must fall back to a real
    NPI-recognized taxonomy (Family Medicine) so results are found."""

    def _source_without_npi_suggestion(self):
        src = ClinicalTablesSource()
        # Simulate NPI returning no taxonomy suggestion for the input term.
        src._best_suggestion_across_datasets = lambda *args, **kwargs: ""
        return src

    def test_primary_care_umbrella_expands_to_family_medicine(self):
        src = self._source_without_npi_suggestion()
        self.assertEqual(src.suggest_specialty_terms(("primary care",)), ("family medicine",))

    def test_primary_care_umbrella_is_case_insensitive(self):
        src = self._source_without_npi_suggestion()
        self.assertEqual(src.suggest_specialty_terms(("Primary Care",)), ("family medicine",))

    def test_umbrella_wins_even_when_npi_echoes_the_umbrella_term(self):
        # The real live failure: NPI's suggest endpoint returns a truthy
        # self-echo ("primary care") that still yields zero providers. The
        # umbrella map must override it, not defer to it.
        src = ClinicalTablesSource()
        src._best_suggestion_across_datasets = lambda *args, **kwargs: "primary care"
        self.assertEqual(src.suggest_specialty_terms(("primary care",)), ("family medicine",))

    def test_real_npi_suggestion_is_preferred_over_umbrella(self):
        src = ClinicalTablesSource()
        src._best_suggestion_across_datasets = lambda *args, **kwargs: "cardiovascular disease"
        self.assertEqual(src.suggest_specialty_terms(("cardiology",)), ("cardiovascular disease",))

    def test_unmapped_term_falls_back_to_literal(self):
        src = self._source_without_npi_suggestion()
        self.assertEqual(src.suggest_specialty_terms(("dermatology",)), ("dermatology",))


if __name__ == "__main__":
    unittest.main()
