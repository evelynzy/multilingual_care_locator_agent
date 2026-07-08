import unittest

from provider_search.sources.clinicaltables import (
    _UMBRELLA_TAXONOMY_TERMS,
    ClinicalTablesSource,
)
from provider_search.specialty_families import normalize_specialty_family_id


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


class UmbrellaMapCoversBrokenFamiliesTests(unittest.TestCase):
    """Every specialty family NPI files under a different taxonomy name (F5)
    must be rewritten by the umbrella map; the mapped values are live-verified
    against the real service (probe at ZIP 94110, 2026-07-08)."""

    def test_every_f5_family_term_is_rewritten(self):
        for term in (
            "orthopedics", "endocrinology", "pulmonology", "rheumatology",
            "nephrology", "oncology", "physical therapy",
        ):
            self.assertIn(term, _UMBRELLA_TAXONOMY_TERMS, term)
            self.assertNotEqual(_UMBRELLA_TAXONOMY_TERMS[term], term, term)

    def test_internal_medicine_subspecialties_use_full_display_names(self):
        # Bare subspecialty names ("pulmonary disease", "hematology & oncology")
        # return zero from NPI; only the full "internal medicine, ..." display
        # form matches. Pin the three values a plainer name would silently break.
        self.assertEqual(
            _UMBRELLA_TAXONOMY_TERMS["endocrinology"],
            "internal medicine, endocrinology, diabetes & metabolism",
        )
        self.assertEqual(
            _UMBRELLA_TAXONOMY_TERMS["pulmonology"],
            "internal medicine, pulmonary disease",
        )
        self.assertEqual(
            _UMBRELLA_TAXONOMY_TERMS["oncology"],
            "internal medicine, hematology & oncology",
        )

    def test_rewrites_are_casefolded_keys(self):
        for key in _UMBRELLA_TAXONOMY_TERMS:
            self.assertEqual(key, key.casefold(), key)

    def test_every_umbrella_key_classifies_to_a_family(self):
        # The ranking gate derives family ids from the REQUESTED term; a dict
        # key with no family alias fetches providers that are then all dropped
        # as specialty_mismatch (the "orthopedist" gap found 2026-07-08).
        for key in _UMBRELLA_TAXONOMY_TERMS:
            self.assertIsNotNone(normalize_specialty_family_id(key), key)

    def test_every_umbrella_key_and_value_share_a_family(self):
        # Retrieval searches the VALUE; the gate classifies providers by what
        # the value's taxonomy maps to and the request by what the KEY maps
        # to. If those differ, retrieval and gating disagree and results
        # silently vanish.
        for key, value in _UMBRELLA_TAXONOMY_TERMS.items():
            self.assertEqual(
                normalize_specialty_family_id(key),
                normalize_specialty_family_id(value),
                f"{key!r} -> {value!r}",
            )


if __name__ == "__main__":
    unittest.main()
