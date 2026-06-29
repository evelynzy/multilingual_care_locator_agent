"""Edge-case ranking/gating tests for provider_search.ranking.

Covers three high-risk behaviors the audit flagged:
- specialty-family bridge: a request whose text differs from the provider's
  taxonomy still matches via derived family ids ("Ob/Gyn" vs "Obstetrics & Gynecology");
- a misspelled specialty must NOT falsely match a real specialist;
- same-city/different-state location alignment (a known precision limitation,
  pinned here with a NOTE so a future fix is detected).
"""

import unittest

from provider_search.models import ProviderSearchRequest
from provider_search.normalization import build_canonical_provider
from provider_search.ranking import (
    evaluate_provider_gate,
    rank_provider_results,
)
from provider_search.specialty_families import derive_request_specialty_family_ids


def _provider(**overrides):
    base = dict(
        provider_id="npi-1",
        name="Test Provider",
        source_name="ClinicalTables",
        dataset="npi_idv",
    )
    base.update(overrides)
    return build_canonical_provider(**base)


class SpecialtyFamilyBridgeTests(unittest.TestCase):
    def test_obgyn_request_matches_obstetrics_gynecology_via_family_id(self):
        # "Ob/Gyn" text does not equal "Obstetrics & Gynecology" text, so this
        # only admits if the family-id bridge in _match_specialties works.
        family_ids = derive_request_specialty_family_ids(("Ob/Gyn",))
        self.assertTrue(family_ids, "expected 'Ob/Gyn' to derive a specialty family")

        provider = _provider(
            name="Cupertino OB/GYN Associates",
            specialties=("Obstetrics & Gynecology",),
            specialty_family_ids=family_ids,
            taxonomy="Obstetrics & Gynecology",
        )
        request = ProviderSearchRequest(specialties=("Ob/Gyn",))

        evaluation = evaluate_provider_gate(request, provider)

        self.assertTrue(evaluation.admitted)
        self.assertTrue(evaluation.matched_specialties)

    def test_misspelled_specialty_does_not_match_real_specialist(self):
        dermatology_family = derive_request_specialty_family_ids(("Dermatology",))
        provider = _provider(
            name="Bay Dermatology",
            specialties=("Dermatology",),
            specialty_family_ids=dermatology_family,
            taxonomy="Dermatology",
        )
        # A typo should not silently resolve to dermatology.
        request = ProviderSearchRequest(specialties=("Dermatologee",))

        evaluation = evaluate_provider_gate(request, provider)

        self.assertFalse(evaluation.admitted)
        self.assertEqual(evaluation.drop_reason, "specialty_mismatch")
        self.assertEqual(evaluation.matched_specialties, ())


class LocationAlignmentTests(unittest.TestCase):
    def _location_alignment(self, result):
        return result.provider.ranking_metadata["score_breakdown"]["location_alignment"]

    def test_same_city_different_state_currently_scores_equal(self):
        # No specialty filter, so both providers are admitted and differ only by
        # state. _build_score_breakdown gives location_alignment = 1.0 on ANY token
        # overlap, so "Los Angeles TX" earns the same location credit as
        # "Los Angeles CA" for a CA search.
        ca_provider = _provider(
            provider_id="npi-ca",
            name="LA CA Clinic",
            city="Los Angeles",
            state="CA",
        )
        tx_provider = _provider(
            provider_id="npi-tx",
            name="LA TX Clinic",
            city="Los Angeles",
            state="TX",
        )
        request = ProviderSearchRequest(location="Los Angeles CA")

        ranked = rank_provider_results(request, [ca_provider, tx_provider])
        # provider_id is canonicalized by build_canonical_provider, so key by the
        # preserved display name instead.
        by_name = {r.provider.name: r for r in ranked}

        # NOTE: current behavior (suspected precision limitation): the out-of-state
        # provider gets full location credit. Pinned so a future state-aware fix
        # surfaces here as an unexpected change.
        self.assertEqual(self._location_alignment(by_name["LA CA Clinic"]), 1.0)
        self.assertEqual(self._location_alignment(by_name["LA TX Clinic"]), 1.0)


if __name__ == "__main__":
    unittest.main()
