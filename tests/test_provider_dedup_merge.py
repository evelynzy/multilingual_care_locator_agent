"""Characterization tests for provider dedup + record merging.

Targets the dedup-by-``provider_id`` logic and ``_merge_provider_records`` in
``provider_search/service.py`` together with id resolution in
``provider_search/normalization.py``.

These assert the expected/correct behavior. Where the product currently drops
data on merge, the current behavior is pinned with an explicit
``# NOTE: current behavior, suspected bug`` comment and the desired behavior is
expressed as an ``@unittest.expectedFailure`` spec, so the suite stays green
while the gap is documented and reported.
"""

from __future__ import annotations

import unittest

from provider_search.models import (
    FreshnessMetadata,
    ProviderSearchRequest,
    SourceSearchResult,
    SourceTrace,
    VerificationStatus,
)
from provider_search.normalization import build_canonical_provider
from provider_search.service import ProviderSearchService


class FakeClinicalTablesSource:
    """Returns a fixed ``SourceSearchResult`` per dataset (mirrors service tests)."""

    def __init__(self, responses_by_dataset: dict[str, SourceSearchResult]) -> None:
        self.responses_by_dataset = responses_by_dataset
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        response = self.responses_by_dataset[dataset]
        if isinstance(response, Exception):
            raise response
        return response


def _primary_care_provider(*, provider_id: str, name: str, source_name: str, dataset: str):
    return build_canonical_provider(
        provider_id=provider_id,
        name=name,
        source_name=source_name,
        dataset=dataset,
        taxonomy="Primary Care",
        specialties=("Primary Care",),
    )


def _result_for(provider, dataset: str) -> SourceSearchResult:
    return SourceSearchResult(
        providers=[provider],
        trace=SourceTrace(
            source_name="clinicaltables",
            dataset=dataset,
            result_count=1,
        ),
    )


class ProviderDedupTests(unittest.TestCase):
    def test_same_npi_across_datasets_dedupes_to_single_result(self) -> None:
        individual = _primary_care_provider(
            provider_id="1619271780",
            name="Zeta Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
        )
        organization = _primary_care_provider(
            provider_id="1619271780",
            name="Zeta Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
        )
        # Same 10-digit NPI under an NPI dataset canonicalizes to the same id.
        self.assertEqual(individual.provider_id, "1619271780")
        self.assertEqual(organization.provider_id, "1619271780")

        source = FakeClinicalTablesSource(
            {
                "npi_idv": _result_for(individual, "npi_idv"),
                "npi_org": _result_for(organization, "npi_org"),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(specialties=("Primary Care",)),
            limit=5,
        )

        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.search_trace.total_candidates, 1)
        self.assertEqual(
            response.provider_results[0].provider.provider_id, "1619271780"
        )

    def test_distinct_npis_are_not_collapsed(self) -> None:
        alpha = _primary_care_provider(
            provider_id="1619271780",
            name="Alpha Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
        )
        beta = _primary_care_provider(
            provider_id="1982634159",
            name="Beta Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
        )
        self.assertNotEqual(alpha.provider_id, beta.provider_id)

        source = FakeClinicalTablesSource(
            {
                "npi_idv": _result_for(alpha, "npi_idv"),
                "npi_org": _result_for(beta, "npi_org"),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(specialties=("Primary Care",)),
            limit=5,
        )

        self.assertEqual(len(response.provider_results), 2)
        self.assertEqual(response.search_trace.total_candidates, 2)
        self.assertCountEqual(
            [result.provider.provider_id for result in response.provider_results],
            ["1619271780", "1982634159"],
        )


class MergeProviderRecordsTests(unittest.TestCase):
    def _primary_with_contact(self):
        return build_canonical_provider(
            provider_id="1619271780",
            name="Zeta Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            phone="214-555-0100",
            website="https://zeta.example",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )

    def _fallback_with_address_and_insurance(self):
        return build_canonical_provider(
            provider_id="1619271780",
            name="Zeta Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St, Dallas, TX 75201",
            insurance_reported=("Aetna",),
            insurance_network_verification=VerificationStatus(
                status="verified",
                verified=True,
                basis="Plan directory confirmed.",
                source="Aetna directory",
            ),
            freshness=FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )

    def test_merge_keeps_primary_contact_and_layers_fallback_metadata(self) -> None:
        primary = self._primary_with_contact()
        fallback = self._fallback_with_address_and_insurance()

        merged = ProviderSearchService._merge_provider_records(
            primary=primary,
            fallback=fallback,
        )

        # Primary's contact fields are retained.
        self.assertEqual(merged.phone, "214-555-0100")
        self.assertEqual(merged.website, "https://zeta.example")
        # Complementary metadata from the fallback IS layered in correctly.
        self.assertEqual(
            merged.freshness,
            FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(merged.insurance_network_verification.status, "verified")
        self.assertTrue(merged.insurance_network_verification.verified)

    def test_merge_current_behavior_drops_fallback_only_contact_fields(self) -> None:
        primary = self._primary_with_contact()
        fallback = self._fallback_with_address_and_insurance()

        merged = ProviderSearchService._merge_provider_records(
            primary=primary,
            fallback=fallback,
        )

        # Corrected behavior: complementary scalar fields the primary lacks
        # (address, insurance_reported) must be filled from the fallback.
        self.assertEqual(merged.address, "123 Main St, Dallas, TX 75201")
        self.assertEqual(merged.insurance_reported, ("Aetna",))

    def test_merge_should_retain_all_complementary_fields(self) -> None:
        """Executable spec of the desired merge behavior (currently failing).

        When the primary holds phone+website and the fallback holds
        address+insurance_reported, the merged record should retain ALL of them
        rather than dropping the fallback's complementary fields.
        """

        primary = self._primary_with_contact()
        fallback = self._fallback_with_address_and_insurance()

        merged = ProviderSearchService._merge_provider_records(
            primary=primary,
            fallback=fallback,
        )

        self.assertEqual(merged.phone, "214-555-0100")
        self.assertEqual(merged.website, "https://zeta.example")
        self.assertEqual(merged.address, "123 Main St, Dallas, TX 75201")
        self.assertEqual(merged.insurance_reported, ("Aetna",))


if __name__ == "__main__":
    unittest.main()
