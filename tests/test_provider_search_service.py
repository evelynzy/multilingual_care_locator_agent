from __future__ import annotations

import unittest

from provider_search.models import (
    FreshnessMetadata,
    ProviderSearchCacheEntry,
    ProviderSearchRequest,
    SourceSearchResult,
    SourceTrace,
    VerificationStatus,
)
from provider_search.normalization import build_canonical_provider
from provider_search.ranking import RANKING_VERSION, rank_provider_results
from provider_search.service import ProviderSearchService


class FakeClinicalTablesSource:
    def __init__(self, responses_by_dataset: dict[str, SourceSearchResult]) -> None:
        self.responses_by_dataset = responses_by_dataset
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        response = self.responses_by_dataset[dataset]
        if isinstance(response, Exception):
            raise response
        return response


class FakeCache:
    def __init__(
        self,
        entry: ProviderSearchCacheEntry | None = None,
        *,
        get_error: Exception | None = None,
        set_error: Exception | None = None,
    ) -> None:
        self.entry = entry
        self.get_error = get_error
        self.set_error = set_error
        self.get_calls: list[str] = []
        self.set_entries: list[ProviderSearchCacheEntry] = []

    def get(self, cache_key: str) -> ProviderSearchCacheEntry | None:
        self.get_calls.append(cache_key)
        if self.get_error is not None:
            raise self.get_error
        return self.entry

    def set(self, entry: ProviderSearchCacheEntry) -> bool:
        self.set_entries.append(entry)
        if self.set_error is not None:
            raise self.set_error
        return True


class ProviderSearchRankingTests(unittest.TestCase):
    def test_rank_provider_results_prioritizes_request_alignment(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Primary Care",),
            location="Pittsburgh, PA",
            insurance=("Aetna",),
            preferred_languages=("Spanish",),
            keywords=("same day",),
        )
        strong_match = build_canonical_provider(
            provider_id="provider-1",
            name="Harmony Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Pittsburgh",
            state="PA",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            languages=("Spanish",),
            insurance_reported=("Aetna",),
            insurance_network_verification=VerificationStatus(
                status="verified",
                verified=True,
                basis="Plan directory confirmed.",
                source="Aetna directory",
            ),
            accepting_new_patients_status=VerificationStatus(
                status="accepting",
                verified=True,
                basis="Clinic confirmed availability.",
                source="Clinic staff",
            ),
        ).with_updates(description="Same day primary care visits available.")
        partial_match = build_canonical_provider(
            provider_id="provider-2",
            name="Northside Medical Group",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Pittsburgh",
            state="PA",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            languages=("English",),
            insurance_reported=("Aetna",),
        )
        weak_match = build_canonical_provider(
            provider_id="provider-3",
            name="Riverside Cardiology",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Cleveland",
            state="OH",
            taxonomy="Cardiology",
            specialties=("Cardiology",),
            languages=("English",),
            insurance_reported=("Other Plan",),
        )

        ranked = rank_provider_results(request, [weak_match, partial_match, strong_match])

        self.assertEqual(
            [result.provider.provider_id for result in ranked],
            [
                strong_match.provider_id,
                partial_match.provider_id,
                weak_match.provider_id,
            ],
        )
        self.assertEqual(ranked[0].provider.ranking_metadata["ranking_version"], RANKING_VERSION)
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertGreater(ranked[1].score, ranked[2].score)

    def test_rank_provider_results_does_not_change_order_when_cache_state_changes(self) -> None:
        request = ProviderSearchRequest(specialties=("Primary Care",))
        cached_provider = build_canonical_provider(
            provider_id="provider-cached",
            name="Zeta Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        uncached_provider = build_canonical_provider(
            provider_id="provider-uncached",
            name="Alpha Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )

        ranked = rank_provider_results(
            request,
            [uncached_provider, cached_provider],
            cached_provider_ids=(cached_provider.provider_id,),
        )

        baseline = rank_provider_results(request, [uncached_provider, cached_provider])

        self.assertEqual(
            [result.provider.provider_id for result in ranked],
            [result.provider.provider_id for result in baseline],
        )
        self.assertEqual(ranked[0].provider.provider_id, uncached_provider.provider_id)
        self.assertFalse(ranked[0].provider.ranking_metadata["cached_identity_match"])
        self.assertTrue(ranked[1].provider.ranking_metadata["cached_identity_match"])
        self.assertEqual(ranked[1].retriever_metadata["cache_hint"], "matched_prior_result")


class ProviderSearchServiceTests(unittest.TestCase):
    def test_search_orchestrates_live_sources_and_updates_cache(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="provider-1",
            name="Harmony Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Pittsburgh",
            state="PA",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            languages=("Spanish",),
            insurance_reported=("Aetna",),
        )
        urgent_care_provider = build_canonical_provider(
            provider_id="provider-2",
            name="Downtown Walk-In",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Pittsburgh",
            state="PA",
            taxonomy="Urgent Care",
            specialties=("Urgent Care",),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[primary_care_provider],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        result_count=1,
                    ),
                ),
                "npi_org": SourceSearchResult(
                    providers=[urgent_care_provider],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_org",
                        result_count=1,
                    ),
                ),
            }
        )
        cache = FakeCache()
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=cache,
            per_dataset_limit=4,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Pittsburgh, PA 15213",
                insurance=("Aetna",),
                preferred_languages=("Spanish",),
                keywords=("same day",),
            ),
            limit=2,
        )

        self.assertEqual(len(source.calls), 2)
        first_dataset, first_request = source.calls[0]
        self.assertEqual(first_dataset, "npi_idv")
        self.assertEqual(first_request.search_terms, "Primary Care same day")
        self.assertEqual(first_request.city_hint, "Pittsburgh")
        self.assertEqual(first_request.state_hint, "PA")
        self.assertEqual(first_request.zip_hint, "15213")
        self.assertEqual(first_request.limit, 4)

        self.assertEqual(len(response.provider_results), 2)
        self.assertEqual(response.provider_results[0].provider.name, "Harmony Family Clinic")
        self.assertFalse(response.search_trace.cache_hit)
        self.assertEqual(
            response.search_trace.sources_attempted,
            ("clinicaltables:npi_idv", "clinicaltables:npi_org"),
        )
        self.assertEqual(response.search_trace.total_candidates, 2)
        self.assertEqual(len(cache.set_entries), 1)
        self.assertEqual(
            cache.set_entries[0].provider_ids,
            tuple(result.provider.provider_id for result in response.provider_results),
        )

    def test_search_treats_cache_as_non_authoritative_and_deduplicates_live_results(self) -> None:
        cached_entry = ProviderSearchCacheEntry(
            cache_key="provider-search:test",
            request_fingerprint="fingerprint-1",
            provider_ids=("source:clinicaltables:npi-idv:provider-123:cached",),
            sources=("NPI Registry (individual)",),
            stored_at="2026-04-22T12:00:00+00:00",
            expires_at=None,
        )
        cache = FakeCache(entry=cached_entry)
        provider_from_first_dataset = build_canonical_provider(
            provider_id="provider-123",
            name="Zeta Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            insurance_network_verification=VerificationStatus(
                status="verified",
                verified=True,
                basis="Plan directory confirmed.",
                source="Aetna directory",
            ),
        )
        provider_from_second_dataset = build_canonical_provider(
            provider_id="provider-123",
            name="Zeta Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            freshness=FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[provider_from_first_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=1),
                ),
                "npi_org": SourceSearchResult(
                    providers=[provider_from_second_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=1),
                    missing_location_hint="Add a city or ZIP code to narrow the search.",
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=cache,
            per_dataset_limit=3,
        )

        response = service.search(
            ProviderSearchRequest(specialties=("Primary Care",)),
            limit=1,
        )

        self.assertEqual(len(source.calls), 2)
        self.assertTrue(response.search_trace.cache_hit)
        self.assertEqual(len(response.provider_results), 1)
        provider = response.provider_results[0].provider
        self.assertTrue(provider.provider_id.startswith("source:clinicaltables:npi-idv:provider-123:"))
        self.assertEqual(provider.insurance_network_verification.status, "verified")
        self.assertEqual(
            provider.freshness,
            FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(
            response.missing_location_hint,
            "Add a city or ZIP code to narrow the search.",
        )
        self.assertEqual(cache.set_entries[-1].provider_ids, (provider.provider_id,))

    def test_search_degrades_when_cache_backend_fails(self) -> None:
        provider = build_canonical_provider(
            provider_id="provider-1",
            name="Harmony Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[provider],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=1),
                ),
                "npi_org": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=0),
                ),
            }
        )
        cache = FakeCache(
            get_error=RuntimeError("cache read failed"),
            set_error=RuntimeError("cache write failed"),
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=cache,
        )

        response = service.search(ProviderSearchRequest(specialties=("Primary Care",)), limit=1)

        self.assertEqual(len(response.provider_results), 1)
        self.assertFalse(response.search_trace.cache_hit)
        self.assertEqual(response.provider_results[0].provider.provider_id, provider.provider_id)
        self.assertEqual(len(cache.get_calls), 1)
        self.assertEqual(len(cache.set_entries), 1)

    def test_search_degrades_when_one_source_backend_fails(self) -> None:
        provider = build_canonical_provider(
            provider_id="provider-1",
            name="Harmony Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": RuntimeError("clinicaltables timeout"),
                "npi_org": SourceSearchResult(
                    providers=[provider],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=1),
                    missing_location_hint="Add a city or ZIP code to narrow the search.",
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
        )

        response = service.search(ProviderSearchRequest(specialties=("Primary Care",)), limit=1)

        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.provider_id, provider.provider_id)
        self.assertEqual(response.missing_location_hint, "Add a city or ZIP code to narrow the search.")
        self.assertEqual(response.search_trace.total_candidates, 1)
        self.assertEqual(response.search_trace.sources_attempted, ("clinicaltables:npi_idv", "clinicaltables:npi_org"))
        self.assertEqual(response.search_trace.source_traces[0].error, "clinicaltables timeout")
        self.assertEqual(response.search_trace.source_traces[1].result_count, 1)


if __name__ == "__main__":
    unittest.main()
