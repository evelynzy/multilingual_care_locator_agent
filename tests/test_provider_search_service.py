from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from provider_search.models import (
    FreshnessMetadata,
    ProviderSearchCacheEntry,
    ProviderSearchRequest,
    SourceSearchResult,
    SourceTrace,
    VerificationStatus,
)
from provider_search.normalization import build_canonical_provider, build_request_fingerprint
from provider_search.ranking import RANKING_VERSION, rank_provider_results
from provider_search.service import ProviderSearchService
from provider_search.sources.clinicaltables import ClinicalTablesSource


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


class RetryAwareClinicalTablesSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.search_terms == "Dermatology"
            and request.query_filter == "addr_practice.state:TX"
        ):
            provider = build_canonical_provider(
                provider_id="provider-location-retry",
                name="Dallas Family Clinic",
                source_name="ClinicalTables",
                dataset=dataset,
                city="Dallas",
                state="TX",
                taxonomy="Dermatology",
                specialties=("Dermatology",),
            )
            return SourceSearchResult(
                providers=[provider],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=1,
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class PediatricRetryClinicalTablesSource:
    def __init__(
        self,
        specialty_retry_providers: list,
        *,
        location_only_providers: list | None = None,
    ) -> None:
        self.specialty_retry_providers = list(specialty_retry_providers)
        self.location_only_providers = list(location_only_providers or [])
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.search_terms == "Pediatrics"
            and request.query_filter == "addr_practice.state:NY AND addr_practice.zip:10013*"
        ):
            return SourceSearchResult(
                providers=[],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=0,
                ),
            )
        if (
            request.search_terms == "Pediatrics"
            and request.query_filter == "addr_practice.state:NY"
        ):
            return SourceSearchResult(
                providers=list(self.specialty_retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.specialty_retry_providers),
                ),
            )
        if request.search_terms == "Manhattan, NY 10013":
            return SourceSearchResult(
                providers=list(self.location_only_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.location_only_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class NearbyDentalClinicalTablesSource:
    def __init__(
        self,
        local_zip_providers: list,
        nearby_state_providers: list,
    ) -> None:
        self.local_zip_providers = list(local_zip_providers)
        self.nearby_state_providers = list(nearby_state_providers)
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.query_filter == "addr_practice.zip:33012*"
            and request.search_terms in {"Dentistry", "Dentistry 33012"}
        ):
            return SourceSearchResult(
                providers=list(self.local_zip_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.local_zip_providers),
                ),
            )
        if (
            request.query_filter == "addr_practice.state:FL"
            and request.search_terms in {"Dentistry", "Dentistry FL"}
        ):
            return SourceSearchResult(
                providers=list(self.nearby_state_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.nearby_state_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class PrimaryCareSynonymClinicalTablesSource:
    def __init__(self, retry_providers: list) -> None:
        self.retry_providers = list(retry_providers)
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if request.query_filter == "addr_practice.state:TX" and request.search_terms in {
            "Primary Care",
            "Primary Care Dallas",
            "Primary Care TX 75001",
            "Primary Care 75001 Dallas TX",
        }:
            return SourceSearchResult(
                providers=list(self.retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.retry_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class PrimaryCareRetryClinicalTablesSource:
    def __init__(
        self,
        retry_providers: list,
        *,
        location_only_providers: list | None = None,
    ) -> None:
        self.retry_providers = list(retry_providers)
        self.location_only_providers = list(location_only_providers or [])
        self.calls: list[tuple[str, object]] = []

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.search_terms == "Primary Care"
            and request.query_filter == "addr_practice.state:TX AND addr_practice.zip:75001*"
        ):
            return SourceSearchResult(
                providers=[],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=0,
                ),
            )
        if (
            request.search_terms == "Primary Care"
            and request.query_filter == "addr_practice.state:TX"
        ):
            return SourceSearchResult(
                providers=list(self.retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.retry_providers),
                ),
            )
        if request.search_terms == "Dallas, TX 75001":
            return SourceSearchResult(
                providers=list(self.location_only_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.location_only_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class ObgynZipClinicalTablesSource:
    def __init__(
        self,
        noisy_zip_providers: list,
        canonical_term_providers: list,
    ) -> None:
        self.noisy_zip_providers = list(noisy_zip_providers)
        self.canonical_term_providers = list(canonical_term_providers)
        self.calls: list[tuple[str, object]] = []

    def suggest_specialty_terms(self, specialties: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(specialty.strip() for specialty in specialties if specialty.strip())

    def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if dataset == "npi_idv" and (
            request.query_filter == "addr_practice.zip:98101*"
            and request.search_terms == "OB/GYN"
        ):
            return SourceSearchResult(
                providers=list(self.noisy_zip_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.noisy_zip_providers),
                ),
            )
        if request.query_filter == "addr_practice.zip:98101*" and request.search_terms in {
            "Obstetrics & Gynecology",
            "Obstetrics & Gynecology 98101",
        }:
            return SourceSearchResult(
                providers=list(self.canonical_term_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.canonical_term_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


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
            ],
        )
        self.assertEqual(ranked[0].provider.ranking_metadata["ranking_version"], RANKING_VERSION)
        self.assertGreater(ranked[0].score, ranked[1].score)

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

    def test_rank_provider_results_matches_curated_primary_care_specialty_equivalents(self) -> None:
        request = ProviderSearchRequest(specialties=("Primary Care",))
        family_medicine_provider = build_canonical_provider(
            provider_id="provider-family-med",
            name="Harmony Family Medicine",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Physician/Family Practice",
            specialties=("Physician/Family Practice", "Family Medicine", "207Q00000X"),
        )
        internal_medicine_provider = build_canonical_provider(
            provider_id="provider-internal-med",
            name="Northside Internal Medicine",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Physician/Internal Medicine",
            specialties=("Physician/Internal Medicine", "Internal Medicine", "207R00000X"),
        )
        primary_care_clinic_provider = build_canonical_provider(
            provider_id="provider-primary-care-clinic",
            name="Concentra Primary Care",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Clinic/Center",
            specialties=("Clinic/Center", "Clinic/Center, Primary Care", "261QP2300X"),
        )

        ranked = rank_provider_results(
            request,
            [family_medicine_provider, internal_medicine_provider, primary_care_clinic_provider],
            limit=5,
        )

        self.assertEqual(len(ranked), 3)
        self.assertTrue(
            all(
                result.provider.ranking_metadata.get("matched_specialties") == ("Primary Care",)
                for result in ranked
            )
        )

    def test_rank_provider_results_matches_curated_dentistry_specialty_equivalents(self) -> None:
        request = ProviderSearchRequest(specialties=("Dentistry",), location="33012")
        dentist_provider = build_canonical_provider(
            provider_id="provider-dentist",
            name="Florida Children's Dentistry, P.A.",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Dentist",
            specialties=("Dentist", "Dentist, Pediatric Dentistry", "1223P0221X"),
        )
        periodontics_provider = build_canonical_provider(
            provider_id="provider-periodontics",
            name="Caplin and Gober Dentistry, PA",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Dentist",
            specialties=("Dentist", "Dentist, Periodontics", "1223P0300X"),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )

        ranked = rank_provider_results(
            request,
            [dentist_provider, periodontics_provider, radiology_provider],
            limit=5,
        )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            {result.provider.name for result in ranked},
            {"Florida Children's Dentistry, P.A.", "Caplin and Gober Dentistry, PA"},
        )
        self.assertTrue(
            all(
                result.provider.ranking_metadata.get("matched_specialties") == ("Dentistry",)
                for result in ranked
            )
        )
        self.assertTrue(
            all(result.provider.specialty_family_ids == ("dentistry",) for result in ranked)
        )

    def test_rank_provider_results_matches_curated_pediatrics_specialty_family_and_rejects_radiology(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Pediatrics",),
            location="Manhattan, NY 10013",
            keywords=("pediatric", "child health"),
        )
        pediatric_org_provider = build_canonical_provider(
            provider_id="provider-org-peds",
            name="Downtown Kids Clinic",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Clinic/Center",
            specialties=(
                "Clinic/Center",
                "Pediatrics",
                "208000000X",
                "Pediatric Gastroenterology",
                "2080P0206X",
            ),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )

        ranked = rank_provider_results(
            request,
            [pediatric_org_provider, radiology_provider],
            limit=5,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].provider.name, "Downtown Kids Clinic")
        self.assertEqual(ranked[0].provider.specialty_family_ids, ("pediatrics",))
        self.assertEqual(
            ranked[0].provider.ranking_metadata.get("matched_specialties"),
            ("Pediatrics",),
        )

    def test_rank_provider_results_prefers_direct_specialty_evidence_over_generic_family_match(
        self,
    ) -> None:
        request = ProviderSearchRequest(
            specialties=("Cardiology",),
            location="98101",
        )
        generic_family_match = build_canonical_provider(
            provider_id="provider-cardiology-generic",
            name="Apex Specialty Clinic",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Santa Clara",
            state="CA",
            taxonomy="Clinic/Center",
            specialties=("Clinic/Center",),
            specialty_family_ids=("cardiology",),
        )
        direct_specialty_match = build_canonical_provider(
            provider_id="provider-cardiology-direct",
            name="Zen Cardiology",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Cardiology",
            specialties=("Cardiology",),
        )

        ranked = rank_provider_results(
            request,
            [generic_family_match, direct_specialty_match],
            limit=5,
        )

        self.assertEqual(
            [result.provider.provider_id for result in ranked],
            [
                direct_specialty_match.provider_id,
                generic_family_match.provider_id,
            ],
        )
        self.assertEqual(
            ranked[0].provider.ranking_metadata.get("matched_specialties"),
            ("Cardiology",),
        )
        self.assertEqual(
            ranked[1].provider.ranking_metadata.get("matched_specialties"),
            ("Cardiology",),
        )
        self.assertEqual(
            ranked[0].provider.ranking_metadata["score_breakdown"]["specialty_specificity"],
            0.75,
        )
        self.assertEqual(
            ranked[1].provider.ranking_metadata["score_breakdown"]["specialty_specificity"],
            0.0,
        )

    def test_rank_provider_results_accepts_code_only_obgyn_structured_evidence(self) -> None:
        request = ProviderSearchRequest(
            specialties=("OB/GYN",),
            location="98101",
        )
        obgyn_provider = build_canonical_provider(
            provider_id="provider-obgyn-code-only",
            name="Cupertino Women's Health",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="207V00000X",
            specialties=("207V00000X",),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )

        ranked = rank_provider_results(
            request,
            [obgyn_provider, radiology_provider],
            limit=5,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].provider.name, "Cupertino Women's Health")
        self.assertEqual(
            ranked[0].provider.specialty_family_ids,
            ("obstetrics-gynecology",),
        )
        self.assertEqual(
            ranked[0].provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )

    def test_rank_provider_results_accepts_multi_candidate_obgyn_descendant_code_pool(self) -> None:
        request = ProviderSearchRequest(
            specialties=("OB/GYN",),
            location="98101",
        )
        gynecology_provider = build_canonical_provider(
            provider_id="provider-obgyn-gynecology",
            name="Cupertino Gynecology Group",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="207VC0200X",
            specialties=("207VC0200X",),
        )
        maternal_fetal_provider = build_canonical_provider(
            provider_id="provider-obgyn-mfm",
            name="South Bay Maternal Fetal Medicine",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="207VM0101X",
            specialties=("207VM0101X",),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )

        ranked = rank_provider_results(
            request,
            [gynecology_provider, maternal_fetal_provider, radiology_provider],
            limit=5,
        )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            {result.provider.name for result in ranked},
            {
                "Cupertino Gynecology Group",
                "South Bay Maternal Fetal Medicine",
            },
        )
        self.assertTrue(
            all(
                result.provider.specialty_family_ids == ("obstetrics-gynecology",)
                for result in ranked
            )
        )
        self.assertTrue(
            all(
                result.provider.ranking_metadata.get("matched_specialties") == ("OB/GYN",)
                for result in ranked
            )
        )

    def test_rank_provider_results_filters_location_only_candidates_for_constrained_searches(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Pediatrics",),
            location="Manhattan, NY 10013",
            keywords=("pediatric", "child health"),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )

        ranked = rank_provider_results(request, [radiology_provider], limit=3)

        self.assertEqual(ranked, [])

    def test_rank_provider_results_requires_specialty_evidence_when_specialty_requested(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Pediatrics",),
            location="Manhattan, NY 10013",
            keywords=("child health",),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        ).with_updates(description="Child health imaging and pediatric scans.")

        ranked = rank_provider_results(request, [radiology_provider], limit=3)

        self.assertEqual(ranked, [])

    def test_rank_provider_results_keeps_keyword_only_matches_when_no_specialty_requested(self) -> None:
        request = ProviderSearchRequest(
            location="Manhattan, NY 10013",
            keywords=("child health",),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        ).with_updates(description="Child health imaging and pediatric scans.")

        ranked = rank_provider_results(request, [radiology_provider], limit=3)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].provider.name, "Downtown Imaging Associates")
        self.assertEqual(
            ranked[0].provider.ranking_metadata.get("matched_keywords"),
            ("child health",),
        )
        self.assertEqual(
            ranked[0].provider.ranking_metadata.get("matched_specialties"),
            (),
        )


class ProviderSearchServiceTests(unittest.TestCase):
    def test_search_uses_keyword_terms_when_no_specialty_requested(self) -> None:
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        service.search(
            ProviderSearchRequest(
                location="Manhattan, NY 10013",
                keywords=("pediatric", "child health"),
            ),
            limit=3,
        )

        _, request = source.calls[0]
        self.assertEqual(request.search_terms, "pediatric child health")
        self.assertEqual(
            request.query_filter,
            "addr_practice.state:NY AND addr_practice.zip:10013*",
        )

    def test_search_zip_only_obgyn_98101_accepts_live_taxonomy_desc_payload_without_gate_changes(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(len(session.get.call_args_list), 1)
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
        self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
        self.assertEqual(len(response_payload.provider_results), 1)
        provider = response_payload.provider_results[0].provider
        self.assertEqual(provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(provider.taxonomy, "Obstetrics & Gynecology")
        self.assertEqual(provider.specialty_family_ids, ("obstetrics-gynecology",))
        self.assertEqual(
            provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )

    def test_search_zip_only_obgyn_98101_accepts_live_v3_rows_payload_without_broad_retry(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["1619271780"],
            None,
            [[
                "Cupertino OB/GYN Associates",
                "",
                "",
                "",
                "",
                "",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "",
                "",
                "",
                "Santa Clara",
                "CA",
                "98101",
                "",
                "408-555-0100",
                ["English"],
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(len(session.get.call_args_list), 1)
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
        self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
        self.assertEqual(len(response_payload.provider_results), 1)
        self.assertEqual(response_payload.search_trace.total_candidates, 1)
        provider = response_payload.provider_results[0].provider
        self.assertEqual(provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(provider.provider_id, "1619271780")
        self.assertEqual(provider.taxonomy, "Obstetrics & Gynecology")
        self.assertEqual(provider.specialty_family_ids, ("obstetrics-gynecology",))

    def test_search_zip_only_obgyn_98101_admits_generic_provider_type_when_taxonomy_evidence_is_specific(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "Clinic/Center",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(len(response_payload.provider_results), 1)
        result = response_payload.provider_results[0]
        self.assertEqual(result.provider.taxonomy, "Obstetrics & Gynecology")
        self.assertEqual(
            result.provider.specialties,
            ("Clinic/Center", "Obstetrics & Gynecology", "207V00000X"),
        )
        self.assertEqual(result.provider.specialty_family_ids, ("obstetrics-gynecology",))
        self.assertEqual(
            result.provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )

    def test_search_zip_only_obgyn_98101_admits_physician_prefixed_live_taxonomy_variant(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "Clinic/Center",
                "Physician/Obstetrics & Gynecology",
                "",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(len(response_payload.provider_results), 1)
        result = response_payload.provider_results[0]
        self.assertEqual(result.provider.taxonomy, "Physician/Obstetrics & Gynecology")
        self.assertEqual(
            result.provider.specialties,
            ("Clinic/Center", "Physician/Obstetrics & Gynecology"),
        )
        self.assertEqual(result.provider.specialty_family_ids, ("obstetrics-gynecology",))
        self.assertEqual(
            result.provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )

    def test_search_zip_only_obgyn_98101_prefers_direct_specialty_evidence_over_generic_family_match(
        self,
    ) -> None:
        generic_family_match = build_canonical_provider(
            provider_id="provider-obgyn-generic",
            name="Apex Specialty Clinic",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Santa Clara",
            state="CA",
            taxonomy="Clinic/Center",
            specialties=("Clinic/Center",),
            specialty_family_ids=("obstetrics-gynecology",),
        )
        direct_specialty_match = build_canonical_provider(
            provider_id="provider-obgyn-direct",
            name="Zen Women's Health",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Physician/Obstetrics & Gynecology",
            specialties=("Physician/Obstetrics & Gynecology",),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[generic_family_match, direct_specialty_match],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        result_count=2,
                    ),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(
            [result.provider.provider_id for result in response_payload.provider_results],
            [
                direct_specialty_match.provider_id,
                generic_family_match.provider_id,
            ],
        )
        self.assertEqual(
            response_payload.provider_results[0].provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )
        self.assertEqual(
            response_payload.provider_results[1].provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )
        self.assertEqual(
            response_payload.provider_results[0].provider.ranking_metadata["score_breakdown"][
                "specialty_specificity"
            ],
            0.75,
        )
        self.assertEqual(
            response_payload.provider_results[1].provider.ranking_metadata["score_breakdown"][
                "specialty_specificity"
            ],
            0.0,
        )

    def test_search_cardiology_98101_admits_live_cardiology_taxonomy_variant(
        self,
    ) -> None:
        live_cardiology_provider = build_canonical_provider(
            provider_id="provider-cardiology-live",
            name="Zen Cardiology",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Physician/Internal Medicine, Cardiovascular Disease",
            specialties=(
                "Clinic/Center",
                "Physician/Internal Medicine, Cardiovascular Disease",
                "207RC0000X",
            ),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[live_cardiology_provider],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        result_count=1,
                    ),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        response_payload = service.search(
            ProviderSearchRequest(
                specialties=("Cardiology",),
                location="98101",
            ),
            limit=5,
        )

        self.assertEqual(len(response_payload.provider_results), 1)
        result = response_payload.provider_results[0]
        self.assertEqual(result.provider.name, "Zen Cardiology")
        self.assertEqual(result.provider.specialty_family_ids, ("cardiology",))
        self.assertEqual(
            result.provider.ranking_metadata.get("matched_specialties"),
            ("Cardiology",),
        )

    def test_search_emits_scoped_clinicaltables_request_log_when_fingerprint_matches(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        request = ProviderSearchRequest(
            specialties=("OB/GYN",),
            location="98101",
        )
        request_fingerprint = build_request_fingerprint(request)

        with patch.dict(
            "os.environ",
            {
                "PROVIDER_SEARCH_DEBUG": "1",
                "CARE_LOCATOR_LOCAL_DEBUG": "1",
                "PROVIDER_SEARCH_DEBUG_FINGERPRINT": request_fingerprint,
            },
            clear=False,
        ):
            with self.assertLogs(level="INFO") as captured:
                service.search(request, limit=5)

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug_request", joined_logs)
        self.assertIn(f"request_fingerprint={request_fingerprint}", joined_logs)
        self.assertIn("dataset=npi_idv", joined_logs)
        self.assertIn("terms=obstetrics gynecology", joined_logs)
        self.assertIn("q=addr_practice.zip:98101*", joined_logs)
        self.assertIn(
            "sf=provider_type,licenses.medicare.type,licenses.taxonomy.classification,licenses.taxonomy.specialization,licenses.taxonomy.code",
            joined_logs,
        )

    def test_search_does_not_emit_scoped_clinicaltables_request_log_when_fingerprint_does_not_match(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        with patch.dict(
            "os.environ",
            {
                "PROVIDER_SEARCH_DEBUG": "1",
                "CARE_LOCATOR_LOCAL_DEBUG": "1",
                "PROVIDER_SEARCH_DEBUG_FINGERPRINT": "not-the-request-fingerprint",
            },
            clear=False,
        ):
            with self.assertLogs(level="INFO") as captured:
                service.search(
                    ProviderSearchRequest(
                        specialties=("OB/GYN",),
                        location="98101",
                    ),
                    limit=5,
                )

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug request_fingerprint=", joined_logs)
        self.assertNotIn("provider_search_debug_request", joined_logs)

    def test_search_does_not_emit_scoped_clinicaltables_request_log_without_selector(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        with patch.dict(
            "os.environ",
            {
                "PROVIDER_SEARCH_DEBUG": "1",
                "CARE_LOCATOR_LOCAL_DEBUG": "1",
            },
            clear=False,
        ):
            with self.assertLogs(level="INFO") as captured:
                service.search(
                    ProviderSearchRequest(
                        specialties=("OB/GYN",),
                        location="98101",
                    ),
                    limit=5,
                )

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug request_fingerprint=", joined_logs)
        self.assertNotIn("provider_search_debug_request", joined_logs)

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
                location="Pittsburgh, PA 80202",
                insurance=("Aetna",),
                preferred_languages=("Spanish",),
                keywords=("same day",),
            ),
            limit=2,
        )

        self.assertEqual(len(source.calls), 4)
        first_dataset, first_request = source.calls[0]
        self.assertEqual(first_dataset, "npi_idv")
        self.assertEqual(first_request.search_terms, "Primary Care")
        self.assertEqual(first_request.city_hint, "Pittsburgh")
        self.assertEqual(first_request.state_hint, "PA")
        self.assertEqual(first_request.zip_hint, "80202")
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:PA AND addr_practice.zip:80202*",
        )
        self.assertEqual(first_request.limit, 4)
        third_dataset, third_request = source.calls[2]
        self.assertEqual(third_dataset, "npi_idv")
        self.assertEqual(third_request.search_terms, "Primary Care Pittsburgh PA 80202")
        self.assertEqual(third_request.query_filter, first_request.query_filter)

        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Harmony Family Clinic")
        self.assertFalse(response.search_trace.cache_hit)
        self.assertEqual(
            response.search_trace.sources_attempted,
            (
                "clinicaltables:npi_idv",
                "clinicaltables:npi_org",
                "clinicaltables:npi_idv",
                "clinicaltables:npi_org",
            ),
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

    def test_search_applies_secondary_display_dedupe_for_duplicate_primary_care_org_results(self) -> None:
        cache = FakeCache()
        provider_from_individual_dataset = build_canonical_provider(
            provider_id="1111111111",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        provider_from_org_dataset = build_canonical_provider(
            provider_id="2222222222",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
            freshness=FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=111,
                last_updated_epoch=222,
            ),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[provider_from_individual_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=1),
                ),
                "npi_org": SourceSearchResult(
                    providers=[provider_from_org_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=1),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=cache,
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=5,
        )

        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.search_trace.total_candidates, 2)
        representative = response.provider_results[0]
        self.assertEqual(
            representative.provider.freshness,
            FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=111,
                last_updated_epoch=222,
            ),
        )
        self.assertCountEqual(
            representative.retriever_metadata["display_dedupe_provider_ids"],
            ["1111111111", "2222222222"],
        )
        self.assertEqual(representative.retriever_metadata["display_dedupe_count"], 2)
        self.assertCountEqual(
            cache.set_entries[-1].provider_ids,
            ("1111111111", "2222222222"),
        )

    def test_search_keeps_same_site_org_results_separate_when_service_lines_differ(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="1111111111",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        pediatrics_provider = build_canonical_provider(
            provider_id="2222222222",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Pediatrics",
            specialties=("Pediatrics",),
            phone="214-555-0100",
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[primary_care_provider],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=1),
                ),
                "npi_org": SourceSearchResult(
                    providers=[pediatrics_provider],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=1),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                location="Dallas, TX 75001",
            ),
            limit=5,
        )

        self.assertEqual(len(response.provider_results), 2)
        self.assertEqual(response.search_trace.total_candidates, 2)
        self.assertCountEqual(
            [result.provider.taxonomy for result in response.provider_results],
            ["Primary Care", "Pediatrics"],
        )
        self.assertTrue(
            all(
                "display_dedupe_count" not in result.retriever_metadata
                for result in response.provider_results
            )
        )

    def test_search_logs_opt_in_debug_counts_for_single_result_analysis(self) -> None:
        provider_from_individual_dataset = build_canonical_provider(
            provider_id="1111111111",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        provider_from_org_dataset = build_canonical_provider(
            provider_id="2222222222",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[provider_from_individual_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=1),
                ),
                "npi_org": SourceSearchResult(
                    providers=[provider_from_org_dataset],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=1),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        with patch.dict(
            "os.environ",
            {"PROVIDER_SEARCH_DEBUG": "1", "CARE_LOCATOR_LOCAL_DEBUG": "1"},
        ):
            with self.assertLogs("provider_search.service", level="INFO") as captured:
                service.search(
                    ProviderSearchRequest(
                        specialties=("Primary Care",),
                        location="Dallas, TX 75001",
                    ),
                    limit=5,
                )

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug_plan", joined_logs)
        self.assertIn("provider_search_debug request_fingerprint=", joined_logs)
        self.assertIn("raw_candidates=2", joined_logs)
        self.assertIn("post_gate=2", joined_logs)
        self.assertIn("post_display_dedupe=1", joined_logs)
        self.assertIn("provider_search_debug_variant", joined_logs)
        self.assertIn("provider_search_debug_candidate", joined_logs)
        self.assertIn("provider_search_debug_source dataset=npi_idv result_count=1 error=none", joined_logs)
        self.assertIn("provider_search_debug_source dataset=npi_org result_count=1 error=none", joined_logs)
        self.assertIn("provider_search_debug_display", joined_logs)
        self.assertIn("display_dedupe_count=2", joined_logs)
        self.assertIn("display_dedupe_provider_keys=", joined_logs)
        self.assertNotIn("provider_search_debug_candidate_detail", joined_logs)
        self.assertNotIn("1111111111", joined_logs)
        self.assertNotIn("2222222222", joined_logs)

    def test_search_emits_scoped_candidate_dump_when_fingerprint_matches(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Primary Care",),
            location="Dallas, TX 75001",
        )
        admitted_provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        dropped_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[admitted_provider, dropped_provider],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=2),
                ),
                "npi_org": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )
        request_fingerprint = build_request_fingerprint(request)

        with patch.dict(
            "os.environ",
            {
                "PROVIDER_SEARCH_DEBUG": "1",
                "CARE_LOCATOR_LOCAL_DEBUG": "1",
                "PROVIDER_SEARCH_DEBUG_FINGERPRINT": request_fingerprint,
            },
        ):
            with self.assertLogs("provider_search.service", level="INFO") as captured:
                service.search(request, limit=5)

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug_candidate_detail", joined_logs)
        self.assertIn("gate_outcome=admitted", joined_logs)
        self.assertIn("gate_outcome=dropped", joined_logs)
        self.assertIn("drop_reason=specialty_mismatch", joined_logs)
        self.assertNotIn("Dallas Family Clinic", joined_logs)
        self.assertNotIn("Downtown Imaging Associates", joined_logs)
        self.assertNotIn("75001", joined_logs)

    def test_search_does_not_emit_scoped_candidate_dump_when_fingerprint_does_not_match(self) -> None:
        request = ProviderSearchRequest(
            specialties=("Primary Care",),
            location="Dallas, TX 75001",
        )
        provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
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
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        with patch.dict(
            "os.environ",
            {
                "PROVIDER_SEARCH_DEBUG": "1",
                "CARE_LOCATOR_LOCAL_DEBUG": "1",
                "PROVIDER_SEARCH_DEBUG_FINGERPRINT": "not-the-request-fingerprint",
            },
        ):
            with self.assertLogs("provider_search.service", level="INFO") as captured:
                service.search(request, limit=5)

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug_plan", joined_logs)
        self.assertNotIn("provider_search_debug_candidate_detail", joined_logs)

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

    def test_search_applies_city_state_query_filter_for_location_specific_searches(self) -> None:
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                ),
                "npi_org": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                location="Pittsburgh, PA",
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 2)
        for dataset, request in source.calls:
            self.assertEqual(dataset in {"npi_idv", "npi_org"}, True)
            self.assertEqual(request.search_terms, "Pittsburgh, PA")
            self.assertEqual(request.city_hint, "Pittsburgh")
            self.assertEqual(request.state_hint, "PA")
            self.assertIsNone(request.zip_hint)
            self.assertEqual(
                request.query_filter,
                'addr_practice.state:PA AND addr_practice.city:"Pittsburgh"',
            )
        self.assertEqual(len(response.provider_results), 0)
        self.assertEqual(response.search_trace.total_candidates, 0)

    def test_search_builds_non_empty_query_filter_for_dallas_zip_search(self) -> None:
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=3,
        )

        _, request = source.calls[0]
        self.assertEqual(
            request.query_filter,
            "addr_practice.state:TX AND addr_practice.zip:75001*",
        )

    def test_search_zip_only_obgyn_98101_continues_to_canonical_variant_after_single_noisy_idv_hit(self) -> None:
        noisy_zip_providers = [
            build_canonical_provider(
                provider_id="provider-noise-0",
                name="Noisy Clinician 0",
                source_name="ClinicalTables",
                dataset="npi_idv",
                city="Santa Clara",
                state="CA",
                specialties=(),
                taxonomy=None,
            )
        ]
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
        )
        source = ObgynZipClinicalTablesSource(
            noisy_zip_providers,
            [specialty_bearing_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology"],
        )
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(
            response.provider_results[0].provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )
        self.assertEqual(response.search_trace.total_candidates, 2)

    def test_search_zip_only_obgyn_98101_uses_canonical_zip_variant_when_non_location_variant_stays_low_signal(
        self,
    ) -> None:
        noisy_zip_providers = [
            build_canonical_provider(
                provider_id="provider-noise-0",
                name="Noisy Clinician 0",
                source_name="ClinicalTables",
                dataset="npi_idv",
                city="Santa Clara",
                state="CA",
                specialties=(),
                taxonomy=None,
            )
        ]
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
        )

        class CanonicalZipOnlyObgynSource(ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset == "npi_idv" and (
                    request.query_filter == "addr_practice.zip:98101*"
                    and request.search_terms in {"OB/GYN", "Obstetrics & Gynecology"}
                ):
                    return SourceSearchResult(
                        providers=list(self.noisy_zip_providers),
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=len(self.noisy_zip_providers),
                        ),
                    )
                if request.query_filter == "addr_practice.zip:98101*" and (
                    request.search_terms == "Obstetrics & Gynecology 98101"
                ):
                    return SourceSearchResult(
                        providers=list(self.canonical_term_providers),
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=len(self.canonical_term_providers),
                        ),
                    )
                return SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset=dataset,
                        status_code=200,
                        result_count=0,
                    ),
                )

        source = CanonicalZipOnlyObgynSource(
            noisy_zip_providers,
            [specialty_bearing_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology", "Obstetrics & Gynecology 98101"],
        )
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(response.search_trace.total_candidates, 2)

    def test_search_zip_only_obgyn_98101_uses_demo_broad_recall_variant_after_precise_path(
        self,
    ) -> None:
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
            raw={"addr_practice.zip": "98101"},
        )

        class DemoBroadRecallObgynSource(ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset != "npi_idv":
                    return SourceSearchResult(
                        providers=[],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=0,
                        ),
                    )
                if request.search_terms in {
                    "OB/GYN",
                    "Obstetrics & Gynecology",
                    "Obstetrics & Gynecology 98101",
                }:
                    return SourceSearchResult(
                        providers=[],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=0,
                        ),
                    )
                if request.search_terms == "ob gyn 98101":
                    return SourceSearchResult(
                        providers=[specialty_bearing_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
                        ),
                    )
                raise AssertionError(f"Unexpected search request: {request.search_terms!r}")

        source = DemoBroadRecallObgynSource([], [])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology", "Obstetrics & Gynecology 98101", "ob gyn 98101"],
        )
        broad_request = source.calls[-1][1]
        self.assertFalse(broad_request.specialty_driven)
        self.assertIsNone(broad_request.query_filter)
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(
            response.provider_results[0].provider.ranking_metadata.get("matched_specialties"),
            ("OB/GYN",),
        )

    def test_search_zip_only_obgyn_98101_ignores_out_of_area_broad_recall_hits(
        self,
    ) -> None:
        out_of_area_provider = build_canonical_provider(
            provider_id="provider-obgyn-sf",
            name="San Francisco OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="San Francisco",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
            raw={"addr_practice.zip": "94105"},
        )

        class OutOfAreaBroadRecallObgynSource(ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset != "npi_idv":
                    return SourceSearchResult(
                        providers=[],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=0,
                        ),
                    )
                if request.search_terms in {
                    "OB/GYN",
                    "Obstetrics & Gynecology",
                    "Obstetrics & Gynecology 98101",
                }:
                    return SourceSearchResult(
                        providers=[],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=0,
                        ),
                    )
                if request.search_terms == "ob gyn 98101":
                    return SourceSearchResult(
                        providers=[out_of_area_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
                        ),
                    )
                raise AssertionError(f"Unexpected search request: {request.search_terms!r}")

        source = OutOfAreaBroadRecallObgynSource([], [])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms[:4],
            ["OB/GYN", "Obstetrics & Gynecology", "Obstetrics & Gynecology 98101", "ob gyn 98101"],
        )
        self.assertGreaterEqual(len(searched_terms), 4)
        self.assertEqual(response.provider_results, ())
        self.assertEqual(response.search_trace.total_candidates, 0)

    def test_search_zip_only_obgyn_98101_does_not_stop_on_unrelated_specialty_bearing_first_hit(
        self,
    ) -> None:
        unrelated_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Santa Clara Imaging Group",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
        )

        class UnrelatedSpecialtyFirstHitObgynSource(ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: str, request: object) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset == "npi_idv" and (
                    request.query_filter == "addr_practice.zip:98101*"
                    and request.search_terms == "OB/GYN"
                ):
                    return SourceSearchResult(
                        providers=[unrelated_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
                        ),
                    )
                if request.query_filter == "addr_practice.zip:98101*" and (
                    request.search_terms == "Obstetrics & Gynecology"
                ):
                    return SourceSearchResult(
                        providers=[specialty_bearing_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
                        ),
                    )
                return SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(
                        source_name="clinicaltables",
                        dataset=dataset,
                        status_code=200,
                        result_count=0,
                    ),
                )

        source = UnrelatedSpecialtyFirstHitObgynSource([], [])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            limit=5,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology"],
        )
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(response.search_trace.total_candidates, 2)

    def test_search_keeps_source_friendly_ent_specialty_terms_without_family_label_rewrite(self) -> None:
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        service.search(
            ProviderSearchRequest(
                specialties=("ENT",),
                location="Austin, TX 78701",
            ),
            limit=3,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertIn("ENT", searched_terms)
        self.assertIn("ENT Austin TX 78701", searched_terms)
        self.assertNotIn("ENT / Otolaryngology", searched_terms)
        self.assertNotIn("ENT / Otolaryngology Austin TX 78701", searched_terms)

    def test_search_keeps_source_friendly_psychiatry_specialty_terms_without_family_label_rewrite(self) -> None:
        source = FakeClinicalTablesSource(
            {
                "npi_idv": SourceSearchResult(
                    providers=[],
                    trace=SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                ),
            }
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        service.search(
            ProviderSearchRequest(
                specialties=("Psychiatry",),
                location="San Jose, CA 95112",
            ),
            limit=3,
        )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertIn("Psychiatry", searched_terms)
        self.assertIn("Psychiatry San Jose CA 95112", searched_terms)
        self.assertNotIn("Psychiatry / Behavioral Health", searched_terms)
        self.assertNotIn("Psychiatry / Behavioral Health San Jose CA 95112", searched_terms)

    def test_search_retries_with_relaxed_location_filter_when_specialty_search_survives_no_candidates(self) -> None:
        source = RetryAwareClinicalTablesSource()
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Dermatology",),
                location="Dallas, TX 75001",
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        _, relaxed_assisted_request = source.calls[3]
        self.assertEqual(first_request.search_terms, "Dermatology")
        self.assertEqual(second_request.search_terms, "Dermatology Dallas TX 75001")
        self.assertEqual(retry_request.search_terms, "Dermatology")
        self.assertEqual(relaxed_assisted_request.search_terms, "Dermatology Dallas TX 75001")
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:TX AND addr_practice.zip:75001*",
        )
        self.assertEqual(second_request.query_filter, first_request.query_filter)
        self.assertEqual(retry_request.query_filter, "addr_practice.state:TX")
        self.assertEqual(relaxed_assisted_request.query_filter, retry_request.query_filter)
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Dallas Family Clinic")

    def test_search_primary_care_75001_uses_relaxed_specialty_retry_before_location_only(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        location_only_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        source = PrimaryCareRetryClinicalTablesSource(
            [primary_care_provider],
            location_only_providers=[location_only_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        self.assertEqual(first_request.search_terms, "Primary Care")
        self.assertEqual(second_request.search_terms, "Primary Care Dallas TX 75001")
        self.assertEqual(retry_request.search_terms, "Primary Care")
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:TX AND addr_practice.zip:75001*",
        )
        self.assertEqual(retry_request.query_filter, "addr_practice.state:TX")
        self.assertNotIn("Dallas, TX 75001", [request.search_terms for _, request in source.calls])
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Dallas Family Clinic")

    def test_search_primary_care_75001_keeps_only_primary_care_choice_when_retry_returns_mixed_candidates(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
        )
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        source = PrimaryCareRetryClinicalTablesSource(
            [primary_care_provider, radiology_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Dallas Family Clinic")
        self.assertEqual(
            response.provider_results[0].provider.ranking_metadata.get("matched_specialties"),
            ("Primary Care",),
        )
        self.assertEqual(
            response.provider_results[0].provider.specialty_family_ids,
            ("primary-care",),
        )

    def test_search_primary_care_75001_keeps_second_visible_result_after_display_dedupe(self) -> None:
        duplicate_primary_care_individual = build_canonical_provider(
            provider_id="provider-primary-care-individual",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        duplicate_primary_care_org = build_canonical_provider(
            provider_id="provider-primary-care-org",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        second_visible_provider = build_canonical_provider(
            provider_id="provider-second-visible",
            name="Zzz Addison Primary Care",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="456 Belt Line Rd",
            city="Addison",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="972-555-0199",
        )
        source = PrimaryCareRetryClinicalTablesSource(
            [
                duplicate_primary_care_individual,
                duplicate_primary_care_org,
                second_visible_provider,
            ]
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv", "npi_org"),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=2,
        )

        self.assertEqual(len(source.calls), 8)
        self.assertEqual(len(response.provider_results), 2)
        self.assertEqual(
            [result.provider.name for result in response.provider_results],
            ["Dallas Family Clinic", "Zzz Addison Primary Care"],
        )
        self.assertEqual(
            response.provider_results[0].retriever_metadata["display_dedupe_count"],
            2,
        )

    def test_search_retries_pediatric_request_with_specialty_terms_instead_of_location_only(self) -> None:
        pediatric_provider = build_canonical_provider(
            provider_id="provider-peds",
            name="Canal Pediatrics",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Pediatrics",
            specialties=("Pediatrics",),
        ).with_updates(description="Pediatric and child health visits.")
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        location_only_providers = [
            radiology_provider.with_updates(provider_id=f"provider-radiology-{index}")
            for index in range(15)
        ]
        source = PediatricRetryClinicalTablesSource(
            [pediatric_provider],
            location_only_providers=location_only_providers,
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Pediatrics",),
                location="Manhattan, NY 10013",
                keywords=("pediatric", "child health"),
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        self.assertEqual(
            first_request.search_terms,
            "Pediatrics",
        )
        self.assertEqual(second_request.search_terms, "Pediatrics Manhattan NY 10013")
        self.assertEqual(retry_request.search_terms, first_request.search_terms)
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:NY AND addr_practice.zip:10013*",
        )
        self.assertEqual(retry_request.query_filter, "addr_practice.state:NY")
        self.assertEqual(len(response.provider_results), 1)
        self.assertEqual(response.provider_results[0].provider.name, "Canal Pediatrics")
        self.assertEqual(response.search_trace.total_candidates, 1)

    def test_search_keeps_only_relevant_pediatric_result_when_retry_returns_mixed_candidates(self) -> None:
        pediatric_provider = build_canonical_provider(
            provider_id="provider-peds",
            name="Canal Pediatrics",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Pediatrics",
            specialties=("Pediatrics",),
        ).with_updates(description="Pediatric and child health visits.")
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        source = PediatricRetryClinicalTablesSource(
            [pediatric_provider, radiology_provider]
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Pediatrics",),
                location="Manhattan, NY 10013",
                keywords=("pediatric", "child health"),
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        self.assertEqual(first_request.search_terms, "Pediatrics")
        self.assertEqual(second_request.search_terms, "Pediatrics Manhattan NY 10013")
        self.assertEqual(retry_request.search_terms, first_request.search_terms)
        self.assertEqual(retry_request.query_filter, "addr_practice.state:NY")
        self.assertEqual(len(response.provider_results), 1)
        result = response.provider_results[0]
        self.assertEqual(result.provider.name, "Canal Pediatrics")
        self.assertEqual(
            result.provider.ranking_metadata.get("matched_specialties"),
            ("Pediatrics",),
        )

    def test_search_keeps_live_style_org_candidate_when_nppes_specialties_include_pediatrics(self) -> None:
        pediatric_org_provider = build_canonical_provider(
            provider_id="provider-org-peds",
            name="Downtown Kids Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Clinic/Center",
            specialties=(
                "Clinic/Center",
                "Pediatrics",
                "208000000X",
                "Pediatric Gastroenterology",
                "2080P0206X",
            ),
        )
        source = PediatricRetryClinicalTablesSource([pediatric_org_provider])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Pediatrics",),
                location="Manhattan, NY 10013",
                keywords=("pediatric", "child health"),
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 4)
        self.assertEqual(len(response.provider_results), 1)
        result = response.provider_results[0]
        self.assertEqual(result.provider.name, "Downtown Kids Clinic")
        self.assertEqual(result.provider.taxonomy, "Clinic/Center")
        self.assertEqual(
            result.provider.ranking_metadata.get("matched_specialties"),
            ("Pediatrics",),
        )
        self.assertIn("208000000X", result.provider.specialties)
        self.assertEqual(result.provider.specialty_family_ids, ("pediatrics",))

    def test_search_admits_zip_driven_dentistry_descendants_without_nearby_retry(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-1",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry", "1223P0221X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-2",
                name="Hialeah Square Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, General Practice", "1223G0001X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-3",
                name="Caplin and Gober Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Periodontics", "1223P0300X"),
            ),
        ]
        nearby_state_provider = build_canonical_provider(
            provider_id="provider-nearby-1",
            name="Miami Lakes Dentistry Center",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Miami Lakes",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
        )
        source = NearbyDentalClinicalTablesSource(local_zip_providers, [nearby_state_provider])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Dentistry",),
                location="33012",
            ),
            limit=3,
        )

        self.assertEqual(len(source.calls), 1)
        _, first_request = source.calls[0]
        self.assertEqual(first_request.search_terms, "Dentistry")
        self.assertEqual(first_request.query_filter, "addr_practice.zip:33012*")
        self.assertEqual(len(response.provider_results), 3)
        self.assertEqual(response.search_trace.total_candidates, 3)
        self.assertCountEqual(
            [result.provider.name for result in response.provider_results],
            [
                "Florida Children's Dentistry, P.A.",
                "Hialeah Square Dentistry, PA",
                "Caplin and Gober Dentistry, PA",
            ],
        )
        self.assertTrue(
            all(
                result.provider.ranking_metadata.get("matched_specialties") == ("Dentistry",)
                for result in response.provider_results
            )
        )
        self.assertTrue(
            all(result.provider.specialty_family_ids == ("dentistry",) for result in response.provider_results)
        )

    def test_search_dentista_33012_keeps_second_visible_result_after_local_display_dedupe(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-individual",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (individual)",
                dataset="npi_idv",
                address="123 Palm Ave",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry"),
                phone="305-555-0101",
            ),
            build_canonical_provider(
                provider_id="provider-local-org",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                address="123 Palm Ave",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry"),
                phone="305-555-0101",
            ),
            build_canonical_provider(
                provider_id="provider-local-second",
                name="Zzz Family Dental",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                address="900 Pine St",
                city="Miami",
                state="FL",
                taxonomy="Dentistry",
                specialties=("Dentistry",),
                phone="305-555-0102",
            ),
        ]
        source = NearbyDentalClinicalTablesSource(
            local_zip_providers,
            [],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv", "npi_org"),
            per_dataset_limit=5,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Dentistry",),
                location="33012",
            ),
            limit=2,
        )

        self.assertEqual(len(source.calls), 2)
        self.assertEqual(len(response.provider_results), 2)
        self.assertEqual(
            [result.provider.name for result in response.provider_results],
            ["Florida Children's Dentistry, P.A.", "Zzz Family Dental"],
        )
        self.assertEqual(
            response.provider_results[0].retriever_metadata["display_dedupe_count"],
            2,
        )

    def test_search_keeps_live_primary_care_org_candidates_when_canonical_specialties_use_curated_synonyms(self) -> None:
        retry_providers = [
            build_canonical_provider(
                provider_id="provider-clinic-primary-care",
                name="Concentra Primary Care PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Dallas",
                state="TX",
                taxonomy="Clinic/Center",
                specialties=("Clinic/Center", "Clinic/Center, Primary Care", "261QP2300X"),
            ),
            build_canonical_provider(
                provider_id="provider-family-med",
                name="North Dallas Primary Care Doctors PLLC",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Dallas",
                state="TX",
                taxonomy="Physician/Internal Medicine",
                specialties=("Physician/Internal Medicine", "Family Medicine", "Internal Medicine"),
            ),
        ]
        source = PrimaryCareSynonymClinicalTablesSource(retry_providers)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=10,
        )

        response = service.search(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX 75001",
            ),
            limit=5,
        )

        self.assertEqual(len(response.provider_results), 2)
        self.assertCountEqual(
            [result.provider.name for result in response.provider_results],
            ["Concentra Primary Care PA", "North Dallas Primary Care Doctors PLLC"],
        )
        self.assertTrue(
            all(
                result.provider.ranking_metadata.get("matched_specialties") == ("Primary Care",)
                for result in response.provider_results
            )
        )


if __name__ == "__main__":
    unittest.main()
