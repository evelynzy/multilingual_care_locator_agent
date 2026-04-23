import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from provider_search.cache import (
    DEFAULT_PROVIDER_CACHE_FILENAME,
    DEFAULT_PROVIDER_CACHE_SUBDIR,
    SQLiteProviderSearchCache,
    resolve_provider_cache_path,
)
from provider_search.models import (
    CanonicalProvider,
    FreshnessMetadata,
    MedicareOptOutStatus,
    ProviderSearchCacheEntry,
    ProviderSearchRequest,
    VerificationStatus,
)
from provider_search.normalization import (
    build_canonical_provider,
    build_request_fingerprint,
    normalize_provider,
    normalize_search_request,
    normalize_search_result,
)


class ProviderSearchNormalizationTests(unittest.TestCase):
    def test_normalize_search_request_collapses_whitespace_and_dedupes_lists(self) -> None:
        request = ProviderSearchRequest(
            specialties=(" Primary Care ", "primary care", "Pediatrics"),
            location="  San   Francisco,   CA ",
            insurance=(" Aetna ", "", "aetna"),
            preferred_languages=(" English ", "english", "Spanish"),
            keywords=(" same day ", "Same Day", "nearby"),
        )

        normalized = normalize_search_request(request)

        self.assertEqual(normalized.specialties, ("Primary Care", "Pediatrics"))
        self.assertEqual(normalized.location, "San Francisco, CA")
        self.assertEqual(normalized.insurance, ("Aetna",))
        self.assertEqual(normalized.preferred_languages, ("English", "Spanish"))
        self.assertEqual(normalized.keywords, ("same day", "nearby"))

    def test_build_request_fingerprint_is_stable_across_case_and_spacing(self) -> None:
        left = ProviderSearchRequest(
            specialties=("Primary Care",),
            location="San Francisco, CA",
            insurance=("Aetna",),
            preferred_languages=("English",),
            keywords=("same day",),
        )
        right = ProviderSearchRequest(
            specialties=(" primary   care ",),
            location=" san francisco, ca ",
            insurance=(" aetna ",),
            preferred_languages=(" english ",),
            keywords=(" Same Day ",),
        )

        self.assertEqual(build_request_fingerprint(left), build_request_fingerprint(right))

    def test_build_canonical_provider_generates_stable_source_aware_id_when_missing(self) -> None:
        left = build_canonical_provider(
            provider_id="  ",
            name="Harmony Family Clinic",
            source_name="ClinicalTables",
            dataset="npi_idv",
            address="123 Main St",
            city="Pittsburgh",
            state="PA",
            taxonomy="Family Medicine",
        )
        right = build_canonical_provider(
            provider_id=None,
            name=" Harmony Family Clinic ",
            source_name="ClinicalTables",
            dataset="npi_idv",
            address=" 123 Main St ",
            city=" Pittsburgh ",
            state="PA",
            taxonomy="Family Medicine",
        )

        self.assertTrue(left.provider_id.startswith("generated:clinicaltables:npi-idv:"))
        self.assertEqual(left.provider_id, right.provider_id)

    def test_normalize_provider_uses_reported_insurance_and_provenance_source(self) -> None:
        provider = normalize_provider(
            {
                "provider_id": " provider-123 ",
                "name": " Harmony Family Clinic ",
                "specialties": ["Primary Care", " primary care "],
                "languages": ["English", "Spanish"],
                "insurance_reported": ["Medicare", "Aetna"],
                "accepted_insurance": ["Should Not Win"],
                "city": " San Francisco ",
                "state": " CA ",
                "country": " USA ",
                "provenance": {"source": "NPI Registry"},
            }
        )

        self.assertTrue(provider.provider_id.startswith("source:npi-registry:unknown:provider-123:"))
        self.assertEqual(provider.name, "Harmony Family Clinic")
        self.assertEqual(provider.specialties, ("Primary Care",))
        self.assertEqual(provider.insurance_reported, ("Medicare", "Aetna"))
        self.assertEqual(provider.source, "NPI Registry")
        self.assertEqual(provider.location_summary, "San Francisco, CA, USA")

    def test_normalize_provider_generates_source_aware_id_for_missing_input_id(self) -> None:
        provider = normalize_provider(
            {
                "provider_id": " ",
                "name": "Harmony Family Clinic",
                "source": "ClinicalTables",
                "provenance": {"dataset": "npi_idv"},
                "city": "Pittsburgh",
                "state": "PA",
            }
        )

        self.assertTrue(provider.provider_id.startswith("generated:clinicaltables:npi-idv:"))

    def test_normalize_provider_namespaces_explicit_source_local_ids_by_source_and_dataset(self) -> None:
        first = normalize_provider(
            {
                "provider_id": "provider-123",
                "name": "Harmony Family Clinic",
                "source": "ClinicalTables",
                "provenance": {"dataset": "npi_idv"},
            }
        )
        second = normalize_provider(
            {
                "provider_id": "provider-123",
                "name": "Harmony Family Clinic",
                "source": "ClinicalTables",
                "provenance": {"dataset": "npi_org"},
            }
        )
        third = normalize_provider(
            {
                "provider_id": "provider-123",
                "name": "Harmony Family Clinic",
                "source": "Trusted Directory",
                "provenance": {"dataset": "npi_idv"},
            }
        )

        self.assertTrue(first.provider_id.startswith("source:clinicaltables:npi-idv:provider-123:"))
        self.assertTrue(second.provider_id.startswith("source:clinicaltables:npi-org:provider-123:"))
        self.assertTrue(third.provider_id.startswith("source:trusted-directory:npi-idv:provider-123:"))
        self.assertNotEqual(first.provider_id, second.provider_id)
        self.assertNotEqual(first.provider_id, third.provider_id)

    def test_normalize_provider_preserves_global_npi_backed_id(self) -> None:
        provider = normalize_provider(
            {
                "provider_id": "1619271780",
                "name": "Harmony Family Clinic",
                "source": "NPI Registry (individual)",
                "provenance": {"dataset": "npi_idv"},
                "NPI": "1619271780",
            }
        )

        self.assertEqual(provider.provider_id, "1619271780")

    def test_normalize_search_result_preserves_nested_retriever_metadata(self) -> None:
        result = normalize_search_result(
            {
                "id": "provider-123",
                "name": "Harmony Family Clinic",
                "source": "ClinicalTables",
                "score": 0.875,
                "retriever_metadata": {
                    "similarity": 0.875,
                    "node_id": "node-1",
                    "freshness": {"last_updated_epoch": 200},
                },
            }
        )

        self.assertTrue(result.provider.provider_id.startswith("source:clinicaltables:unknown:provider-123:"))
        self.assertEqual(result.source, "ClinicalTables")
        self.assertEqual(result.score, 0.875)
        self.assertEqual(
            result.retriever_metadata,
            {
                "similarity": 0.875,
                "node_id": "node-1",
                "freshness": {"last_updated_epoch": 200},
            },
        )

    def test_normalize_provider_preserves_existing_trust_and_freshness_fields(self) -> None:
        provider = normalize_provider(
            {
                "provider_id": " provider-123 ",
                "name": " Harmony Family Clinic ",
                "source": "ClinicalTables",
                "insurance_network_verification": {
                    "status": "verified",
                    "verified": True,
                    "basis": "Plan directory confirmed.",
                    "source": "Aetna directory",
                },
                "accepting_new_patients_status": {
                    "status": "accepting",
                    "verified": True,
                    "basis": "Office confirmed this week.",
                    "source": "Clinic staff",
                },
                "medicare_opt_out": {
                    "opted_out": False,
                    "optout_effective_date": "2025-01-01",
                    "optout_end_date": "2027-01-01",
                },
                "retrieval_metadata": {
                    "last_updated_epoch": 200,
                    "extensions": {"directory": "payer"},
                },
            }
        )

        self.assertEqual(provider.insurance_network_verification.status, "verified")
        self.assertTrue(provider.insurance_network_verification.verified)
        self.assertEqual(provider.insurance_network_verification.basis, "Plan directory confirmed.")
        self.assertEqual(provider.insurance_network_verification.source, "Aetna directory")
        self.assertEqual(provider.accepting_new_patients_status.status, "accepting")
        self.assertTrue(provider.accepting_new_patients_status.verified)
        self.assertEqual(provider.accepting_new_patients_status.source, "Clinic staff")
        self.assertIsNotNone(provider.medicare_opt_out)
        assert provider.medicare_opt_out is not None
        self.assertFalse(provider.medicare_opt_out.opted_out)
        self.assertEqual(
            provider.freshness,
            FreshnessMetadata(
                source="ClinicalTables",
                created_epoch=None,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(provider.retrieval_metadata["last_updated_epoch"], 200)
        self.assertEqual(
            provider.retrieval_metadata["extensions"],
            {"directory": "payer"},
        )

    def test_normalize_provider_accepts_canonical_provider_without_losing_trust_fields(self) -> None:
        provider = CanonicalProvider(
            provider_id=" provider-123 ",
            name=" Harmony Family Clinic ",
            source=" ClinicalTables ",
            insurance_network_verification=VerificationStatus(
                status="verified",
                verified=True,
                basis="Plan directory confirmed.",
                source="Aetna directory",
            ),
            accepting_new_patients_status=VerificationStatus(
                status="accepting",
                verified=True,
                basis="Office confirmed this week.",
                source="Clinic staff",
            ),
            medicare_opt_out=MedicareOptOutStatus(
                opted_out=False,
                optout_effective_date="2025-01-01",
                optout_end_date="2027-01-01",
            ),
            freshness=FreshnessMetadata(
                source="ClinicalTables",
                dataset="clinicaltables",
                created_epoch=100,
                last_updated_epoch=200,
            ),
            retrieval_metadata={"last_updated_epoch": 200},
            provenance={"dataset": "clinicaltables"},
        )

        normalized = normalize_provider(provider)

        self.assertTrue(normalized.provider_id.startswith("source:clinicaltables:clinicaltables:provider-123:"))
        self.assertEqual(normalized.name, "Harmony Family Clinic")
        self.assertEqual(normalized.source, "ClinicalTables")
        self.assertEqual(
            normalized.insurance_network_verification,
            VerificationStatus(
                status="verified",
                verified=True,
                basis="Plan directory confirmed.",
                source="Aetna directory",
            ),
        )
        self.assertEqual(
            normalized.accepting_new_patients_status,
            VerificationStatus(
                status="accepting",
                verified=True,
                basis="Office confirmed this week.",
                source="Clinic staff",
            ),
        )
        self.assertEqual(
            normalized.medicare_opt_out,
            MedicareOptOutStatus(
                opted_out=False,
                optout_effective_date="2025-01-01",
                optout_end_date="2027-01-01",
            ),
        )
        self.assertEqual(
            normalized.freshness,
            FreshnessMetadata(
                source="ClinicalTables",
                dataset="clinicaltables",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(normalized.retrieval_metadata["last_updated_epoch"], 200)

    def test_normalize_search_result_preserves_nested_provider_trust_and_context(self) -> None:
        result = normalize_search_result(
            {
                "provider_id": "outer-source-id",
                "source": "ClinicalTables",
                "provenance": {
                    "source": "ClinicalTables",
                    "dataset": "npi_idv",
                    "trace": {"request_id": "outer-request"},
                },
                "insurance_network_verification": {
                    "status": "verified",
                    "verified": True,
                    "basis": "Outer verification should fill gaps.",
                    "source": "Aetna directory",
                },
                "freshness": {
                    "source": "NPPES Registry",
                    "dataset": "nppes",
                    "created_epoch": 100,
                    "last_updated_epoch": 200,
                },
                "retrieval_metadata": {
                    "outer_context": {"phase": "search"},
                },
                "provider": {
                    "provider_id": " ",
                    "name": "Harmony Family Clinic",
                    "accepting_new_patients_status": {
                        "status": "accepting",
                        "verified": True,
                        "basis": "Office confirmed this week.",
                        "source": "Clinic staff",
                    },
                    "medicare_opt_out": {
                        "opted_out": False,
                    },
                    "provenance": {
                        "source": "ClinicalTables",
                        "trace": {"provider_rank": 1},
                    },
                    "retrieval_metadata": {
                        "inner_context": {"rank": 1},
                    },
                },
                "score": 0.875,
                "retriever_metadata": {
                    "similarity": 0.875,
                    "node_id": "node-1",
                    "trace": {"request_id": "search-123"},
                },
            }
        )

        self.assertTrue(result.provider.provider_id.startswith("source:clinicaltables:npi-idv:outer-source-id:"))
        self.assertEqual(result.provider.insurance_network_verification.status, "verified")
        self.assertTrue(result.provider.insurance_network_verification.verified)
        self.assertTrue(result.provider.accepting_new_patients_status.verified)
        self.assertIsNotNone(result.provider.medicare_opt_out)
        assert result.provider.medicare_opt_out is not None
        self.assertFalse(result.provider.medicare_opt_out.opted_out)
        self.assertEqual(
            result.provider.provenance,
            {
                "source": "ClinicalTables",
                "dataset": "npi_idv",
                "trace": {
                    "request_id": "outer-request",
                    "provider_rank": 1,
                },
            },
        )
        self.assertEqual(
            result.provider.freshness,
            FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(
            result.provider.retrieval_metadata,
            {
                "outer_context": {"phase": "search"},
                "inner_context": {"rank": 1},
            },
        )
        self.assertEqual(
            result.retriever_metadata,
            {
                "similarity": 0.875,
                "node_id": "node-1",
                "trace": {"request_id": "search-123"},
            },
        )
        self.assertEqual(result.source, "ClinicalTables")


class ProviderSearchCacheTests(unittest.TestCase):
    def test_resolve_provider_cache_path_prefers_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_provider_cache_path({"PROVIDER_CACHE_DIR": temp_dir})

            self.assertEqual(resolved, Path(temp_dir) / DEFAULT_PROVIDER_CACHE_FILENAME)

    def test_resolve_provider_cache_path_falls_back_to_temp_directory(self) -> None:
        fallback = Path(tempfile.gettempdir()) / DEFAULT_PROVIDER_CACHE_SUBDIR

        with patch("provider_search.cache._prepare_cache_directory", side_effect=[False, True]):
            resolved = resolve_provider_cache_path({"PROVIDER_CACHE_DIR": "/root/blocked"})

        self.assertEqual(resolved, fallback / DEFAULT_PROVIDER_CACHE_FILENAME)

    def test_sqlite_provider_search_cache_round_trips_phi_free_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "provider_search.sqlite3"
            cache = SQLiteProviderSearchCache(database_path=database_path)
            entry = ProviderSearchCacheEntry(
                cache_key="cache-key-1",
                request_fingerprint="fingerprint-1",
                provider_ids=("provider-123", "provider-456"),
                sources=("ClinicalTables", "NPI Registry"),
                stored_at="2026-04-22T12:00:00+00:00",
                expires_at="2026-04-22T13:00:00+00:00",
            )

            self.assertTrue(cache.set(entry))
            self.assertEqual(cache.get("cache-key-1"), entry)
