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
    MedicareOptOutStatus,
    ProviderSearchCacheEntry,
    ProviderSearchRequest,
    VerificationStatus,
)
from provider_search.normalization import (
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

        self.assertEqual(provider.provider_id, "provider-123")
        self.assertEqual(provider.name, "Harmony Family Clinic")
        self.assertEqual(provider.specialties, ("Primary Care",))
        self.assertEqual(provider.insurance_reported, ("Medicare", "Aetna"))
        self.assertEqual(provider.source, "NPI Registry")
        self.assertEqual(provider.location_summary, "San Francisco, CA, USA")

    def test_normalize_search_result_captures_provider_and_scalar_metadata(self) -> None:
        result = normalize_search_result(
            {
                "id": "provider-123",
                "name": "Harmony Family Clinic",
                "source": "ClinicalTables",
                "score": 0.875,
                "retriever_metadata": {
                    "similarity": 0.875,
                    "node_id": "node-1",
                    "ignored": {"nested": "value"},
                },
            }
        )

        self.assertEqual(result.provider.provider_id, "provider-123")
        self.assertEqual(result.source, "ClinicalTables")
        self.assertEqual(result.score, 0.875)
        self.assertEqual(
            result.retriever_metadata,
            {"similarity": 0.875, "node_id": "node-1"},
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
            retrieval_metadata={"last_updated_epoch": 200},
        )

        normalized = normalize_provider(provider)

        self.assertEqual(normalized.provider_id, "provider-123")
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
        self.assertEqual(normalized.retrieval_metadata["last_updated_epoch"], 200)

    def test_normalize_search_result_preserves_nested_provider_trust_fields(self) -> None:
        result = normalize_search_result(
            {
                "provider": {
                    "provider_id": "provider-123",
                    "name": "Harmony Family Clinic",
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
                    },
                    "retrieval_metadata": {
                        "last_updated_epoch": 200,
                    },
                },
                "score": 0.875,
                "source": "ClinicalTables",
                "retriever_metadata": {
                    "similarity": 0.875,
                    "node_id": "node-1",
                },
            }
        )

        self.assertEqual(result.provider.provider_id, "provider-123")
        self.assertEqual(result.provider.insurance_network_verification.status, "verified")
        self.assertTrue(result.provider.accepting_new_patients_status.verified)
        self.assertIsNotNone(result.provider.medicare_opt_out)
        assert result.provider.medicare_opt_out is not None
        self.assertFalse(result.provider.medicare_opt_out.opted_out)
        self.assertEqual(result.provider.retrieval_metadata["last_updated_epoch"], 200)
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
