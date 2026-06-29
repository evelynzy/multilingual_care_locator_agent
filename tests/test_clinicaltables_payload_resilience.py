"""Characterization tests for ClinicalTables payload parsing resilience.

Targets ``ClinicalTablesSource.parse_search_payload``, ``_normalize_row`` and
``_build_fallback_provider`` in ``provider_search/sources/clinicaltables.py``.

These assert the expected/correct behavior. Where the product currently
behaves in a surprising way, the current behavior is pinned with an explicit
``# NOTE: current behavior, suspected bug`` comment rather than silently
matching it (see ``test_build_fallback_provider_null_fields_do_not_crash``).
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from provider_search.models import SourceSearchRequest
from provider_search.sources.clinicaltables import (
    ClinicalTablesSource,
    DEFAULT_DATASET_CONFIGS,
)


def _make_source_with_body(body) -> tuple[ClinicalTablesSource, Mock]:
    """Build a source whose session returns ``body`` from ``response.json()``."""

    response = Mock()
    response.status_code = 200
    response.json.return_value = body
    response.raise_for_status.return_value = None

    session = Mock()
    session.get.return_value = response
    return ClinicalTablesSource(session=session), session


class ParseSearchPayloadResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ClinicalTablesSource()

    def test_parse_search_payload_returns_empty_for_error_shaped_dict(self) -> None:
        fields, entries = self.source.parse_search_payload(
            "npi_idv", {"error": "rate limited"}
        )

        self.assertEqual(fields, [])
        self.assertEqual(entries, [])

    def test_parse_search_payload_returns_empty_for_non_list_payloads(self) -> None:
        for payload in (None, "oops", 123, 4.5, True):
            with self.subTest(payload=payload):
                fields, entries = self.source.parse_search_payload("npi_idv", payload)
                self.assertEqual(fields, [])
                self.assertEqual(entries, [])

    def test_parse_search_payload_skips_short_rows_but_keeps_well_formed(self) -> None:
        payload = [
            2,
            ["display row"],
            ["name.full", "NPI", "provider_type"],
            [
                ["Harmony Family Clinic", "1619271780", "Family Medicine"],
                ["TooShort", "1619271781"],  # len 2 <= highest resolved index (2) -> skipped
            ],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )

    def test_parse_search_payload_resolves_integer_and_string_descriptors(self) -> None:
        # ``6`` is the npi_idv field_map index for "NPI"; "name.full"/"provider_type"
        # are string descriptors. A short trailing row must be dropped, not crash.
        payload = [
            2,
            ["display row"],
            ["name.full", 6, "provider_type"],
            [
                ["Harmony Family Clinic", "1619271780", "Family Medicine"],
                ["only", "two"],  # skipped: too short for highest resolved index
            ],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )


class SearchDatasetPayloadResilienceTests(unittest.TestCase):
    def test_search_dataset_yields_zero_providers_for_error_shaped_body(self) -> None:
        source, session = _make_source_with_body({"error": "rate limited"})

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="family medicine", limit=5),
        )

        session.get.assert_called_once()
        self.assertEqual(result.providers, [])
        self.assertEqual(result.trace.result_count, 0)
        self.assertIsNone(result.trace.error)

    def test_search_dataset_yields_zero_providers_for_non_list_body(self) -> None:
        source, _ = _make_source_with_body("totally not a list")

        result = source.search_dataset(
            "npi_org",
            SourceSearchRequest(search_terms="clinic", limit=5),
        )

        self.assertEqual(result.providers, [])
        self.assertEqual(result.trace.result_count, 0)
        self.assertIsNone(result.trace.error)


class BuildFallbackProviderResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ClinicalTablesSource()
        self.idv_config = DEFAULT_DATASET_CONFIGS["npi_idv"]
        self.org_config = DEFAULT_DATASET_CONFIGS["npi_org"]

    def test_build_fallback_provider_handles_multi_comma_individual_name(self) -> None:
        row = [
            "Smith, Jr., John",
            "1619271780",
            "Family Medicine",
            "123 Main St",
            "408-555-0100",
        ]

        provider = self.source._build_fallback_provider("npi_idv", row, self.idv_config)

        # Only the first comma is used to split last/first, so the remainder of
        # the name is preserved instead of crashing or being dropped.
        self.assertTrue(provider.name)
        self.assertIn("John", provider.name)
        self.assertIn("Smith", provider.name)
        self.assertEqual(provider.provider_id, "1619271780")
        self.assertEqual(provider.phone, "408-555-0100")
        self.assertEqual(provider.taxonomy, "Family Medicine")

    def test_build_fallback_provider_empty_row_returns_none(self) -> None:
        self.assertIsNone(
            self.source._build_fallback_provider("npi_idv", [], self.idv_config)
        )

    def test_build_fallback_provider_falsy_name_defaults_to_placeholder(self) -> None:
        provider = self.source._build_fallback_provider(
            "npi_idv", [None, "1619271780"], self.idv_config
        )

        self.assertEqual(provider.name, "Healthcare Provider")
        self.assertEqual(provider.provider_id, "1619271780")

    def test_build_fallback_provider_null_fields_do_not_crash(self) -> None:
        row = ["Acme Clinic", "", None, None, None]

        # The primary contract under test: null/empty trailing fields must not
        # raise, and a usable provider record is still produced.
        provider = self.source._build_fallback_provider("npi_org", row, self.org_config)

        self.assertEqual(provider.name, "Acme Clinic")
        self.assertEqual(provider.source, "NPI Registry (organization)")
        # Corrected behavior: None row values must not surface as the string
        # "None"; they must be treated as absent (None / empty string).
        self.assertIsNone(provider.phone)
        self.assertIsNone(provider.taxonomy)
        # absent address normalizes to None (not the string "None")
        self.assertIsNone(provider.address)


if __name__ == "__main__":
    unittest.main()
