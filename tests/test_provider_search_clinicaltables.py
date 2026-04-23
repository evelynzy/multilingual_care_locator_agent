import unittest
from unittest.mock import Mock

from provider_search.models import SourceSearchRequest
from provider_search.sources.clinicaltables import ClinicalTablesSource


class ClinicalTablesSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ClinicalTablesSource()

    def test_parse_fields_payload_supports_headerless_rows(self) -> None:
        payload = [
            [0, "name.full", "Provider Name"],
            [1, "NPI", "NPI"],
            [2, "provider_type", "Taxonomy"],
        ]

        mapping = self.source.parse_fields_payload(payload)

        self.assertEqual(
            mapping,
            {
                "name.full": 0,
                "NPI": 1,
                "provider_type": 2,
            },
        )

    def test_parse_fields_payload_skips_invalid_entries(self) -> None:
        payload = [
            "ignored heading",
            ["not-an-index", "name.full"],
            [1, ""],
            [2, "provider_type"],
        ]

        mapping = self.source.parse_fields_payload(payload)

        self.assertEqual(mapping, {"name.full": 0, "provider_type": 2})

    def test_parse_search_payload_uses_string_fields(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", "NPI", "provider_type"],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )

    def test_parse_search_payload_resolves_integer_field_indexes(self) -> None:
        payload = [
            1,
            ["display row"],
            [0, 6, 7],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(len(entries), 1)

    def test_parse_search_payload_falls_back_to_configured_field_order(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", 6, {"unexpected": "value"}],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"], "skip-me"],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(
            fields,
            self.source.result_field_order["npi_idv"],
        )
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )

    def test_search_dataset_builds_request_and_normalizes_provider(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            ["name.full", "NPI", "provider_type", "addr_practice.full", "addr_practice.phone"],
            [[
                "Harmony Family Clinic",
                "1619271780",
                "Family Medicine",
                "123 Main St, Pittsburgh, PA 15213",
                "412-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="family medicine pittsburgh", limit=1),
        )

        session.get.assert_called_once()
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["timeout"], 6)
        self.assertEqual(kwargs["params"]["terms"], "family medicine pittsburgh")
        self.assertEqual(kwargs["params"]["maxList"], "1")
        self.assertIn("df", kwargs["params"])

        self.assertEqual(result.trace.result_count, 1)
        provider = result.providers[0]
        self.assertEqual(provider.provider_id, "1619271780")
        self.assertEqual(provider.name, "Harmony Family Clinic")
        self.assertEqual(provider.location_summary, "123 Main St, Pittsburgh, PA 15213")
        self.assertEqual(provider.phone, "412-555-0100")
        self.assertEqual(provider.taxonomy, "Family Medicine")
        self.assertEqual(provider.source, "NPI Registry (individual)")

    def test_search_dataset_applies_location_hints(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            ["name.full", "NPI", "provider_type", "addr_practice.city", "addr_practice.state"],
            [["Harmony Family Clinic", "1619271780", "Family Medicine", "Pittsburgh", "PA"]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(
                search_terms="family medicine",
                limit=1,
                state_hint="CA",
            ),
        )

        self.assertEqual(result.providers, [])
        self.assertEqual(result.trace.result_count, 0)

    def test_search_dataset_returns_trace_on_timeout(self) -> None:
        session = Mock()
        session.get.side_effect = RuntimeError("timed out")
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="family medicine", limit=1),
        )

        self.assertEqual(result.providers, [])
        self.assertIn("timed out", result.trace.error)

    def test_search_dataset_generates_stable_source_aware_id_when_npi_missing(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            ["name.full", "NPI", "provider_type", "addr_practice.city", "addr_practice.state"],
            [["Harmony Family Clinic", "", "Family Medicine", "Pittsburgh", "PA"]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        first_result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="family medicine pittsburgh", limit=1),
        )
        second_result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="family medicine pittsburgh", limit=1),
        )

        first_provider_id = first_result.providers[0].provider_id
        second_provider_id = second_result.providers[0].provider_id

        self.assertTrue(first_provider_id.startswith("generated:npi-registry-individual:npi-idv:"))
        self.assertEqual(first_provider_id, second_provider_id)


if __name__ == "__main__":
    unittest.main()
