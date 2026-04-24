import unittest
from unittest.mock import Mock

from provider_search.models import SourceSearchRequest
from provider_search.sources.clinicaltables import ClinicalTablesSource, DEFAULT_DATASET_CONFIGS


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

    def test_parse_search_payload_resolves_mixed_descriptors_positionally(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", 6, {"unexpected": "value"}, 7],
            [["Harmony Family Clinic", "1619271780", "skip-me", "Family Medicine"], "skip-me"],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(
            fields,
            ["name.full", "NPI", "provider_type"],
        )
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )

    def test_parse_search_payload_skips_rows_with_unresolvable_shape(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", 6, 7],
            [["Harmony Family Clinic", "1619271780"]],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(entries, [])

    def test_parse_search_payload_uses_dataset_field_order_when_live_v3_payload_omits_descriptors(
        self,
    ) -> None:
        payload = [
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
                "95051",
                "",
                "408-555-0100",
                ["English"],
            ]],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, DEFAULT_DATASET_CONFIGS["npi_idv"].result_fields)
        self.assertEqual(entries, payload[3])

    def test_parse_search_payload_skips_non_row_entries_when_live_v3_payload_omits_descriptors(
        self,
    ) -> None:
        payload = [
            2,
            ["1619271780", "1619271781"],
            None,
            [
                [
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
                    "95051",
                    "",
                    "408-555-0100",
                    ["English"],
                ],
                "skip-me",
            ],
        ]

        fields, entries = self.source.parse_search_payload("npi_idv", payload)

        self.assertEqual(fields, DEFAULT_DATASET_CONFIGS["npi_idv"].result_fields)
        self.assertEqual(
            entries,
            [payload[3][0]],
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

    def test_default_dataset_configs_request_richer_taxonomy_fields(self) -> None:
        self.assertIn("taxonomies[0].desc", DEFAULT_DATASET_CONFIGS["npi_idv"].result_fields)
        self.assertIn("taxonomies[0].code", DEFAULT_DATASET_CONFIGS["npi_idv"].result_fields)
        self.assertIn("taxonomies[0].desc", DEFAULT_DATASET_CONFIGS["npi_org"].result_fields)
        self.assertIn("taxonomies[0].code", DEFAULT_DATASET_CONFIGS["npi_org"].result_fields)

    def test_search_dataset_uses_taxonomy_desc_when_provider_type_is_missing(self) -> None:
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
                "95051",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="OB/GYN", limit=1, zip_hint="95051"),
        )

        session.get.assert_called_once()
        _, kwargs = session.get.call_args
        self.assertIn("taxonomies[0].desc", kwargs["params"]["df"])
        self.assertIn("taxonomies[0].code", kwargs["params"]["df"])
        self.assertEqual(result.trace.result_count, 1)
        provider = result.providers[0]
        self.assertEqual(provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(provider.taxonomy, "Obstetrics & Gynecology")
        self.assertEqual(provider.specialty_family_ids, ("obstetrics-gynecology",))

    def test_search_dataset_accepts_live_v3_payload_shape_without_field_descriptors(self) -> None:
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
                "95051",
                "",
                "408-555-0100",
                ["English"],
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_idv",
            SourceSearchRequest(search_terms="ob gyn", limit=1, zip_hint="95051"),
        )

        self.assertEqual(result.trace.result_count, 1)
        provider = result.providers[0]
        self.assertEqual(provider.provider_id, "1619271780")
        self.assertEqual(provider.name, "Cupertino OB/GYN Associates")
        self.assertEqual(provider.taxonomy, "Obstetrics & Gynecology")
        self.assertIn("Santa Clara", provider.location_summary or "")
        self.assertIn("95051", provider.location_summary or "")
        self.assertEqual(provider.phone, "408-555-0100")
        self.assertIn("English", provider.languages)
        self.assertEqual(provider.specialty_family_ids, ("obstetrics-gynecology",))

    def test_build_search_request_uses_punctuation_light_specialty_terms_and_puts_location_in_q_with_sf(self) -> None:
        _, params = self.source.build_search_request(
            "npi_idv",
            SourceSearchRequest(
                search_terms="Obstetrics & Gynecology 95051",
                limit=5,
                specialty_driven=True,
                query_filter="addr_practice.zip:95051",
                zip_hint="95051",
            ),
        )

        self.assertEqual(params["terms"], "obstetrics gynecology")
        self.assertEqual(params["q"], "addr_practice.zip:95051")
        self.assertEqual(
            params["sf"],
            ",".join(
                [
                    "provider_type",
                    "licenses.medicare.type",
                    "licenses.taxonomy.classification",
                    "licenses.taxonomy.specialization",
                    "licenses.taxonomy.code",
                ]
            ),
        )

    def test_build_search_request_expands_obgyn_abbreviation_to_punctuation_light_terms(self) -> None:
        _, params = self.source.build_search_request(
            "npi_idv",
            SourceSearchRequest(
                search_terms="OB/GYN",
                limit=5,
                specialty_driven=True,
                query_filter="addr_practice.zip:95051",
                zip_hint="95051",
            ),
        )

        self.assertEqual(params["terms"], "obstetrics gynecology")

    def test_search_dataset_does_not_treat_name_fragment_as_taxonomy_in_mixed_descriptor_payload(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                1,
                {"unexpected": "value"},
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
            ],
            [[
                "Harmony Group Practice",
                "1619271780",
                "Ann",
                "Santa Clara",
                "CA",
                "95051",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset(
            "npi_org",
            SourceSearchRequest(search_terms="ob gyn 95051", limit=1),
        )

        self.assertEqual(result.trace.result_count, 1)
        provider = result.providers[0]
        self.assertEqual(provider.name, "Harmony Group Practice")
        self.assertIsNone(provider.taxonomy)
        self.assertIn("Santa Clara", provider.location_summary or "")
        self.assertNotIn("Ann", provider.specialties)

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

    def test_suggest_specialty_terms_falls_back_to_cleaned_specialty_when_values_unavailable(self) -> None:
        suggested = self.source.suggest_specialty_terms((" Pediatrics ",))

        self.assertEqual(suggested, ("Pediatrics",))

    def test_build_location_assisted_terms_uses_alias_and_hints(self) -> None:
        assisted_terms = self.source.build_location_assisted_terms(
            "Pediatrics",
            location="dallas fort worth",
            city_hint="Dallas",
            state_hint="TX",
            zip_hint="75001",
        )

        self.assertEqual(
            assisted_terms,
            [
                "Pediatrics Dallas TX",
                "Pediatrics 75001 Dallas TX",
            ],
        )


if __name__ == "__main__":
    unittest.main()
