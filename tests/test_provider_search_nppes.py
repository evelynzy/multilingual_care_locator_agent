import unittest
from unittest.mock import Mock

from provider_search.models import FreshnessMetadata
from provider_search.normalization import build_canonical_provider
from provider_search.sources.nppes import NPPESSource


class NPPESSourceTests(unittest.TestCase):
    def test_build_lookup_request_uses_expected_query_params(self) -> None:
        source = NPPESSource(lookup_url="https://example.test/nppes", version="2.1")

        url, params = source.build_lookup_request("1619271780")

        self.assertEqual(url, "https://example.test/nppes")
        self.assertEqual(
            params,
            {
                "number": "1619271780",
                "version": "2.1",
                "limit": "1",
            },
        )

    def test_format_location_compacts_fields(self) -> None:
        address = {
            "address_1": "200 Lothrop St",
            "address_2": "Suite 123",
            "city": "Pittsburgh",
            "state": "PA",
            "postal_code": "15213-2582",
            "country_name": "United States",
        }

        location = NPPESSource.format_location(address)

        self.assertEqual(location, "200 Lothrop St, Suite 123, Pittsburgh, PA 15213-2582")

    def test_parse_payload_normalizes_lookup_entry(self) -> None:
        source = NPPESSource()
        payload = {
            "results": [
                {
                    "addresses": [
                        {"address_purpose": "MAILING", "telephone_number": "000-000-0000"},
                        {
                            "address_purpose": "LOCATION",
                            "address_1": "200 Lothrop St",
                            "city": "Pittsburgh",
                            "state": "PA",
                            "postal_code": "15213-2582",
                            "telephone_number": "412-605-3019",
                        },
                    ],
                    "taxonomies": [{"desc": "Urology"}],
                    "basic": {"organization_name": "UPMC"},
                    "enumeration_type": "NPI-1",
                    "created_epoch": 100,
                    "last_updated_epoch": 200,
                }
            ]
        }

        record = source.parse_payload("1619271780", payload)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.npi, "1619271780")
        self.assertEqual(record.practice_address["telephone_number"], "412-605-3019")
        self.assertEqual(record.mailing_address["telephone_number"], "000-000-0000")
        self.assertEqual(record.taxonomies[0]["desc"], "Urology")

    def test_enrich_provider_overrides_location_phone_and_taxonomy(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "addresses": [
                        {
                            "address_purpose": "LOCATION",
                            "address_1": "200 Lothrop St",
                            "city": "Pittsburgh",
                            "state": "PA",
                            "postal_code": "15213-2582",
                            "telephone_number": "412-605-3019",
                        },
                        {
                            "address_purpose": "MAILING",
                            "telephone_number": "000-000-0000",
                        },
                    ],
                    "taxonomies": [{"desc": "Urology"}],
                    "created_epoch": 100,
                    "last_updated_epoch": 200,
                }
            ]
        }

        session = Mock()
        session.get.return_value = response
        source = NPPESSource(session=session)

        provider = build_canonical_provider(
            provider_id="1619271780",
            name="Healthcare Provider",
            location="Old Address",
            phone="111-111-1111",
            taxonomy=None,
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            raw={},
        )

        enriched = source.enrich_provider(provider)

        self.assertEqual(enriched.location_summary, "200 Lothrop St, Pittsburgh, PA 15213-2582")
        self.assertEqual(enriched.phone, "412-605-3019")
        self.assertEqual(enriched.taxonomy, "Urology")
        self.assertIn("nppes", enriched.raw)
        self.assertTrue(enriched.retrieval_metadata["nppes_enriched"])
        self.assertEqual(
            enriched.freshness,
            FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=100,
                last_updated_epoch=200,
            ),
        )
        self.assertEqual(
            enriched.retrieval_metadata["nppes"],
            {
                "created_epoch": 100,
                "last_updated_epoch": 200,
            },
        )

    def test_lookup_returns_none_on_request_failure(self) -> None:
        session = Mock()
        session.get.side_effect = RuntimeError("boom")
        source = NPPESSource(session=session)

        self.assertIsNone(source.lookup("1619271780"))


if __name__ == "__main__":
    unittest.main()
