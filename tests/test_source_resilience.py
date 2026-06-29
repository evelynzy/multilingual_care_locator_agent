"""Resilience tests for the live data-source adapters.

These exercise the error/edge paths of ClinicalTablesSource and NPPESSource by
injecting a fake ``session`` whose ``.get()`` returns a Mock response or raises.
The existing suite only covered the happy path plus one generic timeout; this
covers HTTP 4xx/5xx, a valid-200-but-malformed-JSON body, connection resets,
timeout propagation, and NPPES-specific degradation.
"""

import unittest
from unittest.mock import Mock

from provider_search.models import CanonicalProvider, ProviderSearchRequest, SourceSearchRequest
from provider_search.service import ProviderSearchService
from provider_search.sources.clinicaltables import ClinicalTablesSource
from provider_search.sources.nppes import NPPESSource


# Local stand-ins for the requests exceptions. The adapters catch a generic
# ``Exception``, so the concrete type is irrelevant to behavior, and using local
# classes keeps these tests independent of whether another test module has
# installed a stubbed ``requests`` in sys.modules (which lacks ``.exceptions``).
class _FakeHTTPError(Exception):
    """Stand-in for requests.exceptions.HTTPError raised by raise_for_status()."""


class _FakeConnectionError(Exception):
    """Stand-in for a connection reset / network failure."""


def _ok_json_response(payload, *, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


class ClinicalTablesResilienceTests(unittest.TestCase):
    def _request(self):
        return SourceSearchRequest(search_terms="family medicine", limit=3)

    def test_http_5xx_returns_error_trace_not_exception(self):
        response = Mock()
        response.status_code = 503
        response.raise_for_status.side_effect = _FakeHTTPError(
            "503 Server Error: Service Unavailable"
        )
        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset("npi_idv", self._request())

        self.assertEqual(result.providers, [])
        self.assertIsNotNone(result.trace)
        self.assertIn("503", result.trace.error)
        self.assertEqual(result.trace.dataset, "npi_idv")
        self.assertIsNotNone(result.trace.request_url)
        # The body must not be parsed once the HTTP status is an error.
        response.json.assert_not_called()

    def test_http_4xx_rate_limit_returns_error_trace(self):
        response = Mock()
        response.status_code = 429
        response.raise_for_status.side_effect = _FakeHTTPError(
            "429 Client Error: Too Many Requests"
        )
        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset("npi_idv", self._request())

        self.assertEqual(result.providers, [])
        self.assertIn("429", result.trace.error)

    def test_valid_200_but_malformed_json_degrades_to_empty(self):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("No JSON object could be decoded")
        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset("npi_idv", self._request())

        # A 200 with an unparseable body must not crash; it yields an empty,
        # successful-shaped trace (no error string, zero results).
        self.assertEqual(result.providers, [])
        self.assertIsNotNone(result.trace)
        self.assertIsNone(result.trace.error)
        self.assertEqual(result.trace.result_count, 0)

    def test_connection_reset_returns_error_trace(self):
        session = Mock()
        session.get.side_effect = _FakeConnectionError("connection reset by peer")
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset("npi_idv", self._request())

        self.assertEqual(result.providers, [])
        self.assertIn("connection reset", result.trace.error)

    def test_timeout_value_is_propagated_to_session_get(self):
        response = _ok_json_response([])  # valid but empty payload -> no providers
        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(timeout=11, session=session)

        source.search_dataset("npi_idv", self._request())

        session.get.assert_called_once()
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["timeout"], 11)

    def test_empty_search_terms_short_circuit_without_network(self):
        session = Mock()
        source = ClinicalTablesSource(session=session)

        result = source.search_dataset("npi_idv", SourceSearchRequest(search_terms="   ", limit=3))

        session.get.assert_not_called()
        self.assertEqual(result.providers, [])
        self.assertIsNotNone(result.trace.error)


class NPPESResilienceTests(unittest.TestCase):
    DIGIT_NPI = "1234567890"

    def _source_returning(self, payload):
        response = _ok_json_response(payload)
        session = Mock()
        session.get.return_value = response
        return NPPESSource(session=session), session

    def test_http_error_returns_none(self):
        response = Mock()
        response.raise_for_status.side_effect = _FakeHTTPError("500")
        session = Mock()
        session.get.return_value = response
        source = NPPESSource(session=session)

        self.assertIsNone(source.lookup(self.DIGIT_NPI))

    def test_malformed_json_returns_none(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")
        session = Mock()
        session.get.return_value = response
        source = NPPESSource(session=session)

        self.assertIsNone(source.lookup(self.DIGIT_NPI))

    def test_non_list_taxonomies_become_empty_list(self):
        payload = {
            "results": [
                {
                    "taxonomies": {"desc": "Cardiology"},  # malformed: dict, not list
                    "addresses": [],
                    "basic": {"name": "Heart Clinic"},
                }
            ]
        }
        source, _ = self._source_returning(payload)

        record = source.lookup(self.DIGIT_NPI)

        self.assertIsNotNone(record)
        self.assertEqual(record.taxonomies, [])

    def test_missing_basic_is_none(self):
        payload = {
            "results": [
                {"taxonomies": [], "addresses": [], "basic": "not-a-dict"}
            ]
        }
        source, _ = self._source_returning(payload)

        record = source.lookup(self.DIGIT_NPI)

        self.assertIsNotNone(record)
        self.assertIsNone(record.basic)

    def test_nonexistent_npi_with_empty_results_is_cached_single_call(self):
        # A well-formed "no match" payload (results: []) is cacheable, so a
        # repeated lookup must not issue a second network request.
        source, session = self._source_returning({"results": []})

        self.assertIsNone(source.lookup(self.DIGIT_NPI))
        self.assertIsNone(source.lookup(self.DIGIT_NPI))

        session.get.assert_called_once()

    def test_non_digit_npi_skips_network(self):
        session = Mock()
        source = NPPESSource(session=session)

        self.assertIsNone(source.lookup("NPI-N/A"))
        session.get.assert_not_called()

    def test_enrich_provider_with_non_digit_id_skips_network(self):
        session = Mock()
        source = NPPESSource(session=session)
        provider = CanonicalProvider(provider_id="generated-abc", name="Some Clinic")

        result = source.enrich_provider(provider)

        self.assertIs(result, provider)
        session.get.assert_not_called()


class _AlwaysFailingClinicalTablesSource:
    """SearchDatasetBackend whose every dataset call raises."""

    def __init__(self):
        self.calls = []

    def search_dataset(self, dataset, request):
        self.calls.append((dataset, request))
        raise RuntimeError(f"{dataset} backend unavailable")


class ServiceLevelResilienceTests(unittest.TestCase):
    def test_search_degrades_to_empty_when_all_sources_fail(self):
        source = _AlwaysFailingClinicalTablesSource()
        service = ProviderSearchService(clinicaltables_source=source)

        response = service.search(
            ProviderSearchRequest(specialties=("Cardiology",), location="95051")
        )

        # No exception escapes, no providers, and both datasets were attempted.
        self.assertEqual(response.provider_results, ())
        self.assertGreaterEqual(len(source.calls), 2)


if __name__ == "__main__":
    unittest.main()
