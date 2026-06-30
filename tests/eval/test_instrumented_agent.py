from __future__ import annotations

import unittest

from eval.instrumented_agent import RecordingSearchService
from provider_search.models import ProviderSearchRequest, ProviderSearchResponse


class _FakeInner:
    def __init__(self):
        self.calls = []

    def search(self, request, limit=5):
        self.calls.append((request, limit))
        return ProviderSearchResponse(request=request)


class RecordingSearchServiceTests(unittest.TestCase):
    def test_records_request_and_response_and_delegates(self):
        inner = _FakeInner()
        rec = RecordingSearchService(inner)
        req = ProviderSearchRequest(specialties=("cardiology",), location="95051")

        resp = rec.search(req, limit=3)

        self.assertEqual(inner.calls, [(req, 3)])
        self.assertIs(rec.last_request, req)
        self.assertIs(rec.last_response, resp)
        self.assertIsInstance(resp, ProviderSearchResponse)

    def test_reset_clears_captures(self):
        rec = RecordingSearchService(_FakeInner())
        rec.search(ProviderSearchRequest(specialties=("ent",)), limit=2)
        rec.reset()
        self.assertIsNone(rec.last_request)
        self.assertIsNone(rec.last_response)


if __name__ == "__main__":
    unittest.main()
