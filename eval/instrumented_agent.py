from __future__ import annotations

from typing import List, Optional

from care import CareLocatorAgent, ParsedCareQuery
from provider_search.models import ProviderSearchRequest, ProviderSearchResponse


class RecordingSearchService:
    """Transparent proxy that records the last search request/response.

    The agent calls `search(request, limit=...)` exactly once per request that
    reaches provider retrieval, so capturing the last call is sufficient per turn.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_request: Optional[ProviderSearchRequest] = None
        self.last_response: Optional[ProviderSearchResponse] = None

    def search(self, request: ProviderSearchRequest, limit: int = 5) -> ProviderSearchResponse:
        self.last_request = request
        response = self._inner.search(request, limit=limit)
        self.last_response = response
        return response

    def reset(self) -> None:
        self.last_request = None
        self.last_response = None


class TracingAgent(CareLocatorAgent):
    """CareLocatorAgent that captures the structured intent and the search call.

    `handle_request` may call `_interpret_user_need` more than once (full history
    then latest-only), so `last_parsed_query` holds the most recent parse; the
    effective merged intent that drove retrieval is read from the recording
    service's `last_request`.
    """

    def __init__(self, provider_search_service) -> None:
        super().__init__(provider_search_service=RecordingSearchService(provider_search_service))
        self.last_parsed_query: Optional[ParsedCareQuery] = None
        self.last_navigation_mode: Optional[str] = None
        self.captured_llm_inputs: List[str] = []

    def _interpret_user_need(
        self, client, message: str, history: List[dict]
    ) -> ParsedCareQuery:
        # These arguments are exactly the post-gate user text the intent LLM
        # call receives (handle_request redacts before calling) — recorded so
        # the phi_redacted metric can verify no raw PHI reaches the wire.
        self.captured_llm_inputs.append(message)
        for turn in history or []:
            if isinstance(turn, dict) and turn.get("role") == "user":
                self.captured_llm_inputs.append(str(turn.get("content") or ""))
        parsed = super()._interpret_user_need(client, message, history)
        self.last_parsed_query = parsed
        return parsed

    def _build_navigation_guidance(self, query, message: str):
        guidance = super()._build_navigation_guidance(query, message)
        self.last_navigation_mode = guidance.get("mode") if isinstance(guidance, dict) else None
        return guidance

    def reset_capture(self) -> None:
        self.last_parsed_query = None
        self.last_navigation_mode = None
        self.captured_llm_inputs = []
        self.provider_search_service.reset()
