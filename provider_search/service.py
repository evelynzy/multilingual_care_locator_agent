from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import re
from typing import Optional, Protocol, Sequence

from provider_search.models import (
    CanonicalProvider,
    ProviderSearchCacheEntry,
    ProviderSearchRequest,
    ProviderSearchResponse,
    SearchTrace,
    SourceSearchRequest,
    SourceSearchResult,
    SourceTrace,
)
from provider_search.normalization import (
    build_request_fingerprint,
    normalize_provider,
    normalize_search_request,
    normalize_search_result,
)
from provider_search.ranking import rank_provider_results


DEFAULT_CLINICALTABLES_DATASETS = ("npi_idv", "npi_org")


class SearchDatasetBackend(Protocol):
    def search_dataset(self, dataset: str, request: SourceSearchRequest) -> SourceSearchResult:
        """Execute a source search for one dataset."""


class ProviderSearchCacheBackend(Protocol):
    def get(self, cache_key: str) -> Optional[ProviderSearchCacheEntry]:
        """Return a PHI-free cache entry if available."""

    def set(self, entry: ProviderSearchCacheEntry) -> bool:
        """Persist a PHI-free cache entry."""


class ProviderSearchService:
    """Facade that orchestrates provider retrieval, normalization, caching, and ranking."""

    def __init__(
        self,
        *,
        clinicaltables_source: SearchDatasetBackend,
        cache: Optional[ProviderSearchCacheBackend] = None,
        datasets: Sequence[str] = DEFAULT_CLINICALTABLES_DATASETS,
        per_dataset_limit: int = 10,
    ) -> None:
        self.clinicaltables_source = clinicaltables_source
        self.cache = cache
        self.datasets = tuple(datasets)
        self.per_dataset_limit = per_dataset_limit

    def search(
        self,
        request: ProviderSearchRequest,
        limit: int = 5,
    ) -> ProviderSearchResponse:
        normalized_request = normalize_search_request(request)
        request_fingerprint = build_request_fingerprint(normalized_request)
        cache_key = self._build_cache_key(request_fingerprint)
        cache_entry = self.cache.get(cache_key) if self.cache is not None else None

        source_request = self._build_source_request(normalized_request, limit=limit)
        source_traces: list[SourceTrace] = []
        missing_location_hint: Optional[str] = None
        deduped_providers: dict[str, CanonicalProvider] = {}

        for dataset in self.datasets:
            source_result = self.clinicaltables_source.search_dataset(dataset, source_request)
            trace = source_result.trace or SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                error="Provider source did not return trace metadata.",
            )
            source_traces.append(trace)
            if missing_location_hint is None and source_result.missing_location_hint:
                missing_location_hint = source_result.missing_location_hint

            for raw_provider in source_result.providers:
                provider = normalize_provider(raw_provider)
                existing_provider = deduped_providers.get(provider.provider_id)
                if existing_provider is None:
                    deduped_providers[provider.provider_id] = provider
                else:
                    deduped_providers[provider.provider_id] = self._merge_provider_records(
                        primary=provider,
                        fallback=existing_provider,
                    )

        ranked_results = rank_provider_results(
            normalized_request,
            list(deduped_providers.values()),
            limit=limit,
            cached_provider_ids=cache_entry.provider_ids if cache_entry is not None else (),
        )

        sources_attempted = tuple(self._trace_label(trace) for trace in source_traces)
        sources_used = tuple(
            dict.fromkeys(
                result.source or result.provider.source
                for result in ranked_results
                if (result.source or result.provider.source)
            )
        )

        if ranked_results and self.cache is not None:
            self.cache.set(
                ProviderSearchCacheEntry(
                    cache_key=cache_key,
                    request_fingerprint=request_fingerprint,
                    provider_ids=tuple(result.provider.provider_id for result in ranked_results),
                    sources=sources_used,
                    stored_at=datetime.now(timezone.utc).isoformat(),
                    expires_at=None,
                )
            )

        return ProviderSearchResponse(
            request=normalized_request,
            provider_results=tuple(ranked_results),
            fallback_resources=(),
            missing_location_hint=missing_location_hint,
            search_trace=SearchTrace(
                source_traces=tuple(source_traces),
                sources_attempted=sources_attempted,
                sources_used=sources_used,
                cache_hit=cache_entry is not None,
                cache_key=cache_key,
                request_fingerprint=request_fingerprint,
                total_candidates=len(deduped_providers),
            ),
        )

    def _build_source_request(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
    ) -> SourceSearchRequest:
        city_hint, state_hint, zip_hint = self._extract_location_hints(request.location)
        search_terms = self._compose_search_terms(request)
        return SourceSearchRequest(
            search_terms=search_terms,
            limit=max(limit, self.per_dataset_limit),
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
        )

    @staticmethod
    def _build_cache_key(request_fingerprint: str) -> str:
        return f"provider-search:{request_fingerprint}"

    @staticmethod
    def _compose_search_terms(request: ProviderSearchRequest) -> str:
        terms = [*request.specialties, *request.keywords]
        if not terms and request.location:
            terms.append(request.location)
        if not terms:
            terms.extend(request.insurance)
        if not terms:
            terms.extend(request.preferred_languages)
        return " ".join(term for term in terms if term).strip()

    @staticmethod
    def _extract_location_hints(location: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if not location:
            return None, None, None

        normalized = location.strip()
        if not normalized:
            return None, None, None

        zip_match = re.search(r"\b\d{5}(?:-\d{4})?\b", normalized)
        zip_hint = zip_match.group(0)[:5] if zip_match else None
        location_without_zip = normalized
        if zip_match:
            location_without_zip = (
                normalized[: zip_match.start()] + normalized[zip_match.end() :]
            ).strip(" ,")

        parts = [part.strip() for part in location_without_zip.split(",") if part.strip()]
        city_hint: Optional[str] = None
        state_hint: Optional[str] = None

        if len(parts) >= 2:
            city_hint = parts[0] or None
            state_hint = ProviderSearchService._normalize_state_hint(parts[1])
        elif len(parts) == 1:
            tokens = parts[0].split()
            if tokens and re.fullmatch(r"[A-Za-z]{2}", tokens[-1]):
                state_hint = tokens[-1].upper()
                city_tokens = tokens[:-1]
                city_hint = " ".join(city_tokens) or None
            else:
                city_hint = parts[0]

        return city_hint, state_hint, zip_hint

    @staticmethod
    def _normalize_state_hint(value: str) -> Optional[str]:
        tokens = value.split()
        if not tokens:
            return None
        candidate = tokens[0].strip()
        if re.fullmatch(r"[A-Za-z]{2}", candidate):
            return candidate.upper()
        return None

    @staticmethod
    def _trace_label(trace: SourceTrace) -> str:
        if trace.dataset:
            return f"{trace.source_name}:{trace.dataset}"
        return trace.source_name

    @staticmethod
    def _merge_provider_records(
        *,
        primary: CanonicalProvider,
        fallback: CanonicalProvider,
    ) -> CanonicalProvider:
        fallback_payload = asdict(fallback)
        fallback_payload["provider"] = primary
        return normalize_search_result(fallback_payload).provider
