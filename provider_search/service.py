from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import logging
import os
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
    normalize_text,
)
from provider_search.ranking import rank_provider_results


DEFAULT_CLINICALTABLES_DATASETS = ("npi_idv", "npi_org")
logger = logging.getLogger(__name__)


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
        cache_entry = self._read_cache_entry(cache_key)

        planned_requests = self._plan_source_requests(normalized_request, limit=limit)
        source_request = planned_requests[0] if planned_requests else self._build_source_request(
            normalized_request,
            limit=limit,
        )
        source_traces, missing_location_hint, deduped_providers = (
            self._collect_planned_source_candidates(planned_requests)
        )

        if self._should_retry_search(
            request=normalized_request,
            source_request=source_request,
            source_traces=source_traces,
            deduped_providers=deduped_providers,
            missing_location_hint=missing_location_hint,
        ):
            retry_requests = self._plan_retry_source_requests(normalized_request, limit=limit)
            retry_traces, retry_missing_location_hint, retry_candidates = (
                self._collect_planned_source_candidates(retry_requests)
            )
            source_traces.extend(retry_traces)
            if missing_location_hint is None and retry_missing_location_hint:
                missing_location_hint = retry_missing_location_hint
            for provider_id, provider in retry_candidates.items():
                existing_provider = deduped_providers.get(provider_id)
                if existing_provider is None:
                    deduped_providers[provider_id] = provider
                else:
                    deduped_providers[provider_id] = self._merge_provider_records(
                        primary=provider,
                        fallback=existing_provider,
                    )

        ranked_results = self._rank_results(
            request=normalized_request,
            deduped_providers=deduped_providers,
            limit=None,
            cache_entry=cache_entry,
        )
        if self._should_retry_with_nearby_state(
            request=normalized_request,
            deduped_providers=deduped_providers,
            ranked_results=ranked_results,
            missing_location_hint=missing_location_hint,
        ):
            nearby_retry_requests = self._plan_nearby_retry_source_requests(
                normalized_request,
                limit=limit,
                candidate_providers=deduped_providers.values(),
            )
            nearby_retry_traces, nearby_retry_missing_location_hint, nearby_retry_candidates = (
                self._collect_planned_source_candidates(nearby_retry_requests)
            )
            source_traces.extend(nearby_retry_traces)
            if missing_location_hint is None and nearby_retry_missing_location_hint:
                missing_location_hint = nearby_retry_missing_location_hint
            for provider_id, provider in nearby_retry_candidates.items():
                existing_provider = deduped_providers.get(provider_id)
                if existing_provider is None:
                    deduped_providers[provider_id] = provider
                else:
                    deduped_providers[provider_id] = self._merge_provider_records(
                        primary=provider,
                        fallback=existing_provider,
                    )
            ranked_results = self._rank_results(
                request=normalized_request,
                deduped_providers=deduped_providers,
                limit=None,
                cache_entry=cache_entry,
            )
        display_results = self._dedupe_display_results(ranked_results)
        if limit is not None:
            display_results = display_results[: max(limit, 0)]
        self._log_debug_summary(
            request_fingerprint=request_fingerprint,
            cache_hit=cache_entry is not None,
            source_traces=source_traces,
            total_candidates=len(deduped_providers),
            ranked_results=ranked_results,
            display_results=display_results,
        )

        sources_attempted = tuple(self._trace_label(trace) for trace in source_traces)
        sources_used = tuple(
            dict.fromkeys(
                result.source or result.provider.source
                for result in ranked_results
                if (result.source or result.provider.source)
            )
        )

        if ranked_results:
            self._write_cache_entry(
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
            provider_results=tuple(display_results),
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

    @staticmethod
    def _rank_results(
        *,
        request: ProviderSearchRequest,
        deduped_providers: dict[str, CanonicalProvider],
        limit: int,
        cache_entry: Optional[ProviderSearchCacheEntry],
    ) -> list["ProviderSearchResult"]:
        return rank_provider_results(
            request,
            list(deduped_providers.values()),
            limit=limit,
            cached_provider_ids=cache_entry.provider_ids if cache_entry is not None else (),
        )

    def _log_debug_summary(
        self,
        *,
        request_fingerprint: str,
        cache_hit: bool,
        source_traces: Sequence[SourceTrace],
        total_candidates: int,
        ranked_results: Sequence["ProviderSearchResult"],
        display_results: Sequence["ProviderSearchResult"],
    ) -> None:
        if not self._debug_enabled():
            return

        logger.info(
            "provider_search_debug request_fingerprint=%s cache_hit=%s raw_candidates=%s post_gate=%s post_display_dedupe=%s",
            request_fingerprint,
            cache_hit,
            total_candidates,
            len(ranked_results),
            len(display_results),
        )
        for trace in source_traces:
            logger.info(
                "provider_search_debug_source dataset=%s result_count=%s error=%s",
                trace.dataset or "unknown",
                trace.result_count,
                trace.error or "none",
            )
        for result in display_results:
            display_dedupe_count = result.retriever_metadata.get("display_dedupe_count")
            display_dedupe_ids = result.retriever_metadata.get("display_dedupe_provider_ids")
            if display_dedupe_count or display_dedupe_ids:
                logger.info(
                    "provider_search_debug_display provider_id=%s display_dedupe_count=%s display_dedupe_provider_ids=%s",
                    result.provider.provider_id,
                    display_dedupe_count or 0,
                    display_dedupe_ids or [],
                )

    def _dedupe_display_results(
        self,
        ranked_results: Sequence["ProviderSearchResult"],
    ) -> list["ProviderSearchResult"]:
        deduped_results: list["ProviderSearchResult"] = []
        index_by_display_key: dict[tuple[str, str], int] = {}

        for result in ranked_results:
            display_key = self._display_dedupe_key(result.provider)
            if display_key is None:
                deduped_results.append(result)
                continue

            existing_index = index_by_display_key.get(display_key)
            if existing_index is None:
                index_by_display_key[display_key] = len(deduped_results)
                deduped_results.append(result)
                continue

            existing_result = deduped_results[existing_index]
            if not self._should_merge_display_duplicate(
                primary=existing_result.provider,
                duplicate=result.provider,
            ):
                deduped_results.append(result)
                continue

            deduped_results[existing_index] = self._merge_display_duplicate_result(
                primary=existing_result,
                duplicate=result,
            )

        return deduped_results

    def _build_source_request(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
        location_only: bool = False,
        relax_location_filter: bool = False,
    ) -> SourceSearchRequest:
        city_hint, state_hint, zip_hint = self._extract_location_hints(request.location)
        search_terms = self._compose_search_terms(request, location_only=location_only)
        query_filter = self._build_location_query_filter(
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
            relax_location_filter=relax_location_filter,
        )
        return SourceSearchRequest(
            search_terms=search_terms,
            limit=max(limit, self.per_dataset_limit),
            query_filter=query_filter,
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
        )

    def _plan_source_requests(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
        location_only: bool = False,
        relax_location_filter: bool = False,
    ) -> list[SourceSearchRequest]:
        city_hint, state_hint, zip_hint = self._extract_location_hints(request.location)
        planned_terms = self._plan_search_terms(
            request,
            location_only=location_only,
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
        )
        query_filter = self._build_location_query_filter(
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
            relax_location_filter=relax_location_filter,
        )

        planned_requests: list[SourceSearchRequest] = []
        seen_terms: set[str] = set()
        for search_terms in planned_terms:
            normalized_terms = search_terms.strip()
            if not normalized_terms or normalized_terms in seen_terms:
                continue
            seen_terms.add(normalized_terms)
            planned_requests.append(
                SourceSearchRequest(
                    search_terms=normalized_terms,
                    limit=max(limit, self.per_dataset_limit),
                    query_filter=query_filter,
                    city_hint=city_hint,
                    state_hint=state_hint,
                    zip_hint=zip_hint,
                )
            )

        if planned_requests:
            return planned_requests
        return [
            self._build_source_request(
                request,
                limit=limit,
                location_only=location_only,
                relax_location_filter=relax_location_filter,
            )
        ]

    def _build_retry_source_request(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
    ) -> SourceSearchRequest:
        if request.specialties:
            return self._build_source_request(
                request,
                limit=limit,
                relax_location_filter=True,
            )
        return self._build_source_request(
            request,
            limit=limit,
            location_only=True,
        )

    def _plan_retry_source_requests(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
    ) -> list[SourceSearchRequest]:
        if request.specialties:
            return self._plan_source_requests(
                request,
                limit=limit,
                relax_location_filter=True,
            )
        return self._plan_source_requests(
            request,
            limit=limit,
            location_only=True,
        )

    def _plan_nearby_retry_source_requests(
        self,
        request: ProviderSearchRequest,
        *,
        limit: int,
        candidate_providers: Sequence[CanonicalProvider],
    ) -> list[SourceSearchRequest]:
        inferred_state = self._infer_retry_state_hint(
            request=request,
            candidate_providers=candidate_providers,
        )
        if inferred_state is None:
            return []

        nearby_request = ProviderSearchRequest(
            specialties=request.specialties,
            location=inferred_state,
            insurance=request.insurance,
            preferred_languages=request.preferred_languages,
            keywords=request.keywords,
        )
        return self._plan_source_requests(
            nearby_request,
            limit=limit,
        )

    def _collect_planned_source_candidates(
        self,
        requests: Sequence[SourceSearchRequest],
    ) -> tuple[list[SourceTrace], Optional[str], dict[str, CanonicalProvider]]:
        source_traces: list[SourceTrace] = []
        missing_location_hint: Optional[str] = None
        deduped_providers: dict[str, CanonicalProvider] = {}

        for request in requests:
            request_traces, request_missing_location_hint, request_candidates = (
                self._collect_source_candidates(request)
            )
            source_traces.extend(request_traces)
            if missing_location_hint is None and request_missing_location_hint:
                missing_location_hint = request_missing_location_hint
            for provider_id, provider in request_candidates.items():
                existing_provider = deduped_providers.get(provider_id)
                if existing_provider is None:
                    deduped_providers[provider_id] = provider
                else:
                    deduped_providers[provider_id] = self._merge_provider_records(
                        primary=provider,
                        fallback=existing_provider,
                    )

        return source_traces, missing_location_hint, deduped_providers

    def _collect_source_candidates(
        self,
        request: SourceSearchRequest,
    ) -> tuple[list[SourceTrace], Optional[str], dict[str, CanonicalProvider]]:
        source_traces: list[SourceTrace] = []
        missing_location_hint: Optional[str] = None
        deduped_providers: dict[str, CanonicalProvider] = {}

        for dataset in self.datasets:
            source_result = self._search_dataset(dataset, request)
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

        return source_traces, missing_location_hint, deduped_providers

    @staticmethod
    def _build_cache_key(request_fingerprint: str) -> str:
        return f"provider-search:{request_fingerprint}"

    def _read_cache_entry(self, cache_key: str) -> Optional[ProviderSearchCacheEntry]:
        if self.cache is None:
            return None

        try:
            return self.cache.get(cache_key)
        except Exception as exc:
            logger.warning("Provider search cache read failed for %s: %s", cache_key, exc)
            return None

    def _write_cache_entry(self, entry: ProviderSearchCacheEntry) -> None:
        if self.cache is None:
            return

        try:
            self.cache.set(entry)
        except Exception as exc:
            logger.warning("Provider search cache write failed for %s: %s", entry.cache_key, exc)

    def _search_dataset(
        self,
        dataset: str,
        request: SourceSearchRequest,
    ) -> SourceSearchResult:
        try:
            return self.clinicaltables_source.search_dataset(dataset, request)
        except Exception as exc:
            logger.warning("Provider source failed for dataset %s: %s", dataset, exc)
            return SourceSearchResult(
                providers=[],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    error=str(exc),
                ),
            )

    @staticmethod
    def _compose_search_terms(
        request: ProviderSearchRequest,
        *,
        location_only: bool = False,
    ) -> str:
        if location_only:
            return (request.location or "").strip()

        terms = list(request.specialties)
        if not terms:
            terms.extend(request.keywords)
        if not terms and request.location:
            terms.append(request.location)
        if not terms:
            terms.extend(request.insurance)
        if not terms:
            terms.extend(request.preferred_languages)
        return " ".join(term for term in terms if term).strip()

    def _plan_search_terms(
        self,
        request: ProviderSearchRequest,
        *,
        location_only: bool,
        city_hint: Optional[str],
        state_hint: Optional[str],
        zip_hint: Optional[str],
    ) -> list[str]:
        base_terms = self._compose_search_terms(request, location_only=location_only)
        if not base_terms.strip():
            return []
        if location_only or not request.specialties:
            return [base_terms]

        planned_terms = [base_terms]

        suggested_specialties = self._suggest_specialty_terms(request.specialties)
        suggested_base_terms = " ".join(suggested_specialties).strip()
        if suggested_base_terms and suggested_base_terms not in planned_terms:
            planned_terms.append(suggested_base_terms)
        else:
            suggested_base_terms = base_terms

        location_assisted_terms = self._build_location_assisted_terms(
            suggested_base_terms,
            location=request.location,
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
        )
        for candidate in location_assisted_terms:
            if candidate not in planned_terms:
                planned_terms.append(candidate)

        return planned_terms

    def _suggest_specialty_terms(self, specialties: Sequence[str]) -> tuple[str, ...]:
        suggest_specialty_terms = getattr(self.clinicaltables_source, "suggest_specialty_terms", None)
        if callable(suggest_specialty_terms):
            suggested = suggest_specialty_terms(specialties)
            if suggested:
                return tuple(term for term in suggested if isinstance(term, str) and term.strip())
        return tuple(term for term in specialties if isinstance(term, str) and term.strip())

    def _build_location_assisted_terms(
        self,
        base_terms: str,
        *,
        location: Optional[str],
        city_hint: Optional[str],
        state_hint: Optional[str],
        zip_hint: Optional[str],
    ) -> list[str]:
        build_location_assisted_terms = getattr(
            self.clinicaltables_source,
            "build_location_assisted_terms",
            None,
        )
        if callable(build_location_assisted_terms):
            return list(
                build_location_assisted_terms(
                    base_terms,
                    location=location,
                    city_hint=city_hint,
                    state_hint=state_hint,
                    zip_hint=zip_hint,
                )
            )

        fallback_terms = " ".join(
            part for part in (city_hint, state_hint, zip_hint) if isinstance(part, str) and part.strip()
        ).strip()
        if not fallback_terms:
            return []
        return [f"{base_terms} {fallback_terms}".strip()]

    @staticmethod
    def _should_retry_search(
        *,
        request: ProviderSearchRequest,
        source_request: SourceSearchRequest,
        source_traces: Sequence[SourceTrace],
        deduped_providers: dict[str, CanonicalProvider],
        missing_location_hint: Optional[str],
    ) -> bool:
        if deduped_providers or missing_location_hint:
            return False
        if not request.location or not source_request.query_filter:
            return False
        if not request.specialties and not request.keywords:
            return False
        if source_request.search_terms == request.location.strip():
            return False
        return any(trace.error is None for trace in source_traces)

    def _should_retry_with_nearby_state(
        self,
        *,
        request: ProviderSearchRequest,
        deduped_providers: dict[str, CanonicalProvider],
        ranked_results: Sequence["ProviderSearchResult"],
        missing_location_hint: Optional[str],
    ) -> bool:
        if ranked_results or missing_location_hint or not deduped_providers:
            return False
        if not request.specialties:
            return False

        city_hint, state_hint, zip_hint = self._extract_location_hints(request.location)
        if not zip_hint or city_hint:
            return False

        return (
            self._infer_retry_state_hint(
                request=request,
                candidate_providers=deduped_providers.values(),
            )
            is not None
        )

    def _infer_retry_state_hint(
        self,
        *,
        request: ProviderSearchRequest,
        candidate_providers: Sequence[CanonicalProvider],
    ) -> Optional[str]:
        _, state_hint, _ = self._extract_location_hints(request.location)
        if state_hint:
            return state_hint

        inferred_states = {
            provider.state.strip().upper()
            for provider in candidate_providers
            if isinstance(provider.state, str) and re.fullmatch(r"[A-Za-z]{2}", provider.state.strip())
        }
        if len(inferred_states) != 1:
            return None
        return next(iter(inferred_states))

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
    def _build_location_query_filter(
        *,
        city_hint: Optional[str],
        state_hint: Optional[str],
        zip_hint: Optional[str],
        relax_location_filter: bool = False,
    ) -> Optional[str]:
        filters: list[str] = []

        if state_hint:
            filters.append(f"addr_practice.state:{state_hint}")
            if relax_location_filter:
                return " AND ".join(filters)

        if zip_hint:
            filters.append(f"addr_practice.zip:{zip_hint}")
        elif city_hint and state_hint:
            escaped_city = city_hint.replace('"', '\\"')
            filters.append(f'addr_practice.city:"{escaped_city}"')

        if not filters:
            return None
        return " AND ".join(filters)

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

    @staticmethod
    def _display_dedupe_key(
        provider: CanonicalProvider,
    ) -> Optional[tuple[str, str]]:
        normalized_name = normalize_text(provider.name, lowercase=True)
        normalized_location = normalize_text(
            provider.location_summary or provider.address,
            lowercase=True,
        )
        if not normalized_name or not normalized_location:
            return None
        return normalized_name, normalized_location

    @staticmethod
    def _should_merge_display_duplicate(
        *,
        primary: CanonicalProvider,
        duplicate: CanonicalProvider,
    ) -> bool:
        if not (
            ProviderSearchService._is_organization_style_provider(primary)
            or ProviderSearchService._is_organization_style_provider(duplicate)
        ):
            return False

        primary_phone = normalize_text(primary.phone, lowercase=True)
        duplicate_phone = normalize_text(duplicate.phone, lowercase=True)
        if primary_phone and duplicate_phone and primary_phone != duplicate_phone:
            return False

        if (
            ProviderSearchService._visible_service_line_key(primary)
            != ProviderSearchService._visible_service_line_key(duplicate)
        ):
            return False

        return True

    @staticmethod
    def _is_organization_style_provider(provider: CanonicalProvider) -> bool:
        source_values = (
            provider.source,
            provider.provenance.get("source") if isinstance(provider.provenance, dict) else None,
            provider.provenance.get("dataset") if isinstance(provider.provenance, dict) else None,
            provider.retrieval_metadata.get("dataset")
            if isinstance(provider.retrieval_metadata, dict)
            else None,
        )
        normalized_values = [
            normalize_text(value, lowercase=True) or ""
            for value in source_values
        ]
        return any("organization" in value or "npi_org" in value for value in normalized_values)

    @staticmethod
    def _visible_service_line_key(provider: CanonicalProvider) -> tuple[Optional[str], tuple[str, ...]]:
        normalized_taxonomy = normalize_text(provider.taxonomy, lowercase=True)
        normalized_specialties = tuple(
            sorted(
                {
                    normalized_value
                    for normalized_value in (
                        normalize_text(value, lowercase=True)
                        for value in provider.specialties
                    )
                    if normalized_value
                }
            )
        )
        return normalized_taxonomy, normalized_specialties

    def _merge_display_duplicate_result(
        self,
        *,
        primary: "ProviderSearchResult",
        duplicate: "ProviderSearchResult",
    ) -> "ProviderSearchResult":
        merged_provider = self._merge_provider_records(
            primary=primary.provider,
            fallback=duplicate.provider,
        )
        merged_metadata = dict(duplicate.retriever_metadata)
        merged_metadata.update(primary.retriever_metadata)

        provider_ids = self._collect_display_duplicate_provider_ids(primary, duplicate)
        merged_sources = self._collect_display_duplicate_sources(primary, duplicate)
        if len(provider_ids) > 1:
            merged_metadata["display_dedupe_provider_ids"] = provider_ids
            merged_metadata["display_dedupe_count"] = len(provider_ids)
        if len(merged_sources) > 1:
            merged_metadata["display_dedupe_sources"] = merged_sources

        return normalize_search_result(
            {
                "provider": merged_provider,
                "score": primary.score,
                "source": primary.source or merged_provider.source,
                "retriever_metadata": merged_metadata,
            }
        )

    @staticmethod
    def _collect_display_duplicate_provider_ids(
        primary: "ProviderSearchResult",
        duplicate: "ProviderSearchResult",
    ) -> list[str]:
        provider_ids: list[str] = []
        for result in (primary, duplicate):
            metadata_ids = result.retriever_metadata.get("display_dedupe_provider_ids")
            if isinstance(metadata_ids, list):
                candidate_ids = [str(value) for value in metadata_ids if value]
            else:
                candidate_ids = [result.provider.provider_id]
            for provider_id in candidate_ids:
                if provider_id and provider_id not in provider_ids:
                    provider_ids.append(provider_id)
        return provider_ids

    @staticmethod
    def _collect_display_duplicate_sources(
        primary: "ProviderSearchResult",
        duplicate: "ProviderSearchResult",
    ) -> list[str]:
        sources: list[str] = []
        for result in (primary, duplicate):
            metadata_sources = result.retriever_metadata.get("display_dedupe_sources")
            if isinstance(metadata_sources, list):
                candidate_sources = [str(value) for value in metadata_sources if value]
            else:
                candidate_sources = [
                    source
                    for source in (result.source, result.provider.source)
                    if isinstance(source, str) and source
                ]
            for source in candidate_sources:
                if source not in sources:
                    sources.append(source)
        return sources

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("PROVIDER_SEARCH_DEBUG", "").strip() == "1"
