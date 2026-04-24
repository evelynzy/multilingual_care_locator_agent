from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
from provider_search.ranking import evaluate_provider_gate, rank_provider_results
from provider_search.specialty_families import (
    SPECIALTY_FAMILY_BY_ID,
    derive_request_specialty_family_ids,
)


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
        self._log_debug_plan(
            request_fingerprint=request_fingerprint,
            request=normalized_request,
            planned_requests=planned_requests,
        )
        source_request = planned_requests[0] if planned_requests else self._build_source_request(
            normalized_request,
            limit=limit,
        )
        source_traces, missing_location_hint, deduped_providers = (
            self._collect_planned_source_candidates(
                planned_requests,
                request=normalized_request,
                continue_until_usable_specialty_evidence=(
                    self._should_continue_zip_only_specialty_variants(normalized_request)
                ),
            )
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
        self._log_scoped_candidate_details(
            request_fingerprint=request_fingerprint,
            request=normalized_request,
            providers=deduped_providers.values(),
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
                display_dedupe_keys = tuple(
                    self._provider_debug_key(str(provider_id))
                    for provider_id in display_dedupe_ids or ()
                    if provider_id
                )
                logger.info(
                    "provider_search_debug_display provider_key=%s display_dedupe_count=%s display_dedupe_provider_keys=%s",
                    self._provider_debug_key(result.provider.provider_id),
                    display_dedupe_count or 0,
                    display_dedupe_keys,
                )

    def _log_debug_plan(
        self,
        *,
        request_fingerprint: str,
        request: ProviderSearchRequest,
        planned_requests: Sequence[SourceSearchRequest],
    ) -> None:
        if not self._debug_enabled():
            return

        logger.info(
            "provider_search_debug_plan request_fingerprint=%s specialties=%s requested_family_ids=%s location_present=%s keyword_count=%s variants=%s",
            request_fingerprint,
            tuple(request.specialties),
            tuple(request.specialty_family_ids),
            bool(request.location),
            len(request.keywords),
            tuple(self._debug_variant_signature(source_request) for source_request in planned_requests),
        )

    def _log_debug_variant_result(
        self,
        *,
        source_request: SourceSearchRequest,
        source_traces: Sequence[SourceTrace],
        candidates: dict[str, CanonicalProvider],
    ) -> None:
        if not self._debug_enabled():
            return

        logger.info(
            "provider_search_debug_variant terms_key=%s filter_shape=%s traces=%s candidates=%s",
            self._debug_text_key(source_request.search_terms),
            self._debug_query_filter_shape(source_request.query_filter),
            tuple(
                (trace.dataset or "unknown", trace.result_count, bool(trace.error))
                for trace in source_traces
            ),
            len(candidates),
        )

    def _log_debug_variant_stop(
        self,
        *,
        source_request: SourceSearchRequest,
        reason: str,
        candidate_count: int,
    ) -> None:
        if not self._debug_enabled():
            return

        logger.info(
            "provider_search_debug_variant_stop reason=%s terms_key=%s filter_shape=%s candidate_count=%s",
            reason,
            self._debug_text_key(source_request.search_terms),
            self._debug_query_filter_shape(source_request.query_filter),
            candidate_count,
        )

    def _log_debug_candidate(
        self,
        *,
        provider: CanonicalProvider,
        dataset: str,
    ) -> None:
        if not self._debug_enabled():
            return

        logger.info(
            "provider_search_debug_candidate dataset=%s provider_key=%s taxonomy=%s specialty_evidence=%s family_ids=%s freshness_present=%s",
            dataset,
            self._provider_debug_key(provider.provider_id),
            provider.taxonomy or "",
            tuple(provider.specialties[:6]),
            tuple(provider.specialty_family_ids),
            bool(
                provider.freshness is not None
                and (
                    provider.freshness.created_epoch is not None
                    or provider.freshness.last_updated_epoch is not None
                )
            ),
        )

    def _log_scoped_candidate_details(
        self,
        *,
        request_fingerprint: str,
        request: ProviderSearchRequest,
        providers: Sequence[CanonicalProvider],
    ) -> None:
        if not self._scoped_debug_enabled(request_fingerprint):
            return

        for provider in sorted(providers, key=lambda candidate: candidate.provider_id):
            gate_evaluation = evaluate_provider_gate(request, provider)
            logger.info(
                "provider_search_debug_candidate_detail request_fingerprint=%s provider_key=%s source=%s dataset=%s taxonomy=%s specialties=%s family_ids=%s gate_outcome=%s drop_reason=%s",
                request_fingerprint,
                self._provider_debug_key(provider.provider_id),
                provider.source or "",
                self._debug_provider_dataset(provider),
                provider.taxonomy or "",
                tuple(provider.specialties[:8]),
                tuple(provider.specialty_family_ids),
                "admitted" if gate_evaluation.admitted else "dropped",
                gate_evaluation.drop_reason or "none",
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
            specialty_driven=bool(request.specialties) and not location_only,
            request_fingerprint=build_request_fingerprint(request),
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
                    specialty_driven=bool(request.specialties) and not location_only,
                    request_fingerprint=build_request_fingerprint(request),
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
        *,
        request: Optional[ProviderSearchRequest] = None,
        continue_until_usable_specialty_evidence: bool = False,
    ) -> tuple[list[SourceTrace], Optional[str], dict[str, CanonicalProvider]]:
        source_traces: list[SourceTrace] = []
        missing_location_hint: Optional[str] = None
        deduped_providers: dict[str, CanonicalProvider] = {}

        for source_request in requests:
            request_traces, request_missing_location_hint, request_candidates = (
                self._collect_source_candidates(source_request)
            )
            self._log_debug_variant_result(
                source_request=source_request,
                source_traces=request_traces,
                candidates=request_candidates,
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
            if (
                continue_until_usable_specialty_evidence
                and request is not None
                and self._has_request_relevant_specialty_evidence(
                    request=request,
                    providers=deduped_providers.values(),
                )
            ):
                self._log_debug_variant_stop(
                    source_request=source_request,
                    reason="usable_specialty_evidence",
                    candidate_count=len(deduped_providers),
                )
                break

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
                self._log_debug_candidate(provider=provider, dataset=dataset)
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

        planned_terms: list[str] = []
        specialty_term_variants = self._plan_specialty_term_variants(request.specialties)
        for candidate in specialty_term_variants:
            if candidate not in planned_terms:
                planned_terms.append(candidate)

        location_assisted_seed = specialty_term_variants[-1] if specialty_term_variants else base_terms
        for candidate in self._build_location_assisted_terms(
            location_assisted_seed,
            location=request.location,
            city_hint=city_hint,
            state_hint=state_hint,
            zip_hint=zip_hint,
        ):
            if candidate not in planned_terms:
                planned_terms.append(candidate)

        return planned_terms

    def _plan_specialty_term_variants(self, specialties: Sequence[str]) -> list[str]:
        cleaned_specialties = tuple(
            term.strip() for term in specialties if isinstance(term, str) and term.strip()
        )
        if not cleaned_specialties:
            return []

        variants: list[str] = []
        base_terms = " ".join(cleaned_specialties).strip()
        if base_terms:
            variants.append(base_terms)

        suggested_specialties = self._suggest_specialty_terms(cleaned_specialties)
        suggested_terms = " ".join(suggested_specialties).strip()
        if suggested_terms and suggested_terms not in variants:
            variants.append(suggested_terms)

        canonical_specialties = tuple(
            self._canonical_specialty_search_term(term)
            for term in cleaned_specialties
        )
        canonical_terms = " ".join(canonical_specialties).strip()
        if canonical_terms and canonical_terms != suggested_terms and canonical_terms not in variants:
            variants.append(canonical_terms)

        return variants

    def _suggest_specialty_terms(self, specialties: Sequence[str]) -> tuple[str, ...]:
        cleaned_specialties = tuple(
            term for term in specialties if isinstance(term, str) and term.strip()
        )
        if not cleaned_specialties:
            return ()

        suggest_specialty_terms = getattr(self.clinicaltables_source, "suggest_specialty_terms", None)
        if callable(suggest_specialty_terms):
            suggested = tuple(
                term
                for term in suggest_specialty_terms(cleaned_specialties)
                if isinstance(term, str) and term.strip()
            )
            if len(suggested) == len(cleaned_specialties):
                return suggested
        return tuple(
            self._canonical_specialty_search_term(term)
            for term in cleaned_specialties
        )

    def _has_request_relevant_specialty_evidence(
        self,
        *,
        request: ProviderSearchRequest,
        providers: Sequence[CanonicalProvider],
    ) -> bool:
        requested_specialties = tuple(
            specialty.strip()
            for specialty in request.specialties
            if isinstance(specialty, str) and specialty.strip()
        )
        if not requested_specialties:
            return False

        requested_lookup = {specialty.casefold() for specialty in requested_specialties}
        requested_family_ids = set(derive_request_specialty_family_ids(requested_specialties))

        for provider in providers:
            if self._provider_has_request_relevant_specialty_evidence(
                provider=provider,
                requested_lookup=requested_lookup,
                requested_family_ids=requested_family_ids,
            ):
                return True
        return False

    @staticmethod
    def _provider_has_request_relevant_specialty_evidence(
        *,
        provider: CanonicalProvider,
        requested_lookup: set[str],
        requested_family_ids: set[str],
    ) -> bool:
        provider_family_ids = {
            family_id
            for family_id in getattr(provider, "specialty_family_ids", ())
            if isinstance(family_id, str) and family_id.strip()
        }
        if requested_family_ids and provider_family_ids.intersection(requested_family_ids):
            return True

        provider_values = list(provider.specialties)
        if provider.taxonomy:
            provider_values.append(provider.taxonomy)
        for value in provider_values:
            normalized_value = normalize_text(value, lowercase=True)
            if normalized_value and normalized_value in requested_lookup:
                return True

        return False

    def _should_continue_zip_only_specialty_variants(
        self,
        request: ProviderSearchRequest,
    ) -> bool:
        if not request.specialties:
            return False
        city_hint, _, zip_hint = self._extract_location_hints(request.location)
        return bool(zip_hint and not city_hint)

    @staticmethod
    def _canonical_specialty_search_term(specialty: str) -> str:
        if not ProviderSearchService._needs_canonical_specialty_fallback(specialty):
            return specialty

        family_ids = derive_request_specialty_family_ids((specialty,))
        if len(family_ids) != 1:
            return specialty

        family = SPECIALTY_FAMILY_BY_ID.get(family_ids[0])
        if family is None or not family.label.strip():
            return specialty
        return family.label

    @staticmethod
    def _needs_canonical_specialty_fallback(specialty: str) -> bool:
        normalized_specialty = specialty.strip()
        if not normalized_specialty:
            return False
        return re.search(r"[^A-Za-z0-9 ]", normalized_specialty) is not None

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
            filters.append(f"addr_practice.zip:{zip_hint}*")
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
        return (
            os.getenv("PROVIDER_SEARCH_DEBUG", "").strip() == "1"
            and os.getenv("CARE_LOCATOR_LOCAL_DEBUG", "").strip() == "1"
        )

    def _scoped_debug_enabled(self, request_fingerprint: str) -> bool:
        if not self._debug_enabled():
            return False
        selector = os.getenv("PROVIDER_SEARCH_DEBUG_FINGERPRINT", "").strip()
        return bool(selector) and selector == request_fingerprint

    @staticmethod
    def _provider_debug_key(provider_id: str) -> str:
        if not provider_id:
            return "missing"
        return hashlib.sha1(str(provider_id).encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _debug_text_key(value: Optional[str]) -> str:
        normalized_value = normalize_text(value)
        if not normalized_value:
            return "missing"
        return hashlib.sha1(normalized_value.encode("utf-8")).hexdigest()[:8]

    def _debug_variant_signature(
        self,
        source_request: SourceSearchRequest,
    ) -> tuple[str, str]:
        return (
            self._debug_text_key(source_request.search_terms),
            self._debug_query_filter_shape(source_request.query_filter),
        )

    @staticmethod
    def _debug_query_filter_shape(query_filter: Optional[str]) -> str:
        normalized_filter = normalize_text(query_filter)
        if not normalized_filter:
            return "none"

        fragments: list[str] = []
        if "addr_practice.city" in normalized_filter:
            fragments.append("city")
        if "addr_practice.state" in normalized_filter:
            fragments.append("state")
        if "addr_practice.zip" in normalized_filter:
            fragments.append("zip")
        if not fragments:
            return "other"
        if " AND " in normalized_filter:
            return "+".join(fragments)
        return fragments[0]

    @staticmethod
    def _debug_dataset_from_mapping(mapping: object) -> str:
        if not isinstance(mapping, dict):
            return ""
        dataset = mapping.get("dataset")
        if isinstance(dataset, str):
            return dataset
        return ""

    def _debug_provider_dataset(self, provider: CanonicalProvider) -> str:
        freshness_dataset = provider.freshness.dataset if provider.freshness is not None else None
        for candidate in (
            freshness_dataset,
            self._debug_dataset_from_mapping(provider.provenance),
            self._debug_dataset_from_mapping(provider.retrieval_metadata),
        ):
            if candidate:
                return candidate
        return ""
