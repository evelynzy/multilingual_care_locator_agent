from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from provider_search.models import CanonicalProvider, ProviderSearchRequest, ProviderSearchResult
from provider_search.normalization import normalize_provider, normalize_search_request


RANKING_VERSION = "deterministic-v1"

_ACCEPTING_STATUSES = {"accepting", "accepting new patients", "open"}
_TELEHEALTH_TERMS = {"telehealth", "virtual", "video", "remote", "online"}


def rank_provider_results(
    request: ProviderSearchRequest,
    providers: Sequence[CanonicalProvider | dict[str, object]],
    *,
    limit: Optional[int] = None,
    cached_provider_ids: Iterable[str] = (),
) -> list[ProviderSearchResult]:
    """Rank canonical providers with a deterministic, request-aware score."""

    normalized_request = normalize_search_request(request)
    cached_ids = {provider_id for provider_id in cached_provider_ids if provider_id}

    ranked_results: list[ProviderSearchResult] = []
    for raw_provider in providers:
        provider = normalize_provider(raw_provider)
        matched_specialties = _match_values(
            normalized_request.specialties,
            provider.specialties,
            extra_values=(provider.taxonomy,) if provider.taxonomy else (),
        )
        matched_keywords = _match_keywords(normalized_request.keywords, provider)
        if normalized_request.specialties and not matched_specialties:
            continue
        if normalized_request.keywords and not normalized_request.specialties and not matched_keywords:
            continue
        breakdown = _build_score_breakdown(
            normalized_request,
            provider,
            matched_specialties=matched_specialties,
            matched_keywords=matched_keywords,
        )
        score = round(sum(breakdown.values()), 6)
        ranking_metadata = dict(provider.ranking_metadata)
        ranking_metadata.update(
            {
                "ranking_version": RANKING_VERSION,
                "score_breakdown": breakdown,
                "cached_identity_match": provider.provider_id in cached_ids,
                "matched_specialties": matched_specialties,
                "matched_keywords": matched_keywords,
                "matched_languages": _match_values(
                    normalized_request.preferred_languages,
                    provider.languages,
                ),
                "matched_insurance": _match_values(
                    normalized_request.insurance,
                    provider.insurance_reported,
                ),
            }
        )

        retriever_metadata = {
            "ranking_version": RANKING_VERSION,
        }
        if provider.provider_id in cached_ids:
            retriever_metadata["cache_hint"] = "matched_prior_result"

        ranked_results.append(
            ProviderSearchResult(
                provider=provider.with_updates(ranking_metadata=ranking_metadata),
                score=score,
                source=provider.source,
                retriever_metadata=retriever_metadata,
            )
        )

    ranked_results.sort(key=_provider_result_sort_key)
    if limit is None:
        return ranked_results
    return ranked_results[: max(limit, 0)]


def _build_score_breakdown(
    request: ProviderSearchRequest,
    provider: CanonicalProvider,
    *,
    matched_specialties: Sequence[str] = (),
    matched_keywords: Sequence[str] = (),
) -> dict[str, float]:
    location_matches = _count_token_overlap(
        _tokenize(request.location),
        _tokenize(provider.location_summary),
    )
    specialty_matches = len(matched_specialties)
    language_matches = len(_match_values(request.preferred_languages, provider.languages))
    insurance_matches = len(_match_values(request.insurance, provider.insurance_reported))
    keyword_matches = len(matched_keywords)
    telehealth_requested = bool(_match_values(request.keywords, _TELEHEALTH_TERMS))
    accepting_verified = (
        provider.accepting_new_patients_status.verified
        and provider.accepting_new_patients_status.status.casefold() in _ACCEPTING_STATUSES
    )
    insurance_verified = provider.insurance_network_verification.verified
    freshness_present = provider.freshness is not None and (
        provider.freshness.created_epoch is not None
        or provider.freshness.last_updated_epoch is not None
    )

    return {
        "specialty_alignment": 4.0 * specialty_matches,
        "keyword_alignment": 1.75 * keyword_matches,
        "language_alignment": 1.5 * language_matches,
        "insurance_alignment": 1.5 * insurance_matches,
        "location_alignment": 1.0 if location_matches > 0 else 0.0,
        "accepting_new_patients": 0.75 if accepting_verified else 0.0,
        "insurance_verified": 0.5 if insurance_verified else 0.0,
        "telehealth_alignment": 0.5 if telehealth_requested and provider.telehealth else 0.0,
        "freshness_metadata": 0.25 if freshness_present else 0.0,
    }
def _provider_result_sort_key(result: ProviderSearchResult) -> tuple[float, int, int, str, str]:
    provider = result.provider
    score = result.score if result.score is not None else 0.0
    accepting_verified = int(
        provider.accepting_new_patients_status.verified
        and provider.accepting_new_patients_status.status.casefold() in _ACCEPTING_STATUSES
    )
    insurance_verified = int(provider.insurance_network_verification.verified)
    return (
        -score,
        -accepting_verified,
        -insurance_verified,
        provider.name.casefold(),
        provider.provider_id,
    )


def _match_values(
    requested: Iterable[str],
    available: Iterable[str],
    *,
    extra_values: Iterable[str] = (),
) -> tuple[str, ...]:
    requested_lookup = {
        value.casefold(): value
        for value in requested
        if isinstance(value, str) and value.strip()
    }
    available_values = list(available) + list(extra_values)
    matches: list[str] = []
    seen: set[str] = set()
    for value in available_values:
        if not isinstance(value, str):
            continue
        key = value.casefold()
        if key in requested_lookup and key not in seen:
            seen.add(key)
            matches.append(requested_lookup[key])
    return tuple(matches)


def _match_keywords(
    keywords: Iterable[str],
    provider: CanonicalProvider,
) -> tuple[str, ...]:
    provider_tokens = _provider_tokens(provider)
    matches: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        normalized_keyword = keyword.strip()
        if not normalized_keyword:
            continue
        tokens = _tokenize(normalized_keyword)
        if not tokens:
            continue
        if any(token in provider_tokens for token in tokens):
            lookup_key = normalized_keyword.casefold()
            if lookup_key not in seen:
                seen.add(lookup_key)
                matches.append(normalized_keyword)
    return tuple(matches)


def _provider_tokens(provider: CanonicalProvider) -> set[str]:
    tokens: set[str] = set()
    for value in (
        provider.name,
        provider.description,
        provider.location_summary,
        provider.taxonomy,
        *provider.specialties,
        *provider.languages,
        *provider.insurance_reported,
    ):
        tokens.update(_tokenize(value))
    return tokens


def _count_token_overlap(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def _tokenize(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 or token.isdigit()
    }
