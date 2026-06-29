from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
import re
from typing import Iterable, Optional, Sequence

from provider_search.models import CanonicalProvider, ProviderSearchRequest, ProviderSearchResult
from provider_search.normalization import normalize_provider, normalize_search_request, normalize_text
from provider_search.specialty_families import (
    SPECIALTY_FAMILY_BY_ID,
    derive_request_specialty_family_ids,
)


RANKING_VERSION = "deterministic-v1"
logger = logging.getLogger(__name__)

_ACCEPTING_STATUSES = {"accepting", "accepting new patients", "open"}
_TELEHEALTH_TERMS = {"telehealth", "virtual", "video", "remote", "online"}
_GENERIC_SPECIALTY_EVIDENCE_PREFIXES = ("clinic center",)

# Two-letter US state codes used for state-aware location scoring.
_US_STATE_CODES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
})


def _extract_state_code(text: Optional[str]) -> Optional[str]:
    """Return the first standalone 2-letter US state code found in *text*."""
    if not text:
        return None
    for token in re.split(r"[,\s]+", text.strip()):
        upper = token.strip().upper()
        if upper in _US_STATE_CODES:
            return upper
    return None


@dataclass(frozen=True)
class ProviderGateEvaluation:
    admitted: bool
    drop_reason: Optional[str]
    matched_specialties: tuple[str, ...]
    matched_keywords: tuple[str, ...]


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
        gate_evaluation = evaluate_provider_gate(normalized_request, provider)
        if not gate_evaluation.admitted:
            _log_gate_drop(
                reason=gate_evaluation.drop_reason or "filtered",
                request=normalized_request,
                provider=provider,
            )
            continue
        breakdown = _build_score_breakdown(
            normalized_request,
            provider,
            matched_specialties=gate_evaluation.matched_specialties,
            matched_keywords=gate_evaluation.matched_keywords,
        )
        score = round(sum(breakdown.values()), 6)
        ranking_metadata = dict(provider.ranking_metadata)
        ranking_metadata.update(
            {
                "ranking_version": RANKING_VERSION,
                "score_breakdown": breakdown,
                "cached_identity_match": provider.provider_id in cached_ids,
                "matched_specialties": gate_evaluation.matched_specialties,
                "matched_keywords": gate_evaluation.matched_keywords,
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


def evaluate_provider_gate(
    request: ProviderSearchRequest,
    provider: CanonicalProvider | dict[str, object],
) -> ProviderGateEvaluation:
    normalized_request = normalize_search_request(request)
    normalized_provider = normalize_provider(provider)
    matched_specialties = _match_specialties(normalized_request, normalized_provider)
    matched_keywords = _match_keywords(normalized_request.keywords, normalized_provider)
    if normalized_request.specialties and not matched_specialties:
        return ProviderGateEvaluation(
            admitted=False,
            drop_reason="specialty_mismatch",
            matched_specialties=matched_specialties,
            matched_keywords=matched_keywords,
        )
    if normalized_request.keywords and not normalized_request.specialties and not matched_keywords:
        return ProviderGateEvaluation(
            admitted=False,
            drop_reason="keyword_mismatch",
            matched_specialties=matched_specialties,
            matched_keywords=matched_keywords,
        )
    return ProviderGateEvaluation(
        admitted=True,
        drop_reason=None,
        matched_specialties=matched_specialties,
        matched_keywords=matched_keywords,
    )


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
    request_state = _extract_state_code(request.location)
    provider_state = (provider.state or "").strip().upper() or None
    if request_state and provider_state and request_state != provider_state:
        location_score = 0.0
    else:
        location_score = 1.0 if location_matches > 0 else 0.0
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
    specialty_specificity_bonus = _specialty_specificity_bonus(request, provider)

    return {
        "specialty_alignment": 4.0 * specialty_matches,
        "specialty_specificity": specialty_specificity_bonus,
        "keyword_alignment": 1.75 * keyword_matches,
        "language_alignment": 1.5 * language_matches,
        "insurance_alignment": 1.5 * insurance_matches,
        "location_alignment": location_score,
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
    requested_lookup = _build_requested_lookup(requested)
    available_values = list(available) + list(extra_values)
    matches: list[str] = []
    seen_keys: set[str] = set()
    seen_matches: set[str] = set()
    for value in available_values:
        if not isinstance(value, str):
            continue
        key = value.casefold()
        if key not in requested_lookup or key in seen_keys:
            continue
        seen_keys.add(key)
        matched_label = requested_lookup[key]
        matched_lookup_key = matched_label.casefold()
        if matched_lookup_key in seen_matches:
            continue
        seen_matches.add(matched_lookup_key)
        matches.append(matched_label)
    return tuple(matches)


def _build_requested_lookup(requested: Iterable[str]) -> dict[str, str]:
    requested_lookup: dict[str, str] = {}
    for value in requested:
        if not isinstance(value, str):
            continue
        normalized_value = value.strip()
        if not normalized_value:
            continue
        requested_lookup[normalized_value.casefold()] = normalized_value
    return requested_lookup


def _match_specialties(
    request: ProviderSearchRequest,
    provider: CanonicalProvider,
) -> tuple[str, ...]:
    matched_labels = list(
        _match_values(
            request.specialties,
            provider.specialties,
            extra_values=(provider.taxonomy,) if provider.taxonomy else (),
        )
    )
    seen_labels = {label.casefold() for label in matched_labels}
    provider_family_ids = set(provider.specialty_family_ids)
    if not provider_family_ids:
        return tuple(matched_labels)

    for requested_specialty in request.specialties:
        requested_lookup_key = requested_specialty.casefold()
        if requested_lookup_key in seen_labels:
            continue
        requested_family_ids = derive_request_specialty_family_ids((requested_specialty,))
        if provider_family_ids.intersection(requested_family_ids):
            seen_labels.add(requested_lookup_key)
            matched_labels.append(requested_specialty)
    return tuple(matched_labels)


def _specialty_specificity_bonus(
    request: ProviderSearchRequest,
    provider: CanonicalProvider,
) -> float:
    requested_aliases = _requested_specialty_aliases(request)
    if not requested_aliases:
        return 0.0

    for evidence in _iter_normalized_specialty_evidence(provider):
        if evidence in requested_aliases and not _is_generic_specialty_evidence(evidence):
            return 0.75
    return 0.0


def _requested_specialty_aliases(request: ProviderSearchRequest) -> set[str]:
    aliases: set[str] = set()
    for requested_specialty in request.specialties:
        normalized_specialty = _normalize_specialty_match_value(requested_specialty)
        if normalized_specialty:
            aliases.add(normalized_specialty)

    family_ids = request.specialty_family_ids or derive_request_specialty_family_ids(
        request.specialties,
    )
    for family_id in family_ids:
        family = SPECIALTY_FAMILY_BY_ID.get(family_id)
        if family is None:
            continue
        for value in (family.family_id, family.label, *family.aliases):
            normalized_value = _normalize_specialty_match_value(value)
            if normalized_value:
                aliases.add(normalized_value)
    return aliases


def _iter_normalized_specialty_evidence(provider: CanonicalProvider) -> tuple[str, ...]:
    evidence_values: list[str] = []
    seen: set[str] = set()
    for value in (*provider.specialties, provider.taxonomy):
        normalized_value = _normalize_specialty_match_value(value)
        if normalized_value is None or normalized_value in seen:
            continue
        seen.add(normalized_value)
        evidence_values.append(normalized_value)
    return tuple(evidence_values)


def _normalize_specialty_match_value(value: object) -> Optional[str]:
    normalized_value = normalize_text(value, lowercase=True)
    if normalized_value is None:
        return None
    normalized_value = normalized_value.replace("&", " and ")
    normalized_value = normalized_value.replace("/", " ")
    normalized_value = re.sub(r"[^a-z0-9]+", " ", normalized_value).strip()
    if not normalized_value:
        return None
    return normalized_value


def _is_generic_specialty_evidence(value: str) -> bool:
    return any(
        value == prefix or value.startswith(f"{prefix} ")
        for prefix in _GENERIC_SPECIALTY_EVIDENCE_PREFIXES
    )


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


def _log_gate_drop(
    *,
    reason: str,
    request: ProviderSearchRequest,
    provider: CanonicalProvider,
) -> None:
    if not _debug_enabled():
        return

    logger.info(
        "provider_search_debug_gate_drop reason=%s provider_key=%s requested_specialties=%s requested_family_ids=%s provider_family_ids=%s taxonomy=%s specialty_evidence=%s",
        reason,
        _provider_debug_key(provider.provider_id),
        tuple(request.specialties),
        tuple(request.specialty_family_ids),
        tuple(provider.specialty_family_ids),
        provider.taxonomy or "",
        tuple(provider.specialties[:6]),
    )


def _provider_debug_key(provider_id: str) -> str:
    if not provider_id:
        return "missing"
    return hashlib.sha1(str(provider_id).encode("utf-8")).hexdigest()[:8]


def _debug_enabled() -> bool:
    return (
        os.getenv("PROVIDER_SEARCH_DEBUG", "").strip() == "1"
        and os.getenv("CARE_LOCATOR_LOCAL_DEBUG", "").strip() == "1"
    )
