from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional

from provider_search.models import (
    CanonicalProvider,
    PrimitiveMetadataValue,
    ProviderSearchRequest,
    ProviderSearchResult,
    VerificationStatus,
)


INSURANCE_UNVERIFIED_BASIS = (
    "Insurance/network participation is not confirmed by source data."
)
NEW_PATIENTS_UNKNOWN_BASIS = (
    "Source data does not confirm new-patient availability."
)


def normalize_text(value: object, *, lowercase: bool = False) -> Optional[str]:
    """Collapse whitespace and normalize unicode for stable comparisons."""

    if value is None:
        return None

    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    if lowercase:
        return normalized.casefold()
    return normalized


def normalize_string_list(values: Optional[Iterable[object]]) -> tuple[str, ...]:
    """Return de-duplicated, display-safe strings while preserving first-seen order."""

    if values is None:
        return ()

    if isinstance(values, str):
        iterable: Iterable[object] = (values,)
    else:
        iterable = values

    normalized_values = []
    seen = set()
    for item in iterable:
        if item is None:
            continue
        normalized_item = normalize_text(item)
        if normalized_item is None:
            continue

        dedupe_key = normalized_item.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized_values.append(normalized_item)

    return tuple(normalized_values)


def normalize_search_request(request: ProviderSearchRequest) -> ProviderSearchRequest:
    """Normalize provider-search inputs without changing their meaning."""

    return ProviderSearchRequest(
        specialties=normalize_string_list(request.specialties),
        location=normalize_text(request.location),
        insurance=normalize_string_list(request.insurance),
        preferred_languages=normalize_string_list(request.preferred_languages),
        keywords=normalize_string_list(request.keywords),
    )


def build_request_fingerprint(request: ProviderSearchRequest) -> str:
    """Hash a normalized request so cache keys stay PHI-free and stable."""

    normalized = normalize_search_request(request)
    fingerprint_payload = {
        "specialties": [value.casefold() for value in normalized.specialties],
        "location": normalize_text(normalized.location, lowercase=True),
        "insurance": [value.casefold() for value in normalized.insurance],
        "preferred_languages": [value.casefold() for value in normalized.preferred_languages],
        "keywords": [value.casefold() for value in normalized.keywords],
    }

    serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ensure_string_list(value: Optional[Iterable[Any]]) -> tuple[str, ...]:
    return normalize_string_list(value)


def optional_string(value: Any) -> Optional[str]:
    return normalize_text(value)


def build_canonical_provider(
    *,
    provider_id: Any,
    name: Any,
    source_name: str,
    dataset: Optional[str],
    location: Any = "",
    address: Any = None,
    city: Any = None,
    state: Any = None,
    country: Any = None,
    phone: Any = None,
    website: Any = None,
    taxonomy: Any = None,
    specialties: Optional[Iterable[Any]] = None,
    languages: Optional[Iterable[Any]] = None,
    insurance_reported: Optional[Iterable[Any]] = None,
    raw: Optional[dict[str, Any]] = None,
    retrieval_metadata: Optional[dict[str, Any]] = None,
    ranking_metadata: Optional[dict[str, Any]] = None,
) -> CanonicalProvider:
    normalized_taxonomy = optional_string(taxonomy)
    normalized_specialties = list(ensure_string_list(specialties))
    if normalized_taxonomy and normalized_taxonomy not in normalized_specialties:
        normalized_specialties.append(normalized_taxonomy)

    return CanonicalProvider(
        provider_id=optional_string(provider_id) or "",
        name=optional_string(name) or "Unknown Provider",
        specialties=tuple(normalized_specialties),
        languages=ensure_string_list(languages),
        insurance_reported=ensure_string_list(insurance_reported),
        address=optional_string(address) or optional_string(location),
        city=optional_string(city),
        state=optional_string(state),
        country=optional_string(country),
        phone=optional_string(phone),
        website=optional_string(website),
        taxonomy=normalized_taxonomy,
        source=optional_string(source_name),
        insurance_network_verification=VerificationStatus(
            status="unverified",
            verified=False,
            basis=INSURANCE_UNVERIFIED_BASIS,
        ),
        accepting_new_patients_status=VerificationStatus(
            status="unknown",
            verified=False,
            basis=NEW_PATIENTS_UNKNOWN_BASIS,
        ),
        provenance={
            "source": source_name,
            "dataset": dataset,
        },
        retrieval_metadata=retrieval_metadata or {},
        ranking_metadata=ranking_metadata or {},
        raw=raw or {},
    )


def normalize_provider(raw_provider: Mapping[str, object]) -> CanonicalProvider:
    """Map raw provider payloads from different sources onto one typed model."""

    provider_id = normalize_text(
        raw_provider.get("id", raw_provider.get("provider_id", ""))
    ) or ""
    name = normalize_text(raw_provider.get("name", "Unknown Provider")) or "Unknown Provider"

    source = normalize_text(raw_provider.get("source"))
    provenance = raw_provider.get("provenance")
    if source is None and isinstance(provenance, Mapping):
        source = normalize_text(provenance.get("source"))

    telehealth_value = raw_provider.get("telehealth")
    telehealth = telehealth_value if isinstance(telehealth_value, bool) else None

    return CanonicalProvider(
        provider_id=provider_id,
        name=name,
        specialties=normalize_string_list(raw_provider.get("specialties")),
        languages=normalize_string_list(raw_provider.get("languages")),
        insurance_reported=normalize_string_list(
            raw_provider.get("insurance_reported", raw_provider.get("accepted_insurance"))
        ),
        address=normalize_text(raw_provider.get("address")),
        city=normalize_text(raw_provider.get("city")),
        state=normalize_text(raw_provider.get("state")),
        country=normalize_text(raw_provider.get("country")),
        phone=normalize_text(raw_provider.get("phone")),
        website=normalize_text(raw_provider.get("website")),
        telehealth=telehealth,
        description=normalize_text(raw_provider.get("description")),
        source=source,
        taxonomy=normalize_text(raw_provider.get("taxonomy")),
        provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
        retrieval_metadata=_normalize_metadata(raw_provider.get("retrieval_metadata")),
        ranking_metadata=_normalize_metadata(raw_provider.get("ranking_metadata")),
        raw=dict(raw_provider.get("raw")) if isinstance(raw_provider.get("raw"), Mapping) else {},
    )


def normalize_search_result(raw_result: Mapping[str, object]) -> ProviderSearchResult:
    """Normalize a retrieval result onto typed provider/result models."""

    score_value = raw_result.get("score")
    score = None
    if isinstance(score_value, (int, float)):
        score = float(score_value)

    metadata = _normalize_metadata(raw_result.get("retriever_metadata"))
    source = normalize_text(raw_result.get("source"))
    provider = normalize_provider(raw_result)

    return ProviderSearchResult(
        provider=provider,
        score=score,
        source=source or provider.source,
        retriever_metadata=metadata,
    )


def _normalize_metadata(value: object) -> dict[str, PrimitiveMetadataValue]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, PrimitiveMetadataValue] = {}
    for key, item in value.items():
        normalized_key = normalize_text(key)
        if normalized_key is None:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            normalized[normalized_key] = item

    return normalized
