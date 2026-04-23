from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional

from provider_search.models import (
    CanonicalProvider,
    MedicareOptOutStatus,
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
    insurance_network_verification: Optional[VerificationStatus | Mapping[str, Any]] = None,
    accepting_new_patients_status: Optional[VerificationStatus | Mapping[str, Any]] = None,
    medicare_opt_out: Optional[MedicareOptOutStatus | Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    raw: Optional[dict[str, Any]] = None,
    retrieval_metadata: Optional[dict[str, Any]] = None,
    ranking_metadata: Optional[dict[str, Any]] = None,
) -> CanonicalProvider:
    normalized_taxonomy = optional_string(taxonomy)
    normalized_specialties = list(ensure_string_list(specialties))
    if normalized_taxonomy and normalized_taxonomy not in normalized_specialties:
        normalized_specialties.append(normalized_taxonomy)

    normalized_source = optional_string(source_name)
    normalized_dataset = optional_string(dataset)
    normalized_provenance = _normalize_object_dict(provenance)
    if normalized_source:
        normalized_provenance.setdefault("source", normalized_source)
    if normalized_dataset:
        normalized_provenance.setdefault("dataset", normalized_dataset)

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
        source=normalized_source,
        insurance_network_verification=_normalize_verification_status(
            insurance_network_verification,
            default_status="unverified",
            default_basis=INSURANCE_UNVERIFIED_BASIS,
            default_source=normalized_source,
        ),
        accepting_new_patients_status=_normalize_verification_status(
            accepting_new_patients_status,
            default_status="unknown",
            default_basis=NEW_PATIENTS_UNKNOWN_BASIS,
            default_source=normalized_source,
        ),
        medicare_opt_out=_normalize_medicare_opt_out_status(medicare_opt_out),
        provenance=normalized_provenance,
        retrieval_metadata=_normalize_object_dict(retrieval_metadata),
        ranking_metadata=_normalize_object_dict(ranking_metadata),
        raw=raw or {},
    )


def normalize_provider(
    raw_provider: Mapping[str, object] | CanonicalProvider,
) -> CanonicalProvider:
    """Map raw provider payloads from different sources onto one typed model."""

    if isinstance(raw_provider, CanonicalProvider):
        return CanonicalProvider(
            provider_id=optional_string(raw_provider.provider_id) or "",
            name=optional_string(raw_provider.name) or "Unknown Provider",
            specialties=ensure_string_list(raw_provider.specialties),
            languages=ensure_string_list(raw_provider.languages),
            insurance_reported=ensure_string_list(raw_provider.insurance_reported),
            address=optional_string(raw_provider.address),
            city=optional_string(raw_provider.city),
            state=optional_string(raw_provider.state),
            country=optional_string(raw_provider.country),
            phone=optional_string(raw_provider.phone),
            website=optional_string(raw_provider.website),
            telehealth=raw_provider.telehealth if isinstance(raw_provider.telehealth, bool) else None,
            description=optional_string(raw_provider.description),
            source=optional_string(raw_provider.source),
            taxonomy=optional_string(raw_provider.taxonomy),
            insurance_network_verification=_normalize_verification_status(
                raw_provider.insurance_network_verification,
                default_status="unverified",
                default_basis=INSURANCE_UNVERIFIED_BASIS,
                default_source=optional_string(raw_provider.source),
            ),
            accepting_new_patients_status=_normalize_verification_status(
                raw_provider.accepting_new_patients_status,
                default_status="unknown",
                default_basis=NEW_PATIENTS_UNKNOWN_BASIS,
                default_source=optional_string(raw_provider.source),
            ),
            medicare_opt_out=_normalize_medicare_opt_out_status(raw_provider.medicare_opt_out),
            provenance=_normalize_object_dict(raw_provider.provenance),
            retrieval_metadata=_normalize_object_dict(raw_provider.retrieval_metadata),
            ranking_metadata=_normalize_object_dict(raw_provider.ranking_metadata),
            raw=dict(raw_provider.raw),
        )

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
        insurance_network_verification=_normalize_verification_status(
            raw_provider.get("insurance_network_verification"),
            default_status="unverified",
            default_basis=INSURANCE_UNVERIFIED_BASIS,
            default_source=source,
        ),
        accepting_new_patients_status=_normalize_verification_status(
            raw_provider.get("accepting_new_patients_status"),
            default_status="unknown",
            default_basis=NEW_PATIENTS_UNKNOWN_BASIS,
            default_source=source,
        ),
        medicare_opt_out=_normalize_medicare_opt_out_status(
            raw_provider.get("medicare_opt_out")
        ),
        provenance=_normalize_object_dict(provenance),
        retrieval_metadata=_normalize_object_dict(raw_provider.get("retrieval_metadata")),
        ranking_metadata=_normalize_object_dict(raw_provider.get("ranking_metadata")),
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
    provider_payload = raw_result.get("provider")
    if isinstance(provider_payload, (Mapping, CanonicalProvider)):
        provider = normalize_provider(provider_payload)
    else:
        provider = normalize_provider(raw_result)

    return ProviderSearchResult(
        provider=provider,
        score=score,
        source=source or provider.source,
        retriever_metadata=metadata,
    )


def _normalize_verification_status(
    value: object,
    *,
    default_status: str,
    default_basis: str,
    default_source: Optional[str] = None,
) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return VerificationStatus(
            status=optional_string(value.status) or default_status,
            verified=value.verified if isinstance(value.verified, bool) else False,
            basis=optional_string(value.basis) or default_basis,
            source=optional_string(value.source) or default_source,
        )

    if isinstance(value, Mapping):
        verified_value = value.get("verified")
        verified = verified_value if isinstance(verified_value, bool) else False
        return VerificationStatus(
            status=optional_string(value.get("status")) or default_status,
            verified=verified,
            basis=optional_string(value.get("basis")) or default_basis,
            source=optional_string(value.get("source")) or default_source,
        )

    return VerificationStatus(
        status=default_status,
        verified=False,
        basis=default_basis,
        source=default_source,
    )


def _normalize_medicare_opt_out_status(
    value: object,
) -> Optional[MedicareOptOutStatus]:
    if isinstance(value, MedicareOptOutStatus):
        return MedicareOptOutStatus(
            opted_out=value.opted_out if isinstance(value.opted_out, bool) else None,
            optout_effective_date=optional_string(value.optout_effective_date),
            optout_end_date=optional_string(value.optout_end_date),
        )

    if not isinstance(value, Mapping):
        return None

    opted_out_value = value.get("opted_out")
    opted_out = opted_out_value if isinstance(opted_out_value, bool) else None
    return MedicareOptOutStatus(
        opted_out=opted_out,
        optout_effective_date=optional_string(value.get("optout_effective_date")),
        optout_end_date=optional_string(value.get("optout_end_date")),
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


def _normalize_object_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = normalize_text(key)
        if normalized_key is None:
            continue
        normalized[normalized_key] = _normalize_object_value(item)
    return normalized


def _normalize_object_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _normalize_object_dict(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_object_value(item) for item in value]
    if isinstance(value, str):
        return optional_string(value)
    return value
