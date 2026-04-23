from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional

from provider_search.models import (
    CanonicalProvider,
    FreshnessMetadata,
    MedicareOptOutStatus,
    ProviderSearchRequest,
    ProviderSearchResult,
    StructuredMetadataValue,
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
    freshness: Optional[FreshnessMetadata | Mapping[str, Any]] = None,
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
        freshness=_normalize_freshness_metadata(
            freshness,
            default_source=normalized_source,
            default_dataset=normalized_dataset,
            retrieval_metadata=retrieval_metadata,
            raw=raw,
        ),
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
            freshness=_normalize_freshness_metadata(
                raw_provider.freshness,
                default_source=optional_string(raw_provider.source),
                default_dataset=_extract_dataset(
                    raw_provider.provenance,
                    raw_provider.retrieval_metadata,
                ),
                retrieval_metadata=raw_provider.retrieval_metadata,
                raw=raw_provider.raw,
            ),
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
    retrieval_metadata = raw_provider.get("retrieval_metadata")
    ranking_metadata = raw_provider.get("ranking_metadata")
    dataset = _extract_dataset(provenance, retrieval_metadata, raw_provider)

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
        freshness=_normalize_freshness_metadata(
            raw_provider.get("freshness", raw_provider.get("freshness_metadata")),
            default_source=source,
            default_dataset=dataset,
            retrieval_metadata=retrieval_metadata,
            raw=raw_provider,
        ),
        provenance=_normalize_object_dict(provenance),
        retrieval_metadata=_normalize_object_dict(retrieval_metadata),
        ranking_metadata=_normalize_object_dict(ranking_metadata),
        raw=dict(raw_provider.get("raw")) if isinstance(raw_provider.get("raw"), Mapping) else {},
    )


def normalize_search_result(raw_result: Mapping[str, object]) -> ProviderSearchResult:
    """Normalize a retrieval result onto typed provider/result models."""

    score_value = raw_result.get("score")
    score = None
    if isinstance(score_value, (int, float)):
        score = float(score_value)

    metadata = _normalize_object_dict(raw_result.get("retriever_metadata"))
    source = normalize_text(raw_result.get("source"))
    provider_payload = raw_result.get("provider")
    if isinstance(provider_payload, (Mapping, CanonicalProvider)):
        provider = _merge_provider_context(
            primary=normalize_provider(provider_payload),
            fallback=normalize_provider(raw_result),
        )
    else:
        provider = normalize_provider(raw_result)

    return ProviderSearchResult(
        provider=provider,
        score=score,
        source=source or provider.source,
        retriever_metadata=metadata,
    )


def _merge_provider_context(
    *,
    primary: CanonicalProvider,
    fallback: CanonicalProvider,
) -> CanonicalProvider:
    return primary.with_updates(
        source=primary.source or fallback.source,
        insurance_network_verification=_select_verification_status(
            primary.insurance_network_verification,
            fallback.insurance_network_verification,
            default_status="unverified",
            default_basis=INSURANCE_UNVERIFIED_BASIS,
        ),
        accepting_new_patients_status=_select_verification_status(
            primary.accepting_new_patients_status,
            fallback.accepting_new_patients_status,
            default_status="unknown",
            default_basis=NEW_PATIENTS_UNKNOWN_BASIS,
        ),
        medicare_opt_out=primary.medicare_opt_out or fallback.medicare_opt_out,
        freshness=_merge_freshness_metadata(primary.freshness, fallback.freshness),
        provenance=_merge_object_dicts(fallback.provenance, primary.provenance),
        retrieval_metadata=_merge_object_dicts(
            fallback.retrieval_metadata,
            primary.retrieval_metadata,
        ),
        ranking_metadata=_merge_object_dicts(
            fallback.ranking_metadata,
            primary.ranking_metadata,
        ),
        raw=_merge_object_dicts(fallback.raw, primary.raw),
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


def _normalize_freshness_metadata(
    value: object,
    *,
    default_source: Optional[str] = None,
    default_dataset: Optional[str] = None,
    retrieval_metadata: object = None,
    raw: object = None,
) -> Optional[FreshnessMetadata]:
    if isinstance(value, FreshnessMetadata):
        return FreshnessMetadata(
            source=optional_string(value.source) or default_source,
            dataset=optional_string(value.dataset) or default_dataset,
            created_epoch=_coerce_optional_int(value.created_epoch),
            last_updated_epoch=_coerce_optional_int(value.last_updated_epoch),
        )

    direct_created_epoch = _extract_epoch("created_epoch", value, retrieval_metadata, raw)
    direct_last_updated_epoch = _extract_epoch(
        "last_updated_epoch",
        value,
        retrieval_metadata,
        raw,
    )

    if isinstance(value, Mapping):
        source = optional_string(value.get("source")) or default_source
        dataset = optional_string(value.get("dataset")) or default_dataset
    else:
        source = default_source
        dataset = default_dataset

    if direct_created_epoch is None and direct_last_updated_epoch is None:
        return None

    return FreshnessMetadata(
        source=source,
        dataset=dataset,
        created_epoch=direct_created_epoch,
        last_updated_epoch=direct_last_updated_epoch,
    )


def _normalize_object_dict(value: object) -> dict[str, StructuredMetadataValue]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, StructuredMetadataValue] = {}
    for key, item in value.items():
        normalized_key = normalize_text(key)
        if normalized_key is None:
            continue
        normalized[normalized_key] = _normalize_object_value(item)
    return normalized


def _normalize_object_value(value: object) -> StructuredMetadataValue:
    if isinstance(value, Mapping):
        return _normalize_object_dict(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_object_value(item) for item in value]
    if isinstance(value, str):
        return optional_string(value)
    return value


def _merge_object_dicts(
    fallback: Mapping[str, StructuredMetadataValue],
    primary: Mapping[str, StructuredMetadataValue],
) -> dict[str, StructuredMetadataValue]:
    merged = dict(fallback)
    for key, value in primary.items():
        fallback_value = merged.get(key)
        if isinstance(fallback_value, dict) and isinstance(value, dict):
            merged[key] = _merge_object_dicts(fallback_value, value)
        else:
            merged[key] = value
    return merged


def _select_verification_status(
    primary: VerificationStatus,
    fallback: VerificationStatus,
    *,
    default_status: str,
    default_basis: str,
) -> VerificationStatus:
    if _is_verification_status_informative(
        primary,
        default_status=default_status,
        default_basis=default_basis,
    ):
        return primary
    if _is_verification_status_informative(
        fallback,
        default_status=default_status,
        default_basis=default_basis,
    ):
        return fallback
    return primary


def _is_verification_status_informative(
    value: VerificationStatus,
    *,
    default_status: str,
    default_basis: str,
) -> bool:
    return (
        value.verified
        or value.status != default_status
        or value.basis != default_basis
    )


def _merge_freshness_metadata(
    primary: Optional[FreshnessMetadata],
    fallback: Optional[FreshnessMetadata],
) -> Optional[FreshnessMetadata]:
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return FreshnessMetadata(
        source=primary.source or fallback.source,
        dataset=primary.dataset or fallback.dataset,
        created_epoch=primary.created_epoch
        if primary.created_epoch is not None
        else fallback.created_epoch,
        last_updated_epoch=primary.last_updated_epoch
        if primary.last_updated_epoch is not None
        else fallback.last_updated_epoch,
    )


def _extract_dataset(*values: object) -> Optional[str]:
    for value in values:
        if isinstance(value, Mapping):
            dataset = optional_string(value.get("dataset"))
            if dataset is not None:
                return dataset
    return None


def _extract_epoch(field_name: str, *values: object) -> Optional[int]:
    for value in values:
        if isinstance(value, Mapping):
            direct_value = _coerce_optional_int(value.get(field_name))
            if direct_value is not None:
                return direct_value

            nested_freshness = value.get("freshness")
            if isinstance(nested_freshness, Mapping):
                nested_value = _coerce_optional_int(nested_freshness.get(field_name))
                if nested_value is not None:
                    return nested_value

            nppes = value.get("nppes")
            if isinstance(nppes, Mapping):
                nppes_direct_value = _coerce_optional_int(nppes.get(field_name))
                if nppes_direct_value is not None:
                    return nppes_direct_value
                lookup = nppes.get("lookup")
                if isinstance(lookup, Mapping):
                    nppes_value = _coerce_optional_int(lookup.get(field_name))
                    if nppes_value is not None:
                        return nppes_value
    return None


def _coerce_optional_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None
