from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple, Union


PrimitiveMetadataValue = Union[str, int, float, bool, None]


@dataclass(frozen=True)
class ProviderSearchRequest:
    """Normalized, PHI-free inputs for provider retrieval."""

    specialties: Tuple[str, ...] = ()
    location: Optional[str] = None
    insurance: Tuple[str, ...] = ()
    preferred_languages: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSearchRequest:
    """Source-specific request used by live API adapters."""

    search_terms: str
    limit: int
    query_filter: Optional[str] = None
    city_hint: Optional[str] = None
    zip_hint: Optional[str] = None
    state_hint: Optional[str] = None


@dataclass(frozen=True)
class SourceTrace:
    """Diagnostics for a single source call."""

    source_name: str
    dataset: Optional[str] = None
    request_url: Optional[str] = None
    request_params: Mapping[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    result_count: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class VerificationStatus:
    status: str
    verified: bool
    basis: str


@dataclass(frozen=True)
class MedicareOptOutStatus:
    opted_out: Optional[bool] = None
    optout_effective_date: Optional[str] = None
    optout_end_date: Optional[str] = None


@dataclass(frozen=True)
class CanonicalProvider:
    """Canonical provider data shared across retrieval sources."""

    provider_id: str
    name: str
    specialties: Tuple[str, ...] = ()
    languages: Tuple[str, ...] = ()
    insurance_reported: Tuple[str, ...] = ()
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    telehealth: Optional[bool] = None
    description: Optional[str] = None
    source: Optional[str] = None
    taxonomy: Optional[str] = None
    insurance_network_verification: VerificationStatus = field(
        default_factory=lambda: VerificationStatus(
            status="unverified",
            verified=False,
            basis="Insurance/network participation is not confirmed by source data.",
        )
    )
    accepting_new_patients_status: VerificationStatus = field(
        default_factory=lambda: VerificationStatus(
            status="unknown",
            verified=False,
            basis="Source data does not confirm new-patient availability.",
        )
    )
    medicare_opt_out: Optional[MedicareOptOutStatus] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    ranking_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def location_summary(self) -> Optional[str]:
        parts = [part for part in (self.address, self.city, self.state, self.country) if part]
        if not parts:
            return None
        return ", ".join(parts)

    def with_updates(self, **changes: Any) -> "CanonicalProvider":
        return replace(self, **changes)


@dataclass(frozen=True)
class NPPESRecord:
    """Normalized NPPES registry lookup payload."""

    npi: str
    practice_address: Optional[dict[str, Any]]
    mailing_address: Optional[dict[str, Any]]
    taxonomies: list[dict[str, Any]] = field(default_factory=list)
    basic: Optional[dict[str, Any]] = None
    enumeration_type: Optional[str] = None
    created_epoch: Optional[int] = None
    last_updated_epoch: Optional[int] = None
    raw_entry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSearchResult:
    """Source response with canonical providers and request trace metadata."""

    providers: list[CanonicalProvider] = field(default_factory=list)
    trace: Optional[SourceTrace] = None
    missing_location_hint: Optional[str] = None


@dataclass(frozen=True)
class ProviderSearchResult:
    """Canonical provider plus retrieval metadata."""

    provider: CanonicalProvider
    score: Optional[float] = None
    source: Optional[str] = None
    retriever_metadata: Dict[str, PrimitiveMetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSearchCacheEntry:
    """PHI-free cache record keyed by a request fingerprint."""

    cache_key: str
    request_fingerprint: str
    provider_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    stored_at: str
    expires_at: Optional[str] = None
