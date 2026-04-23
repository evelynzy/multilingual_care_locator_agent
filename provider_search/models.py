from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union


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

    @property
    def location_summary(self) -> Optional[str]:
        parts = [part for part in (self.address, self.city, self.state, self.country) if part]
        if not parts:
            return None
        return ", ".join(parts)


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
