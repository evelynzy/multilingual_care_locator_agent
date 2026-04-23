from __future__ import annotations

from typing import Any, Optional

import requests

from provider_search.models import CanonicalProvider, FreshnessMetadata, NPPESRecord


class NPPESSource:
    """Adapter for the NPPES NPI registry lookup API."""

    def __init__(
        self,
        *,
        lookup_url: str = "https://npiregistry.cms.hhs.gov/api/",
        version: str = "2.1",
        timeout: int = 6,
        session: Any = None,
    ) -> None:
        self.lookup_url = lookup_url
        self.version = version
        self.timeout = timeout
        self.session = session or requests
        self._cache: dict[str, Optional[NPPESRecord]] = {}

    def build_lookup_request(self, npi: str) -> tuple[str, dict[str, str]]:
        return (
            self.lookup_url,
            {
                "number": str(npi),
                "version": self.version,
                "limit": "1",
            },
        )

    def lookup(self, npi: str) -> Optional[NPPESRecord]:
        normalized_npi = str(npi).strip()
        if normalized_npi in self._cache:
            return self._cache[normalized_npi]

        if not normalized_npi.isdigit():
            self._cache[normalized_npi] = None
            return None

        url, params = self.build_lookup_request(normalized_npi)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except Exception:
            return None

        try:
            payload = response.json()
        except ValueError:
            return None

        record = self.parse_payload(normalized_npi, payload)
        if record is not None or self._is_cacheable_negative_payload(payload):
            self._cache[normalized_npi] = record
        return record

    def parse_payload(self, npi: str, payload: Any) -> Optional[NPPESRecord]:
        if not isinstance(payload, dict):
            return None

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None

        entry = results[0]
        if not isinstance(entry, dict):
            return None

        taxonomies = entry.get("taxonomies")
        return NPPESRecord(
            npi=npi,
            practice_address=self.select_address(entry.get("addresses"), target="LOCATION"),
            mailing_address=self.select_address(entry.get("addresses"), target="MAILING"),
            taxonomies=[item for item in taxonomies if isinstance(item, dict)]
            if isinstance(taxonomies, list)
            else [],
            basic=entry.get("basic") if isinstance(entry.get("basic"), dict) else None,
            enumeration_type=self._clean_string(entry.get("enumeration_type")),
            created_epoch=entry.get("created_epoch"),
            last_updated_epoch=entry.get("last_updated_epoch"),
            raw_entry=entry,
        )

    def enrich_provider(self, provider: CanonicalProvider) -> CanonicalProvider:
        if not provider.provider_id or not provider.provider_id.isdigit():
            return provider

        record = self.lookup(provider.provider_id)
        if record is None:
            return provider

        location_override = self.format_location(record.practice_address)
        phone_value = None
        if isinstance(record.practice_address, dict):
            phone_value = record.practice_address.get("telephone_number")
        if not phone_value and isinstance(record.mailing_address, dict):
            phone_value = record.mailing_address.get("telephone_number")

        taxonomy = provider.taxonomy or self.first_taxonomy_description(record.taxonomies)

        updated_raw = dict(provider.raw)
        updated_raw["nppes"] = {
            "npi": record.npi,
            "lookup": record.raw_entry,
        }

        updated_retrieval = dict(provider.retrieval_metadata)
        updated_retrieval["nppes_enriched"] = True
        updated_retrieval["nppes"] = {
            "created_epoch": record.created_epoch,
            "last_updated_epoch": record.last_updated_epoch,
        }

        updated_specialties = list(provider.specialties)
        if taxonomy and taxonomy not in updated_specialties:
            updated_specialties.append(taxonomy)

        return provider.with_updates(
            address=location_override or provider.address,
            city=None if location_override else provider.city,
            state=None if location_override else provider.state,
            country=None if location_override else provider.country,
            phone=self._clean_string(phone_value) or provider.phone,
            taxonomy=taxonomy,
            specialties=tuple(updated_specialties),
            freshness=FreshnessMetadata(
                source="NPPES Registry",
                dataset="nppes",
                created_epoch=record.created_epoch,
                last_updated_epoch=record.last_updated_epoch,
            ),
            raw=updated_raw,
            retrieval_metadata=updated_retrieval,
        )

    @staticmethod
    def select_address(addresses: Any, *, target: str) -> Optional[dict[str, Any]]:
        if not isinstance(addresses, list):
            return None

        fallback: Optional[dict[str, Any]] = None
        for entry in addresses:
            if not isinstance(entry, dict):
                continue
            if fallback is None:
                fallback = entry
            purpose = entry.get("address_purpose")
            if isinstance(purpose, str) and purpose.strip().upper() == target.upper():
                return entry
        return fallback

    @staticmethod
    def format_location(address: Optional[dict[str, Any]]) -> str:
        if not isinstance(address, dict):
            return ""

        parts: list[str] = []
        for key in ("address_1", "address_2"):
            value = address.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())

        city = address.get("city")
        state = address.get("state")
        locality_parts = [
            value.strip()
            for value in (city, state)
            if isinstance(value, str) and value.strip()
        ]
        locality = ", ".join(locality_parts)

        postal = address.get("postal_code")
        if isinstance(postal, str) and postal.strip():
            if locality:
                locality = f"{locality} {postal.strip()}"
            else:
                locality = postal.strip()
        if locality:
            parts.append(locality)

        country = address.get("country_name") or address.get("country_code")
        if (
            isinstance(country, str)
            and country.strip()
            and country.strip().upper() not in {"US", "USA", "UNITED STATES"}
        ):
            parts.append(country.strip())

        return ", ".join(parts)

    @staticmethod
    def first_taxonomy_description(taxonomies: list[dict[str, Any]]) -> Optional[str]:
        for entry in taxonomies:
            for key in ("desc", "code"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _clean_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _is_cacheable_negative_payload(payload: Any) -> bool:
        return isinstance(payload, dict) and isinstance(payload.get("results"), list)
