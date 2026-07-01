from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from typing import Any, Iterable, Optional

import requests

from provider_search.models import SourceSearchRequest, SourceSearchResult, SourceTrace
from provider_search.normalization import (
    build_canonical_provider,
    ensure_string_list,
    normalize_text,
)


@dataclass(frozen=True)
class ClinicalTablesDatasetConfig:
    search_url: str
    source_label: str
    result_fields: list[str]
    values_url: Optional[str] = None
    fields_url: Optional[str] = None
    probe_term: Optional[str] = None


DEFAULT_DATASET_CONFIGS: dict[str, ClinicalTablesDatasetConfig] = {
    "npi_idv": ClinicalTablesDatasetConfig(
        search_url="https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/search",
        source_label="NPI Registry (individual)",
        result_fields=[
            "name.full",
            "name.first",
            "name.middle",
            "name.last",
            "name.prefix",
            "name.suffix",
            "NPI",
            "provider_type",
            "taxonomies[0].desc",
            "taxonomies[0].code",
            "addr_practice.full",
            "addr_practice.address_1",
            "addr_practice.address_2",
            "addr_practice.city",
            "addr_practice.state",
            "addr_practice.zip",
            "addr_practice.country_name",
            "addr_practice.phone",
            "languages",
        ],
        probe_term="urology",
    ),
    "npi_org": ClinicalTablesDatasetConfig(
        search_url="https://clinicaltables.nlm.nih.gov/api/npi_org/v3/search",
        source_label="NPI Registry (organization)",
        result_fields=[
            "name.full",
            "NPI",
            "provider_type",
            "taxonomies[0].desc",
            "taxonomies[0].code",
            "addr_practice.full",
            "addr_practice.address_1",
            "addr_practice.address_2",
            "addr_practice.city",
            "addr_practice.state",
            "addr_practice.zip",
            "addr_practice.country_name",
            "addr_practice.phone",
            "languages",
        ],
        probe_term="clinic",
    ),
}

_TAXONOMY_CANDIDATE_FIELDS = ("provider_type", "taxonomies[0].desc")
_SPECIALTY_SEARCH_FIELDS = (
    "provider_type",
    "licenses.medicare.type",
    "licenses.taxonomy.classification",
    "licenses.taxonomy.specialization",
    "licenses.taxonomy.code",
)
_LOCATION_ALIASES = {
    "bay area": "San Francisco CA",
    "sf bay area": "San Francisco CA",
    "silicon valley": "San Jose CA",
    "south bay": "San Jose CA",
    "east bay": "Oakland CA",
    "la": "Los Angeles CA",
    "greater los angeles": "Los Angeles CA",
    "dallas fort worth": "Dallas TX",
}

# Umbrella specialty terms the NPI registry has no taxonomy for. NPI files these
# providers under specific taxonomies (Family Medicine / Internal Medicine), so
# searching the umbrella term literally returns zero results. When NPI offers no
# suggestion of its own, fall back to a representative NPI-recognized taxonomy.
_UMBRELLA_TAXONOMY_TERMS = {
    "primary care": "family medicine",
    "primary care physician": "family medicine",
    "primary care doctor": "family medicine",
    "pcp": "family medicine",
}

logger = logging.getLogger(__name__)


class ClinicalTablesSource:
    """Adapter for ClinicalTables NPI datasets."""

    def __init__(
        self,
        *,
        timeout: int = 6,
        dataset_configs: Optional[dict[str, ClinicalTablesDatasetConfig]] = None,
        session: Any = None,
        nppes_source: Optional[Any] = None,
    ) -> None:
        self.timeout = timeout
        self.dataset_configs = dataset_configs or DEFAULT_DATASET_CONFIGS
        self.session = session or requests
        self.nppes_source = nppes_source
        self.field_map: dict[str, dict[str, int]] = {
            dataset: {field: index for index, field in enumerate(config.result_fields)}
            for dataset, config in self.dataset_configs.items()
        }
        self.result_field_order: dict[str, list[str]] = {
            dataset: list(config.result_fields)
            for dataset, config in self.dataset_configs.items()
        }
        self._suggest_cache: dict[tuple[str, str, str], list[str]] = {}

    def build_search_request(
        self,
        dataset: str,
        request: SourceSearchRequest,
    ) -> tuple[str, dict[str, str]]:
        config = self.dataset_configs[dataset]
        terms = request.search_terms
        if request.specialty_driven:
            terms = self._specialty_search_terms(request) or request.search_terms
        params = {
            "terms": terms,
            "maxList": str(request.limit),
        }
        if request.query_filter:
            params["q"] = request.query_filter
        if request.specialty_driven:
            params["sf"] = ",".join(_SPECIALTY_SEARCH_FIELDS)

        field_names = self.result_field_order.get(dataset)
        if field_names:
            params["df"] = ",".join(field_names)

        return config.search_url, params

    def _specialty_search_terms(self, request: SourceSearchRequest) -> str:
        specialty_terms = self._specialty_terms_only(request)
        normalized_terms = normalize_text(specialty_terms, lowercase=True)
        if not normalized_terms:
            return ""
        if normalized_terms in {"ob/gyn", "ob-gyn", "ob gyn", "obgyn"}:
            return "obstetrics gynecology"

        punctuation_light = re.sub(r"[^a-z0-9]+", " ", normalized_terms).strip()
        if punctuation_light == "obstetrics gynecology":
            return punctuation_light
        return punctuation_light or normalized_terms

    def _specialty_terms_only(self, request: SourceSearchRequest) -> str:
        normalized_terms = normalize_text(request.search_terms)
        if not normalized_terms:
            return ""

        stripped_terms = normalized_terms
        for suffix in self._specialty_location_suffixes(request):
            if not suffix:
                continue
            suffix_pattern = rf"(?:\s+|^){re.escape(suffix)}$"
            candidate = re.sub(suffix_pattern, "", stripped_terms, flags=re.IGNORECASE).strip()
            if candidate and candidate != stripped_terms:
                stripped_terms = candidate
                break
        return stripped_terms

    def _specialty_location_suffixes(self, request: SourceSearchRequest) -> list[str]:
        normalized_city = normalize_text(request.city_hint)
        normalized_state = normalize_text(request.state_hint)
        normalized_zip = normalize_text(request.zip_hint)

        ordered_suffixes: list[str] = []
        seen: set[str] = set()
        suffix_parts = (
            (normalized_city, normalized_state, normalized_zip),
            (normalized_zip, normalized_city, normalized_state),
            (normalized_city, normalized_state),
            (normalized_state, normalized_zip),
            (normalized_city,),
            (normalized_state,),
            (normalized_zip,),
        )
        for parts in suffix_parts:
            suffix = " ".join(part for part in parts if part).strip()
            if not suffix:
                continue
            lookup_key = suffix.casefold()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            ordered_suffixes.append(suffix)
        return ordered_suffixes

    def search_dataset(
        self,
        dataset: str,
        request: SourceSearchRequest,
    ) -> SourceSearchResult:
        config = self.dataset_configs.get(dataset)
        if config is None or not request.search_terms.strip():
            return SourceSearchResult(
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    error="Dataset configuration missing or search terms empty.",
                )
            )

        url, params = self.build_search_request(dataset, request)
        self._log_scoped_request_params(dataset=dataset, request=request, params=params)

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            return SourceSearchResult(
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    request_url=url,
                    request_params=params,
                    error=str(exc),
                )
            )

        try:
            payload = response.json()
        except ValueError:
            payload = None

        fields, entries = self.parse_search_payload(dataset, payload)
        providers = []
        for row in entries[: request.limit]:
            provider = self._normalize_row(
                dataset=dataset,
                row=row,
                fields=fields,
                config=config,
                request=request,
            )
            if provider is not None:
                providers.append(provider)

        return SourceSearchResult(
            providers=providers,
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                request_url=url,
                request_params=params,
                status_code=getattr(response, "status_code", None),
                result_count=len(providers),
            ),
        )

    def _log_scoped_request_params(
        self,
        *,
        dataset: str,
        request: SourceSearchRequest,
        params: dict[str, str],
    ) -> None:
        if not self._scoped_request_debug_enabled(request.request_fingerprint):
            return

        logger.info(
            "provider_search_debug_request request_fingerprint=%s dataset=%s specialty_driven=%s limit=%s terms=%s q=%s sf=%s city_hint_present=%s state_hint_present=%s zip_hint_present=%s",
            request.request_fingerprint,
            dataset,
            request.specialty_driven,
            request.limit,
            params.get("terms", ""),
            params.get("q", ""),
            params.get("sf", ""),
            bool(request.city_hint),
            bool(request.state_hint),
            bool(request.zip_hint),
        )

    @staticmethod
    def _scoped_request_debug_enabled(request_fingerprint: Optional[str]) -> bool:
        if (
            os.getenv("PROVIDER_SEARCH_DEBUG", "").strip() != "1"
            or os.getenv("CARE_LOCATOR_LOCAL_DEBUG", "").strip() != "1"
        ):
            return False
        selector = os.getenv("PROVIDER_SEARCH_DEBUG_FINGERPRINT", "").strip()
        return bool(request_fingerprint) and bool(selector) and selector == request_fingerprint

    def suggest_specialty_terms(self, specialties: Iterable[str]) -> tuple[str, ...]:
        suggested_terms: list[str] = []
        for specialty in specialties:
            cleaned = normalize_text(specialty)
            if not cleaned:
                continue
            # Known umbrella terms take precedence: NPI's suggest endpoint echoes
            # them back (a truthy self-match that still returns zero providers),
            # so we must not defer to it for these.
            umbrella = _UMBRELLA_TAXONOMY_TERMS.get(cleaned.casefold())
            if umbrella:
                candidate = umbrella
            else:
                suggested = self._best_suggestion_across_datasets(
                    cleaned,
                    _TAXONOMY_CANDIDATE_FIELDS,
                )
                candidate = suggested or cleaned
            if candidate not in suggested_terms:
                suggested_terms.append(candidate)
        return tuple(suggested_terms)

    def build_location_assisted_terms(
        self,
        base_terms: str,
        *,
        location: Optional[str],
        city_hint: Optional[str],
        state_hint: Optional[str],
        zip_hint: Optional[str],
    ) -> list[str]:
        cleaned_base_terms = normalize_text(base_terms)
        if not cleaned_base_terms:
            return []

        variants: list[str] = []
        normalized_location, _ = self._normalize_location_alias(location or "")
        for chunk in self._split_location(normalized_location):
            normalized_chunk = normalize_text(chunk)
            if not normalized_chunk:
                continue
            candidate = f"{cleaned_base_terms} {normalized_chunk}".strip()
            if candidate not in variants:
                variants.append(candidate)

        hint_terms = " ".join(
            part for part in (zip_hint, city_hint, state_hint) if isinstance(part, str) and part.strip()
        ).strip()
        if hint_terms:
            candidate = f"{cleaned_base_terms} {hint_terms}".strip()
            if candidate not in variants:
                variants.append(candidate)

        return variants

    def _normalize_row(
        self,
        *,
        dataset: str,
        row: list[Any],
        fields: list[str],
        config: ClinicalTablesDatasetConfig,
        request: SourceSearchRequest,
    ):
        if not isinstance(row, (list, tuple)):
            return None

        data = {
            str(fields[index]): row[index]
            for index in range(min(len(fields), len(row)))
        }

        if not data:
            return self._build_fallback_provider(dataset, row, config)

        name = self._first_match(
            data,
            ["name.full", "name.last", "name.first"],
            default="Healthcare Provider",
        )
        if dataset == "npi_idv":
            first = data.get("name.first")
            middle = data.get("name.middle")
            last = data.get("name.last")
            combined = " ".join(
                str(part).strip()
                for part in (first, middle, last)
                if isinstance(part, str) and part.strip()
            )
            if combined:
                name = combined

        city = self._first_match(data, ["addr_practice.city"])
        state = self._first_match(data, ["addr_practice.state"])
        postal_code = self._first_match(data, ["addr_practice.zip"])

        if request.state_hint:
            state_value = (state or "").strip().upper()
            if state_value and state_value != request.state_hint.strip().upper():
                return None

        if request.zip_hint:
            postal_value = (postal_code or "").strip()
            if not postal_value or not postal_value.startswith(request.zip_hint.strip()):
                return None

        if request.city_hint:
            city_value = (city or "").strip().lower()
            if city_value != request.city_hint.strip().lower():
                return None

        specialty_evidence = self._collect_specialty_evidence(
            data,
            ["provider_type", "taxonomies[0].desc", "taxonomies[0].code"],
        )
        taxonomy = self._first_match(
            data,
            ["taxonomies[0].desc", "taxonomies[0].code", "provider_type"],
        )
        full_address = self._first_match(data, ["addr_practice.full"])
        address_value = full_address or self._first_match(data, ["addr_practice.address_1"])
        provider = build_canonical_provider(
            provider_id=data.get("NPI"),
            name=name,
            source_name=config.source_label,
            dataset=dataset,
            address=address_value or self._build_location_text(data),
            city=None if full_address else city,
            state=None if full_address else state,
            country=None if full_address else self._first_match(data, ["addr_practice.country_name"]),
            phone=self._first_match(data, ["addr_practice.phone"]),
            taxonomy=taxonomy,
            specialties=specialty_evidence,
            languages=ensure_string_list(data.get("languages")),
            raw=data,
            retrieval_metadata={
                "source_name": "clinicaltables",
                "dataset": dataset,
            },
        )

        if self.nppes_source and provider.provider_id.isdigit():
            return self.nppes_source.enrich_provider(provider)
        return provider

    def _build_fallback_provider(
        self,
        dataset: str,
        row: list[Any] | tuple[Any, ...],
        config: ClinicalTablesDatasetConfig,
    ):
        if not row:
            return None

        name_value = str(row[0]).strip() if row[0] else "Healthcare Provider"
        if dataset == "npi_idv" and "," in name_value:
            last, first = [part.strip() for part in name_value.split(",", 1)]
            reformatted = f"{first} {last}".strip()
            if reformatted:
                name_value = reformatted

        def _row_str(value: object) -> str:
            """Convert a row cell to string, treating None as empty."""
            if value is None:
                return ""
            return str(value).strip()

        phone = (_row_str(row[4]) or None) if len(row) > 4 else None
        taxonomy = (_row_str(row[2]) or None) if len(row) > 2 else None
        address = _row_str(row[3]) if len(row) > 3 else ""
        provider_id = row[1] if len(row) > 1 else None

        provider = build_canonical_provider(
            provider_id=provider_id,
            name=name_value,
            source_name=config.source_label,
            dataset=dataset,
            address=address,
            phone=phone,
            taxonomy=taxonomy,
            specialties=[taxonomy] if taxonomy else None,
            raw={"fallback_row": list(row)},
            retrieval_metadata={
                "source_name": "clinicaltables",
                "dataset": dataset,
                "fallback_row": True,
            },
        )
        if self.nppes_source and provider.provider_id.isdigit():
            return self.nppes_source.enrich_provider(provider)
        return provider

    def parse_search_payload(
        self,
        dataset: str,
        payload: Any,
    ) -> tuple[list[str], list[list[Any]]]:
        fields: list[str] = []
        entries: list[list[Any]] = []
        potential_fields: list[Any] | None = None
        potential_entries: list[Any] = []

        if not isinstance(payload, (list, tuple)):
            return fields, entries

        payload_items = list(payload)

        if len(payload_items) >= 4 and isinstance(payload_items[3], (list, tuple)):
            potential_entries = list(payload_items[3])
            if payload_items[2] is None:
                fallback_fields = self.result_field_order.get(dataset, [])
                if fallback_fields:
                    return list(fallback_fields), [
                        list(row) for row in potential_entries if isinstance(row, (list, tuple))
                    ]
            elif isinstance(payload_items[2], (list, tuple)):
                potential_fields = list(payload_items[2])
        elif len(payload_items) >= 3 and isinstance(payload_items[2], (list, tuple)):
            potential_entries = list(payload_items[2])
            if isinstance(payload_items[1], (list, tuple)):
                potential_fields = list(payload_items[1])

        reverse_map = {
            index: name
            for name, index in self.field_map.get(dataset, {}).items()
        }
        resolved_positions: list[int] = []
        fields = []
        if potential_fields is not None:
            for index, descriptor in enumerate(potential_fields):
                resolved_field: Optional[str] = None
                if isinstance(descriptor, str):
                    candidate = descriptor.strip()
                    if candidate:
                        resolved_field = candidate
                elif isinstance(descriptor, int):
                    resolved_field = reverse_map.get(int(descriptor))

                if not resolved_field:
                    continue

                resolved_positions.append(index)
                fields.append(resolved_field)

        if not fields:
            return fields, entries

        max_resolved_position = max(resolved_positions, default=-1)
        for row in potential_entries:
            if not isinstance(row, (list, tuple)):
                continue
            if max_resolved_position >= 0 and len(row) <= max_resolved_position:
                continue
            entries.append([row[index] for index in resolved_positions])
        return fields, entries

    @staticmethod
    def parse_fields_payload(payload: Any) -> dict[str, int]:
        mapping: dict[str, int] = {}

        if not isinstance(payload, list):
            return mapping

        entries = payload if payload and isinstance(payload[0], list) else payload[1:]
        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 2:
                continue

            field_name = entry[1]
            if not isinstance(field_name, str) or not field_name:
                continue

            index_value = entry[0]
            try:
                field_index = int(index_value)
            except (TypeError, ValueError):
                field_index = len(mapping)

            mapping[field_name] = field_index

        return mapping

    def _best_suggestion_across_datasets(
        self,
        term: str,
        candidate_fields: Iterable[str],
    ) -> str:
        cleaned = normalize_text(term)
        if not cleaned:
            return ""

        for dataset in self.dataset_configs:
            suggestions = self._suggest_for_dataset(dataset, cleaned, candidate_fields)
            if suggestions:
                return suggestions[0]

        return cleaned

    def _suggest_for_dataset(
        self,
        dataset: str,
        term: str,
        candidate_fields: Iterable[str],
    ) -> list[str]:
        config = self.dataset_configs.get(dataset)
        if config is None or not config.values_url:
            return []

        field_map = self.field_map.get(dataset, {})
        aggregated: list[str] = []
        for field in candidate_fields:
            field_index = field_map.get(field)
            if field_index is None:
                continue

            cache_key = (dataset, term, field)
            if cache_key in self._suggest_cache:
                aggregated.extend(self._suggest_cache[cache_key])
                continue

            suggestions = self._request_values(dataset, term, field_index, config.values_url)
            self._suggest_cache[cache_key] = suggestions
            aggregated.extend(suggestions)

        return list(dict.fromkeys(aggregated))

    def _request_values(
        self,
        dataset: str,
        term: str,
        field_index: int,
        values_url: str,
    ) -> list[str]:
        try:
            response = self.session.get(
                values_url,
                params={
                    "terms": term,
                    "df": str(field_index),
                    "maxList": "5",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception:
            return []

        try:
            payload = response.json()
        except ValueError:
            payload = None

        suggestions: list[str] = []
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    suggestions.extend(self._flatten_suggestions(value))
                elif isinstance(value, str):
                    suggestions.append(value)
        elif isinstance(payload, list):
            if payload and isinstance(payload[0], int):
                values = payload[1:]
            else:
                values = payload
            suggestions.extend(self._flatten_suggestions(values))

        return list(dict.fromkeys(suggestions))

    @staticmethod
    def _flatten_suggestions(values: Iterable[Any]) -> list[str]:
        flattened: list[str] = []
        for value in values:
            if isinstance(value, str):
                cleaned = normalize_text(value)
                if cleaned:
                    flattened.append(cleaned)
            elif isinstance(value, list):
                flattened.extend(ClinicalTablesSource._flatten_suggestions(value))
        return flattened

    @staticmethod
    def _normalize_location_alias(location: str) -> tuple[str, bool]:
        cleaned = (location or "").strip()
        key = cleaned.lower()
        if key in _LOCATION_ALIASES:
            return _LOCATION_ALIASES[key], True
        return cleaned, False

    @staticmethod
    def _split_location(location: str) -> list[str]:
        parts = [part.strip() for part in re.split(r"[|/,]", location) if part.strip()]
        if parts:
            return parts
        stripped = location.strip()
        return [stripped] if stripped else []

    @staticmethod
    def _first_match(
        data: dict[str, Any],
        field_names: list[str],
        *,
        default: str = "",
    ) -> str:
        for field_name in field_names:
            value = data.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    @staticmethod
    def _collect_specialty_evidence(
        data: dict[str, Any],
        field_names: list[str],
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for field_name in field_names:
            value = data.get(field_name)
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            dedupe_key = cleaned.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append(cleaned)
        return values

    @staticmethod
    def _build_location_text(data: dict[str, Any]) -> str:
        full_address = data.get("addr_practice.full")
        if isinstance(full_address, str) and full_address.strip():
            return full_address.strip()

        location_parts: list[str] = []
        street = data.get("addr_practice.address_1")
        if isinstance(street, str) and street.strip():
            location_parts.append(street.strip())

        city = data.get("addr_practice.city")
        state = data.get("addr_practice.state")
        postal = data.get("addr_practice.zip")
        locality = ", ".join(
            part.strip()
            for part in (city, state)
            if isinstance(part, str) and part.strip()
        )
        if locality and isinstance(postal, str) and postal.strip():
            locality = f"{locality} {postal.strip()}"
        elif not locality and isinstance(postal, str) and postal.strip():
            locality = postal.strip()
        if locality:
            location_parts.append(locality)

        country = data.get("addr_practice.country_name")
        if (
            isinstance(country, str)
            and country.strip()
            and country.strip().upper() not in {"US", "USA"}
        ):
            location_parts.append(country.strip())

        return ", ".join(location_parts)
