from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging

import requests
from huggingface_hub import InferenceClient

from config_loader import get_prompt, get_search_settings
from retriever import ProviderRepository, SearchCriteria

logger = logging.getLogger(__name__)


_US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

_US_STATE_NAMES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


_LOCATION_NOISE_TOKENS = {
    "plan",
    "insurance",
    "coverage",
    "policy",
    "need",
    "looking",
    "search",
    "find",
    "for",
    "provider",
    "doctor",
    "dentist",
    "dental",
    "care",
    "help",
    "support",
    "area",
    "region",
    "state",
    "province",
}


@dataclass
class ParsedCareQuery:
    """Structured representation of the user's care request."""

    detected_language: str
    response_language: str
    summary: str
    medical_need: bool
    location: Optional[str]
    specialties: List[str]
    insurance: List[str]
    preferred_languages: List[str]
    keywords: List[str]
    patient_context: Optional[str]


class CareLocatorAgent:
    """Coordinates LLM reasoning, dataset search, and fallback lookups."""

    def __init__(self, provider_repository: Optional[ProviderRepository] = None) -> None:
        self.provider_repository = provider_repository or ProviderRepository()
        self.prompts = {
            "interpret": get_prompt("interpret_user_need"),
            "response_system": get_prompt("response_system"),
            "response_template": get_prompt("response_user_template"),
            "response_template_fallback_only": get_prompt(
                "response_user_template_fallback_only"
            ),
            "response_template_location_needed": get_prompt(
                "response_user_template_location_needed"
            ),
        }
        self.search_settings = get_search_settings()
        clinical = self.search_settings.get("clinicaltables", {})
        field_probe_terms = clinical.get("field_probe_terms", {}) or {}
        self.ctss_timeout = clinical.get("timeout", 6)
        self.ctss_max_results = clinical.get("max_results", 3)
        self.ctss_values_max_results = clinical.get("values_max_results", 10)

        self.ctss_dataset_configs: Dict[str, Dict[str, Any]] = {
            "npi_idv": {
                "search_url": clinical.get(
                    "individual_search_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/search",
                ),
                "values_url": clinical.get(
                    "individual_values_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/values",
                ),
                "fields_url": clinical.get(
                    "individual_fields_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/fields",
                ),
                "probe_term": field_probe_terms.get("npi_idv", "urology"),
                "source_label": "NPI Registry (individual)",
                "result_fields": [
                    "Provider Last Name (Legal Name)",
                    "Provider First Name",
                    "Provider Middle Name",
                    "Provider Business Practice Location Address Line 1",
                    "NPI",
                    "Provider Language Description_1",
                    "Provider Business Practice Location Address City Name",
                    "Provider Business Practice Location Address State Name",
                    "Provider Business Practice Location Address Postal Code",
                    "Provider Business Practice Location Address Telephone Number",
                    "Provider Taxonomy Description_1",
                    "Healthcare Provider Taxonomy Code_1",
                    "Provider Business Practice Location Address Country Code (If outside U.S.)",
                ],
            },
            "npi_org": {
                "search_url": clinical.get(
                    "organization_search_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/search",
                ),
                "values_url": clinical.get(
                    "organization_values_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/values",
                ),
                "fields_url": clinical.get(
                    "organization_fields_url",
                    "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/fields",
                ),
                "probe_term": field_probe_terms.get("npi_org", "clinic"),
                "source_label": "NPI Registry (organization)",
                "result_fields": [
                    "Provider Organization Name (Legal Business Name)",
                    "Provider Business Practice Location Address Line 1",
                    "NPI",
                    "Authorized Official Title or Position",
                    "Provider Business Practice Location Address City Name",
                    "Provider Business Practice Location Address State Name",
                    "Provider Business Practice Location Address Postal Code",
                    "Provider Business Practice Location Address Telephone Number",
                    "Provider Taxonomy Description_1",
                    "Provider Language Description_1",
                    "Healthcare Provider Taxonomy Code_1",
                ],
            },
        }
        self._ctss_dataset_priority: List[str] = [
            dataset
            for dataset, cfg in self.ctss_dataset_configs.items()
            if cfg.get("search_url")
        ]

        self.fallback_resources = self.search_settings.get("fallback_resources", [])
        self._ctss_field_map: Dict[str, Dict[str, int]] = {}
        self._ctss_result_field_indexes: Dict[str, List[int]] = {}
        self._ctss_result_field_order: Dict[str, List[str]] = {}
        self._ctss_suggest_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._ctss_taxonomy_fields: List[str] = [
            "Provider Taxonomy Description_1",
            "Healthcare Provider Taxonomy Code_1",
        ]
        self._ctss_location_fields: List[str] = [
            "Provider Business Practice Location Address City Name",
            "Provider Business Practice Location Address State Name",
            "Provider Business Practice Location Address Postal Code",
            "Provider Business Practice Location Address Country Code (If outside U.S.)",
            "Provider Business Mailing Address City Name",
            "Provider Business Mailing Address State Name",
            "Provider Business Mailing Address Postal Code",
        ]
        self._location_aliases: Dict[str, str] = {
            "bay area": "San Francisco CA",
            "sf bay area": "San Francisco CA",
            "silicon valley": "San Jose CA",
            "south bay": "San Jose CA",
            "east bay": "Oakland CA",
            "la": "Los Angeles CA",
            "greater los angeles": "Los Angeles CA",
            "dallas fort worth": "Dallas TX",
        }

        state_code_pattern = "|".join(sorted(_US_STATE_CODES))
        state_name_pattern = "|".join(
            re.escape(name)
            for name in sorted(_US_STATE_NAMES.keys(), key=len, reverse=True)
        )
        city_pattern = r"(?P<city>(?-i:[A-Z])[A-Za-z'.-]{2,}(?:\s+(?-i:[A-Z])[A-Za-z'.-]{2,})*)"
        separator_pattern = r"\s*,?\s+"
        self._city_state_code_regex = re.compile(
            rf"{city_pattern}{separator_pattern}(?P<state>{state_code_pattern})\b",
            re.IGNORECASE,
        )
        self._city_state_name_regex = re.compile(
            rf"{city_pattern}{separator_pattern}(?P<state>{state_name_pattern})\b",
            re.IGNORECASE,
        )

        self._initialize_clinicaltables_field_maps()

    # ------------------------------------------------------------------
    def handle_request(
        self,
        client: InferenceClient,
        message: str,
        history: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        logger.info("Handling request. message_length=%s history_turns=%s", len(message), len(history))
        parsed_query = self._interpret_user_need(client, message, history)

        if history:
            latest_only_query = self._interpret_user_need(client, message, [])
            parsed_query = self._merge_parsed_queries(parsed_query, latest_only_query)

        local_results: List[dict] = []
        fallback_results: List[dict] = []

        missing_location_hint = None

        if parsed_query.medical_need:
            search_criteria = SearchCriteria(
                specialties=parsed_query.specialties,
                location=parsed_query.location,
                insurance=parsed_query.insurance,
                preferred_languages=parsed_query.preferred_languages,
                keywords=parsed_query.keywords,
            )

            local_results = self.provider_repository.search(search_criteria)
            logger.info("Local semantic search results=%s", len(local_results))

            if not local_results:
                fallback_results, missing_location_hint = self._search_clinicaltables(
                    parsed_query,
                    limit=self.ctss_max_results,
                )
                logger.info(
                    "ClinicalTables fallback results=%s", len(fallback_results)
                )
                if not fallback_results and not missing_location_hint:
                    fallback_results = self._trusted_resource_fallback(parsed_query)
                    logger.info(
                        "Trusted resource fallback results=%s", len(fallback_results)
                    )
        else:
            logger.info("Parsed request marked as non-medical; skipping provider search")

        no_results_found = (
            parsed_query.medical_need
            and not local_results
            and not fallback_results
            and not missing_location_hint
        )

        response_payload = {
            "query": {
                "detected_language": parsed_query.detected_language,
                "response_language": parsed_query.response_language,
                "summary": parsed_query.summary,
                "medical_need": parsed_query.medical_need,
                "location": parsed_query.location,
                "specialties": parsed_query.specialties,
                "insurance": parsed_query.insurance,
                "preferred_languages": parsed_query.preferred_languages,
                "keywords": parsed_query.keywords,
                "patient_context": parsed_query.patient_context,
            },
            "local_results": local_results,
            "fallback_results": fallback_results,
        }

        fallback_only_mode = self._should_use_fallback_only_template()

        if missing_location_hint:
            response_payload.setdefault("notes", missing_location_hint)
        elif no_results_found:
            response_payload.setdefault(
                "notes",
                "No providers were found in the local dataset or via the ClinicalTables API."
            )

        if self.provider_repository.load_error and not fallback_only_mode:
            response_payload["notes"] = (
                "Local dataset fallback was used because loading the configured Hugging Face "
                f"dataset failed: {self.provider_repository.load_error}"
            )

        if missing_location_hint:
            template_key = "response_template_location_needed"
            fallback_only_mode = False
        else:
            template_key = (
                "response_template_fallback_only"
                if fallback_only_mode
                else "response_template"
            )

        return self._compose_response(
            client,
            response_payload,
            max_tokens,
            temperature,
            top_p,
            template_key=template_key,
        )

    # ------------------------------------------------------------------
    def _interpret_user_need(
        self, client: InferenceClient, message: str, history: List[Dict[str, str]]
    ) -> ParsedCareQuery:
        guidance = self.prompts.get("interpret") or (
            "You are a healthcare triage analyst. Given a user request in any language, "
            "extract structured search criteria for finding care providers. Respond with "
            "strict JSON describing detected language, response language, summary, a boolean medical_need, location, "
            "specialties, insurance, preferred languages, keywords, and patient context."
        )

        logger.debug("Interpret prompt=%s", guidance)

        messages = (
            [{"role": "system", "content": guidance}]
            + history
            + [{"role": "user", "content": message}]
        )

        completion = client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.0,
            top_p=0.1,
        )

        content = completion.choices[0].message["content"] if completion.choices else ""
        logger.debug("Interpret response=%s", content)
        parsed_payload = self._safe_json_parse(content)

        if not parsed_payload:
            logger.warning("Interpret response did not yield valid JSON; retrying with clarification prompt")
            retry_messages = (
                messages
                + [{"role": "assistant", "content": content}]
                + [
                    {
                        "role": "user",
                        "content": (
                            "Please repeat the previous JSON using valid strict JSON syntax only. "
                            "Do not include explanations—return just the JSON object."
                        ),
                    }
                ]
            )
            retry_completion = client.chat_completion(
                retry_messages,
                max_tokens=512,
                temperature=0.0,
                top_p=0.1,
            )
            retry_content = (
                retry_completion.choices[0].message["content"]
                if retry_completion.choices
                else ""
            )
            logger.debug("Interpret retry response=%s", retry_content)
            parsed_payload = self._safe_json_parse(retry_content)

        if not parsed_payload:
            parsed_payload = {
                "detected_language": "unknown",
                "response_language": "English",
                "summary": message,
                "medical_need": True,
                "location": None,
                "specialties": [],
                "insurance": [],
                "preferred_languages": [],
                "keywords": [],
                "patient_context": None,
            }

        detected_language = str(parsed_payload.get("detected_language", "unknown"))
        response_language = parsed_payload.get("response_language") or detected_language or "English"

        if (
            response_language
            and detected_language
            and response_language.lower().strip() == "english"
            and detected_language.lower().strip() != "english"
        ):
            response_language = detected_language

        return ParsedCareQuery(
            detected_language=detected_language,
            response_language=str(response_language or "English"),
            summary=str(parsed_payload.get("summary", "")),
            medical_need=bool(parsed_payload.get("medical_need", True)),
            location=parsed_payload.get("location"),
            specialties=self._ensure_list(parsed_payload.get("specialties")),
            insurance=self._ensure_list(parsed_payload.get("insurance")),
            preferred_languages=self._ensure_list(parsed_payload.get("preferred_languages")),
            keywords=self._ensure_list(parsed_payload.get("keywords")),
            patient_context=parsed_payload.get("patient_context"),
        )

    # ------------------------------------------------------------------
    def _merge_parsed_queries(
        self, full: ParsedCareQuery, latest: ParsedCareQuery
    ) -> ParsedCareQuery:
        def _prefer(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
            return primary if primary not in (None, "", "unknown") else fallback

        def _merge_lists(primary: List[str], fallback: List[str]) -> List[str]:
            merged: List[str] = []
            for item in primary + fallback:
                if item not in merged:
                    merged.append(item)
            return merged

        detected_language = _prefer(latest.detected_language, full.detected_language)
        response_language = _prefer(latest.response_language, full.response_language)

        specialties = latest.specialties or full.specialties
        location = latest.location or full.location
        insurance = latest.insurance or full.insurance
        preferred_languages = _merge_lists(latest.preferred_languages, full.preferred_languages)
        keywords = _merge_lists(latest.keywords, full.keywords)
        patient_context = latest.patient_context or full.patient_context
        summary = latest.summary or full.summary
        medical_need = latest.medical_need if latest.medical_need is not None else full.medical_need

        return ParsedCareQuery(
            detected_language=detected_language or full.detected_language,
            response_language=response_language or full.response_language,
            summary=summary,
            medical_need=medical_need,
            location=location,
            specialties=specialties,
            insurance=insurance,
            preferred_languages=preferred_languages,
            keywords=keywords,
            patient_context=patient_context,
        )

    # ------------------------------------------------------------------
    def _search_clinicaltables(
        self, query: ParsedCareQuery, limit: int = 3
    ) -> tuple[List[dict], Optional[str]]:
        (
            search_terms,
            query_filter,
            location_was_specific,
            city_hint,
            zip_hint,
            state_hint,
        ) = self._compose_clinicaltables_query_parts(query)
        logger.debug(
            "ClinicalTables combined search_terms=%s filter=%s",
            search_terms,
            query_filter,
        )

        if not search_terms.strip():
            return [], None

        if not location_was_specific:
            return [], self._location_hint()

        results: List[dict] = []
        per_dataset_limit = max(limit, self.ctss_max_results)

        for dataset in self._ctss_dataset_priority:
            dataset_results = self._clinicaltables_search_dataset(
                dataset,
                search_terms,
                query_filter,
                per_dataset_limit,
                city_hint=city_hint,
                zip_hint=zip_hint,
                state_hint=state_hint,
            )
            for record in dataset_results:
                results.append(record)
                if len(results) >= limit:
                    return results, None

        if results:
            return results, None

        return results, None

    # ------------------------------------------------------------------
    def _clinicaltables_search_dataset(
        self,
        dataset: str,
        search_terms: str,
        query_filter: Optional[str],
        limit: int,
        *,
        city_hint: Optional[str] = None,
        zip_hint: Optional[str] = None,
        state_hint: Optional[str] = None,
    ) -> List[dict]:
        config = self.ctss_dataset_configs.get(dataset, {})
        url = config.get("search_url")
        if not url:
            return []

        params = {"terms": search_terms, "maxList": str(limit)}
        if query_filter:
            params["q"] = query_filter
        field_indexes = self._ctss_result_field_indexes.get(dataset)
        if field_indexes:
            params["df"] = ",".join(str(index) for index in field_indexes)

        try:
            response = requests.get(url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables %s search failed for terms='%s': %s",
                dataset,
                search_terms,
                exc,
            )
            return []

        try:
            payload = response.json()
        except ValueError:
            payload = None

        logger.debug(
            "ClinicalTables %s raw response status=%s text=%s",
            dataset,
            response.status_code,
            response.text[:500],
        )
        print(response.text)
        fields, entries = self._parse_clinicaltables_payload(dataset, payload)
        logger.debug(
            "ClinicalTables %s parsed fields_count=%s entries_count=%s sample=%s",
            dataset,
            len(fields) if fields else 0,
            len(entries) if entries else 0,
            entries[:1] if entries else entries,
        )
        if not entries:
            return []

        results: List[dict] = []

        for row in entries[:limit]:
            if not isinstance(row, (list, tuple)):
                continue
            data = {
                str(fields[idx]): row[idx]
                for idx in range(min(len(fields), len(row)))
            }

            if not data:
                fallback_record = self._clinicaltables_build_fallback_record(dataset, row, config)
                if fallback_record:
                    logger.debug(
                        "ClinicalTables %s fallback record constructed: %s",
                        dataset,
                        fallback_record,
                    )
                    results.append(fallback_record)
                continue

            name = self._first_match(
                data,
                [
                    "Provider Organization Name (Legal Business Name)",
                    "Provider Last Name (Legal Name)",
                    "Provider First Name",
                ],
                default="Healthcare Provider",
            )

            if dataset == "npi_idv":
                first = data.get("Provider First Name")
                last = data.get("Provider Last Name (Legal Name)")
                middle = data.get("Provider Middle Name")
                combined = " ".join(
                    chunk
                    for chunk in [first, middle, last]
                    if isinstance(chunk, str) and chunk.strip()
                )
                if combined.strip():
                    name = combined.strip()

            city = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address City Name",
                    "Provider Business Mailing Address City Name",
                ],
            )

            state = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address State Name",
                    "Provider Business Mailing Address State Name",
                ],
            )

            postal_code = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address Postal Code",
                    "Provider Business Mailing Address Postal Code",
                ],
            )

            street = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address Line 1",
                    "Provider Business Mailing Address Line 1",
                ],
            )

            country = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address Country Code (If outside U.S.)",
                    "Provider Business Mailing Address Country Code (If outside U.S.)",
                ],
            )

            phone = self._first_match(
                data,
                [
                    "Provider Business Practice Location Address Telephone Number",
                    "Provider Business Mailing Address Telephone Number",
                ],
            )

            taxonomy = self._first_match(
                data,
                [
                    "Provider Taxonomy Description_1",
                    "Healthcare Provider Taxonomy Code_1",
                ],
            )

            languages = self._ensure_list(
                data.get("Provider Language Description_1")
            )

            npi = data.get("NPI")

            if state_hint:
                state_str = str(state).strip().upper() if state else ""
                if state_str and state_str != state_hint.upper():
                    continue

            if zip_hint:
                postal_str = str(postal_code).strip() if postal_code else ""
                if not postal_str or not postal_str.startswith(zip_hint):
                    continue

            if city_hint:
                city_str = str(city).strip().lower() if city else ""
                if city_str != city_hint.strip().lower():
                    continue

            location_parts: List[str] = []
            if street:
                location_parts.append(str(street))

            city_state = ", ".join(
                chunk for chunk in [city, state] if isinstance(chunk, str) and chunk.strip()
            )
            if city_state:
                if postal_code and isinstance(postal_code, str) and postal_code.strip():
                    city_state = f"{city_state} {postal_code.strip()}"
                location_parts.append(city_state)
            elif postal_code:
                location_parts.append(str(postal_code))

            if country and isinstance(country, str) and country.strip() and country.strip().upper() not in {"US", "USA"}:
                location_parts.append(country.strip())

            location_text = ", ".join(location_parts)

            results.append(
                {
                    "name": name,
                    "location": location_text or ", ".join(
                        chunk
                        for chunk in [city, state]
                        if isinstance(chunk, str) and chunk.strip()
                    ),
                    "phone": phone,
                    "npi": npi,
                    "languages": languages,
                    "taxonomy": taxonomy,
                    "source": config.get(
                        "source_label", "NPI Registry via clinicaltables.nlm.nih.gov"
                    ),
                    "dataset": dataset,
                    "raw": data,
                }
            )

        return results

    # ------------------------------------------------------------------
    def _clinicaltables_build_fallback_record(
        self, dataset: str, row: Tuple[Any, ...], config: Dict[str, Any]
    ) -> Optional[dict]:
        if not isinstance(row, (list, tuple)) or not row:
            return None

        def _get(index: int) -> Optional[Any]:
            return row[index] if len(row) > index else None

        raw_name = _get(0)
        name = str(raw_name).strip() if raw_name else "Healthcare Provider"
        if dataset == "npi_idv" and "," in name:
            parts = [part.strip() for part in name.split(",", 1)]
            if len(parts) == 2:
                formatted = f"{parts[1]} {parts[0]}".strip()
                if formatted:
                    name = formatted

        npi = _get(1)
        taxonomy = _get(2)
        address = _get(3)

        phone = None
        if len(row) > 4:
            phone_candidate = _get(4)
            if isinstance(phone_candidate, str) and phone_candidate.strip():
                phone = phone_candidate.strip()

        location = str(address).strip() if address else ""

        return {
            "name": name,
            "location": location,
            "phone": phone,
            "npi": npi,
            "languages": [],
            "taxonomy": taxonomy,
            "source": config.get(
                "source_label", "NPI Registry via clinicaltables.nlm.nih.gov"
            ),
            "dataset": dataset,
            "raw": {"fallback_row": list(row)},
        }

    # ------------------------------------------------------------------
    def _compose_clinicaltables_query_parts(
        self, query: ParsedCareQuery
    ) -> Tuple[str, Optional[str], bool]:
        parts: List[str] = []
        filters: List[str] = []
        location_was_specific = False

        # Specialty: pick the first reasonable suggestion
        for specialty in query.specialties:
            best = self._best_suggestion_across_datasets(
                specialty, self._ctss_taxonomy_fields
            )
            if best:
                parts.append(best)
                break

        # Location: reduce to one normalized chunk (city/state/zip)
        alias_used = False

        location_inputs: List[str] = []
        city_hint: Optional[str] = None
        zip_hint: Optional[str] = None
        state_hint: Optional[str] = None

        inferred_location = self._infer_location_hint(query)
        if inferred_location:
            location_inputs.append(inferred_location)

        if query.location:
            location_inputs.append(query.location)

        for location_input in location_inputs:
            normalized_location, alias_used = self._normalize_location_alias(
                location_input
            )
            for chunk in self._split_location(normalized_location):
                user_chunk = chunk.strip()
                if not user_chunk:
                    continue

                match = self._match_city_state(user_chunk)
                chunk_city: Optional[str] = None
                chunk_state: Optional[str] = None
                if match:
                    chunk_city, chunk_state = match
                else:
                    chunk_state = self._extract_state_code(user_chunk)

                chunk_zip = self._extract_zip_code(user_chunk)

                if chunk_city and not city_hint:
                    city_hint = chunk_city
                if chunk_state and not state_hint:
                    state_hint = chunk_state
                if chunk_zip and not zip_hint:
                    zip_hint = chunk_zip

                if user_chunk not in parts:
                    parts.append(user_chunk)

                suggestion = self._best_suggestion_across_datasets(
                    user_chunk, self._ctss_location_fields
                )

                suggestion_for_filters: Optional[str] = None
                suggestion_for_terms: Optional[str] = None

                if suggestion and suggestion.strip() and suggestion != user_chunk:
                    user_state = self._extract_state_code(user_chunk)
                    suggestion_state = self._extract_state_code(suggestion)
                    if user_state and suggestion_state and user_state != suggestion_state:
                        suggestion_for_filters = None
                        suggestion_for_terms = None
                    else:
                        suggestion_for_filters = suggestion
                        suggestion_for_terms = suggestion

                if suggestion_for_terms and suggestion_for_terms not in parts:
                    parts.append(suggestion_for_terms)

                chunk_filters = self._build_query_filters(
                    user_chunk,
                    suggestion_for_filters,
                    city_hint=chunk_city,
                    zip_hint=chunk_zip,
                    state_hint=chunk_state,
                )
                for filter_value in chunk_filters:
                    if filter_value not in filters:
                        filters.append(filter_value)
                if self._location_filters_are_specific(
                    chunk_filters, user_chunk, alias_used
                ):
                    location_was_specific = True

                should_stop = bool(chunk_filters) or bool(chunk_zip) or bool(chunk_city)
                if location_was_specific or should_stop:
                    break

            if location_was_specific or city_hint or zip_hint:
                break

        if not state_hint:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.state:"):
                    state_hint = filter_value.split(":", 1)[1]
                    break

        if not zip_hint:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.zip:"):
                    zip_hint = filter_value.split(":", 1)[1].strip()
                    break

        if not city_hint and filters:
            for filter_value in filters:
                if filter_value.startswith("addr_practice.city:"):
                    value = filter_value.split(":", 1)[1].strip()
                    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                        value = value[1:-1]
                    city_hint = value
                    break

        query_filter = " AND ".join(filters) if filters else None

        return (
            " ".join(part for part in parts if part),
            query_filter,
            location_was_specific,
            city_hint,
            zip_hint,
            state_hint,
        )

    # ------------------------------------------------------------------
    def _build_query_filters(
        self,
        user_location: str,
        suggestion: Optional[str] = None,
        *,
        city_hint: Optional[str] = None,
        zip_hint: Optional[str] = None,
        state_hint: Optional[str] = None,
    ) -> List[str]:
        filters: List[str] = []

        user_state = self._extract_state_code(user_location)
        suggestion_state = (
            self._extract_state_code(suggestion) if suggestion else None
        )

        state_code = state_hint or user_state or suggestion_state
        if user_state and suggestion_state and user_state != suggestion_state:
            state_code = user_state

        if state_code:
            filters.append(f"addr_practice.state:{state_code}")

        if zip_hint:
            filters.append(f"addr_practice.zip:{zip_hint}")
        elif city_hint and state_code:
            escaped_city = city_hint.replace('"', '\"')
            filters.append(f'addr_practice.city:"{escaped_city}"')

        return list(dict.fromkeys(filters))

    # ------------------------------------------------------------------
    def _location_filters_are_specific(
        self, filters: List[str], user_location: str, alias_used: bool
    ) -> bool:
        if self._extract_zip_code(user_location):
            return True

        if alias_used:
            return False

        has_state = any(filter_str.startswith("addr_practice.state") for filter_str in filters)
        if not has_state:
            return False

        return self._location_has_city(user_location)

    # ------------------------------------------------------------------
    def _location_has_city(self, value: str) -> bool:
        normalized = value.strip().lower()
        if not normalized:
            return False

        if normalized in _US_STATE_NAMES:
            return False

        if normalized.upper() in _US_STATE_CODES:
            return False

        stripped_state_candidate = re.sub(
            r"\b(state|province|region|area|of|the)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped_state_candidate = re.sub(r"\s+", " ", stripped_state_candidate).strip()
        if stripped_state_candidate in _US_STATE_NAMES:
            return False

        tokens = [token.strip() for token in re.split(r"[,\s]+", normalized) if token.strip()]
        for token in tokens:
            upper = token.upper()
            if upper in _US_STATE_CODES or token in _US_STATE_NAMES:
                continue
            if token in _LOCATION_NOISE_TOKENS:
                continue
            if token.isdigit():
                continue
            if len(token) < 3:
                continue
            if not re.search(r"[A-Za-z]", token):
                continue
            return True
        return False

    # ------------------------------------------------------------------
    def _extract_zip_code(self, value: str) -> Optional[str]:
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", value)
        if match:
            return match.group(1)
        return None

    # ------------------------------------------------------------------
    def _extract_state_code(self, value: str) -> Optional[str]:
        tokens = re.split(r"[,\s]+", value)
        for token in tokens:
            upper = token.strip().upper()
            if upper in _US_STATE_CODES:
                return upper
        normalized = value.strip().lower()
        if normalized in _US_STATE_NAMES:
            return _US_STATE_NAMES[normalized]
        stripped = re.sub(
            r"\b(state|province|region|area|of|the)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped in _US_STATE_NAMES:
            return _US_STATE_NAMES[stripped]
        return None

    # ------------------------------------------------------------------
    def _normalize_location_alias(self, location: str) -> Tuple[str, bool]:
        key = location.strip().lower()
        if key in self._location_aliases:
            return self._location_aliases[key], True
        return location, False

    # ------------------------------------------------------------------
    def _location_hint(self) -> str:
        return (
            "Please share a specific city, state, or ZIP code so I can refine the search."
        )

    # ------------------------------------------------------------------
    def _infer_location_hint(self, query: ParsedCareQuery) -> Optional[str]:
        texts: List[str] = []
        if query.summary:
            texts.append(query.summary)
        texts.extend(query.keywords)

        for text in texts:
            if not text or not isinstance(text, str):
                continue
            match = self._match_city_state(text)
            if match:
                city, state_code = match
                return f"{city} {state_code}"
        return None

    # ------------------------------------------------------------------
    def _match_city_state(self, text: str) -> Optional[Tuple[str, str]]:
        for pattern in (self._city_state_code_regex, self._city_state_name_regex):
            match = pattern.search(text)
            if not match:
                continue

            city = match.group("city").strip().strip(",")
            state_fragment = match.group("state").strip()

            state_code = None
            if pattern is self._city_state_code_regex:
                state_code = state_fragment.upper()
            else:
                state_code = _US_STATE_NAMES.get(state_fragment.lower())

            if not state_code:
                continue

            location = f"{city} {state_code}"
            if self._location_has_city(location):
                return city.strip(), state_code

        return None

    # ------------------------------------------------------------------
    def _parse_clinicaltables_payload(
        self,
        dataset: str,
        payload: Any,
    ) -> Tuple[List[str], List[List[Any]]]:
        fields: List[str] = []
        entries: List[List[Any]] = []

        if not isinstance(payload, list):
            return fields, entries

        if len(payload) >= 4 and isinstance(payload[2], list) and isinstance(payload[3], list):
            potential_fields = payload[2]
            potential_entries = payload[3]
        elif len(payload) >= 3 and isinstance(payload[1], list) and isinstance(payload[2], list):
            potential_fields = payload[1]
            potential_entries = payload[2]
        else:
            potential_fields = (
                payload[2] if len(payload) > 2 and isinstance(payload[2], list) else []
            )
            potential_entries = (
                payload[3] if len(payload) > 3 and isinstance(payload[3], list) else []
            )

        if all(isinstance(item, str) for item in potential_fields):
            fields = [str(item) for item in potential_fields]
        elif all(isinstance(item, int) for item in potential_fields):
            reverse_map = {
                index: name
                for name, index in self._ctss_field_map.get(dataset, {}).items()
            }
            fields = [reverse_map.get(int(item), str(item)) for item in potential_fields]
        else:
            configured_fields = self._ctss_result_field_order.get(dataset, [])
            if not configured_fields:
                configured_fields = (
                    self.ctss_dataset_configs.get(dataset, {}).get("result_fields")
                    or []
                )
            if configured_fields:
                fields = list(configured_fields)
            else:
                fields = []

        filtered_entries: List[List[Any]] = []
        for row in potential_entries:
            if isinstance(row, (list, tuple)):
                filtered_entries.append(list(row))
        entries = filtered_entries

        return fields, entries

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_clinicaltables_fields_payload(payload: Any) -> Dict[str, int]:
        mapping: Dict[str, int] = {}

        if not isinstance(payload, list):
            return mapping

        if payload and isinstance(payload[0], list) and len(payload[0]) >= 2:
            entries = payload
        else:
            entries = payload[1:] if len(payload) > 1 else []

        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 2:
                continue

            field_index: Optional[int] = None
            index_candidate = entry[0]
            if isinstance(index_candidate, (int, str)):
                try:
                    field_index = int(index_candidate)
                except (TypeError, ValueError):
                    field_index = None

            field_name_candidate = entry[1]
            if not isinstance(field_name_candidate, str) or not field_name_candidate:
                continue

            if field_index is None:
                field_index = len(mapping)

            mapping[field_name_candidate] = field_index

        return mapping

    # ------------------------------------------------------------------
    def _best_suggestion_across_datasets(
        self, term: Optional[str], candidate_fields: List[str]
    ) -> str:
        if not term:
            return ""

        cleaned = term.strip()
        if not cleaned:
            return ""

        for dataset in self._ctss_dataset_priority:
            suggestions = self._clinicaltables_suggest_for_dataset(
                dataset, cleaned, candidate_fields
            )
            if suggestions:
                return suggestions[0]

        return cleaned

    # ------------------------------------------------------------------
    def _clinicaltables_suggest_for_dataset(
        self, dataset: str, term: str, candidate_fields: List[str]
    ) -> List[str]:
        config = self.ctss_dataset_configs.get(dataset, {})
        values_url = config.get("values_url")
        if not values_url:
            return []

        field_map = self._ctss_field_map.get(dataset)
        if not field_map:
            return []

        aggregated: List[str] = []
        for field in candidate_fields:
            field_index = field_map.get(field)
            if field_index is None:
                continue

            cache_key = (dataset, term, field)
            cached = self._ctss_suggest_cache.get(cache_key)
            if cached is not None:
                aggregated.extend(cached)
                continue

            suggestions = self._clinicaltables_request_values(
                dataset, term, field_index, values_url
            )
            self._ctss_suggest_cache[cache_key] = suggestions
            aggregated.extend(suggestions)

        return list(dict.fromkeys(aggregated))

    # ------------------------------------------------------------------
    def _clinicaltables_request_values(
        self, dataset: str, term: str, field_index: int, values_url: str
    ) -> List[str]:
        params = {
            "terms": term,
            "df": str(field_index),
            "maxList": str(self.ctss_values_max_results),
        }

        logger.debug(
            "ClinicalTables values request dataset=%s term=%s field_index=%s",
            dataset,
            term,
            field_index,
        )

        try:
            response = requests.get(values_url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables values failed dataset=%s term=%s field_index=%s error=%s",
                dataset,
                term,
                field_index,
                exc,
            )
            return []

        try:
            payload = response.json()
        except ValueError:
            payload = None

        suggestions: List[str] = []

        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    suggestions.extend(self._flatten_suggestions(value))
                elif isinstance(value, str):
                    suggestions.append(value)
        elif isinstance(payload, list):
            if payload and isinstance(payload[0], int):
                for item in payload[1:]:
                    if isinstance(item, list):
                        suggestions.extend(self._flatten_suggestions(item))
                    elif isinstance(item, str):
                        suggestions.append(item)
            else:
                suggestions.extend(self._flatten_suggestions(payload))

        return list(dict.fromkeys(suggestions))

    # ------------------------------------------------------------------
    def _initialize_clinicaltables_field_maps(self) -> None:
        for dataset, config in self.ctss_dataset_configs.items():
            field_map = self._fetch_field_map_for_dataset(dataset, config)
            self._ctss_field_map[dataset] = field_map

            requested_fields = config.get("result_fields") or []
            if not field_map or not requested_fields:
                continue

            indexes: List[int] = []
            resolved_fields: List[str] = []
            alias_map: Dict[str, List[str]] = config.get("field_aliases", {})

            for canonical_field in requested_fields:
                candidates = [canonical_field] + alias_map.get(canonical_field, [])
                selected_index: Optional[int] = None
                for candidate in candidates:
                    candidate_index = field_map.get(candidate)
                    if candidate_index is not None:
                        selected_index = candidate_index
                        break
                if selected_index is None:
                    logger.warning(
                        "ClinicalTables %s missing requested field '%s'",
                        dataset,
                        canonical_field,
                    )
                    continue
                indexes.append(selected_index)
                resolved_fields.append(canonical_field)

            if indexes:
                self._ctss_result_field_indexes[dataset] = indexes
                self._ctss_result_field_order[dataset] = resolved_fields

    # ------------------------------------------------------------------
    def _fetch_field_map_for_dataset(
        self, dataset: str, config: Dict[str, Any]
    ) -> Dict[str, int]:
        fields_url = config.get("fields_url")
        if fields_url:
            try:
                response = requests.get(fields_url, timeout=self.ctss_timeout)
                response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "ClinicalTables %s fields fetch failed: %s",
                    dataset,
                    exc,
                )
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                else:
                    mapping = self._parse_clinicaltables_fields_payload(payload)
                    if mapping:
                        logger.debug(
                            "ClinicalTables %s field map loaded (sample=%s)",
                            dataset,
                            list(mapping.keys())[:6],
                        )
                        return mapping

        # Fallback: probe the search endpoint for at least one result and infer fields
        url = config.get("search_url")
        if not url:
            return {}

        probe_term = config.get("probe_term") or "urology"
        params = {"terms": probe_term, "maxList": "1"}

        try:
            response = requests.get(url, params=params, timeout=self.ctss_timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "ClinicalTables %s field map probe failed: %s",
                dataset,
                exc,
            )
            return {}

        try:
            payload = response.json()
        except ValueError:
            payload = None

        fields, _ = self._parse_clinicaltables_payload(dataset, payload)
        if not fields:
            return {}

        mapping = {str(field): idx for idx, field in enumerate(fields)}
        logger.debug(
            "ClinicalTables %s fallback field map loaded (sample=%s)",
            dataset,
            list(mapping.keys())[:6],
        )
        return mapping

    # ------------------------------------------------------------------
    def _flatten_suggestions(self, values: List[Any]) -> List[str]:
        flattened: List[str] = []
        for value in values:
            if isinstance(value, str):
                flattened.append(value)
            elif isinstance(value, list):
                flattened.extend(
                    [item for item in value if isinstance(item, str) and item]
                )
        return flattened

    # ------------------------------------------------------------------
    @staticmethod
    def _split_location(location: str) -> List[str]:
        parts = [part.strip() for part in re.split(r"[|/,]", location) if part.strip()]
        if parts:
            return parts
        stripped = location.strip()
        return [stripped] if stripped else [location]

    # ------------------------------------------------------------------
    def _trusted_resource_fallback(self, query: ParsedCareQuery) -> List[dict]:
        region_hint = query.location or "international"
        if not self.fallback_resources:
            return []

        query_terms = {
            term.lower()
            for term in (query.specialties + query.keywords)
            if isinstance(term, str)
        }
        location_text = (query.location or "").lower()

        suggestions: List[dict] = []
        for resource in self.fallback_resources:
            name = resource.get("name")
            url = resource.get("url")
            description = resource.get("description")
            if not name or not url:
                continue

            specialty_filters = [
                str(item).lower() for item in resource.get("specialties", []) if item
            ]
            region_filters = [
                str(item).lower() for item in resource.get("regions", []) if item
            ]

            if specialty_filters:
                if not query_terms:
                    continue
                if not any(filter_term in query_terms for filter_term in specialty_filters):
                    continue

            if region_filters:
                if location_text:
                    if not any(filter_term in location_text for filter_term in region_filters):
                        continue
                else:
                    if not any(filter_term in {"international", "global"} for filter_term in region_filters):
                        continue

            suggestions.append(
                {
                    "name": name,
                    "location": region_hint,
                    "website": url,
                    "description": description,
                    "source": "Trusted public directories",
                }
            )

        return suggestions

    # ------------------------------------------------------------------
    def _compose_response(
        self,
        client: InferenceClient,
        payload: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        template_key: str = "response_template",
    ) -> str:
        response_language = payload["query"].get("response_language", "English")
        summary_prompt = self.prompts.get("response_system") or (
            "You are a care navigation assistant responding to users in their preferred language."
        )

        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        template = self.prompts.get(template_key) or (
            "Here is the structured data you must convert into a helpful message:\n"
            "```json\n{payload_json}\n```\n"
            "Requirements:\n"
            "- Write in the language specified by query.response_language.\n"
            "- Briefly confirm the understood need.\n"
            "- Present providers as a numbered list with key details.\n"
            "- Mention if the results came from a public API or curated resource list.\n"
            "- Offer next steps or safety guidance when appropriate."
        )
        user_instructions = template.format(payload_json=payload_json)

        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": user_instructions},
        ]

        logger.debug("Compose prompt system=%s", summary_prompt)
        logger.debug("Compose payload JSON=%s", payload_json)

        completion = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        finish_reason = completion.choices[0].finish_reason if completion.choices else "unknown"
        logger.info("Response generation finish_reason=%s", finish_reason)

        return completion.choices[0].message["content"].strip()

    # ------------------------------------------------------------------
    def _should_use_fallback_only_template(self) -> bool:
        repository = self.provider_repository

        if repository is None:
            return True

        has_dataset_id = bool(getattr(repository, "dataset_id", None))
        has_local_records = bool(getattr(repository, "providers", []))

        return (not has_dataset_id) and (not has_local_records)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_parse(content: str) -> Optional[dict]:
        if not content:
            return None

        trimmed = content.strip()
        if trimmed.startswith('```'):
            trimmed = trimmed[3:]
            if trimmed.lower().startswith('json'):
                trimmed = trimmed[4:]
            trimmed = trimmed.lstrip("\n").rstrip("`").strip()

        try:
            return json.loads(trimmed)
        except json.JSONDecodeError as exc:
            repaired = CareLocatorAgent._repair_json(trimmed)
            if repaired and repaired != trimmed:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.warning("Interpret payload repair failed", exc_info=True)
            logger.warning("Interpret payload JSON parse failed: %s", exc)
            logger.debug("Interpret payload raw text=%s", trimmed)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _repair_json(payload: str) -> str:
        if not payload:
            return payload

        repaired = re.sub(r",(\s*[}\]])", r"\1", payload)

        stack: List[str] = []
        for char in repaired:
            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in ("}", "]") and stack:
                if stack[-1] == char:
                    stack.pop()

        while stack:
            repaired += stack.pop()

        return repaired

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [value]
        return []

    # ------------------------------------------------------------------
    @staticmethod
    def _first_match(data: Dict[str, Any], keys: List[str], default: Optional[str] = None) -> Optional[str]:
        for key in keys:
            if key in data and data[key]:
                return str(data[key])
        return default


__all__ = ["CareLocatorAgent", "ParsedCareQuery"]
