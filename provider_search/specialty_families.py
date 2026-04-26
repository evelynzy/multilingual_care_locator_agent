from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Optional


@dataclass(frozen=True)
class SpecialtyFamily:
    """Stable family identifier plus common specialty/taxonomy aliases."""

    family_id: str
    label: str
    aliases: tuple[str, ...]


SPECIALTY_FAMILY_CATALOG: tuple[SpecialtyFamily, ...] = (
    SpecialtyFamily(
        family_id="primary-care",
        label="Primary Care",
        aliases=(
            "primary care",
            "clinic/center, primary care",
            "clinic center primary care",
            "family medicine",
            "family practice",
            "internal medicine",
            "physician/family practice",
            "physician family practice",
            "physician/internal medicine",
            "physician internal medicine",
            "general practice",
            "adult medicine",
            "pcp",
        ),
    ),
    SpecialtyFamily(
        family_id="pediatrics",
        label="Pediatrics",
        aliases=(
            "pediatrics",
            "pediatric",
            "pediatrician",
            "child health",
            "adolescent medicine",
            "pediatric medicine",
        ),
    ),
    SpecialtyFamily(
        family_id="dentistry",
        label="Dentistry",
        aliases=(
            "dentistry",
            "dentist",
            "dentist, dental anesthesiology",
            "dentist, endodontics",
            "dentist, general practice",
            "dentist, oral and maxillofacial pathology",
            "dentist, oral and maxillofacial radiology",
            "dentist, oral and maxillofacial surgery",
            "dentist, oral medicine",
            "dentist, orthodontics and dentofacial orthopedics",
            "dentist, pediatric dentistry",
            "dentist, periodontics",
            "dentist, prosthodontics",
            "dentist, public health",
            "dental anesthesiology",
            "endodontics",
            "oral and maxillofacial pathology",
            "oral and maxillofacial radiology",
            "oral and maxillofacial surgery",
            "oral medicine",
            "orthodontics and dentofacial orthopedics",
            "pediatric dentistry",
            "periodontics",
            "prosthodontics",
            "public health",
            "dentista",
        ),
    ),
    SpecialtyFamily(
        family_id="urgent-care",
        label="Urgent Care",
        aliases=("urgent care", "walk in clinic", "walk-in clinic"),
    ),
    SpecialtyFamily(
        family_id="obstetrics-gynecology",
        label="Obstetrics & Gynecology",
        aliases=(
            "ob gyn",
            "obgyn",
            "obstetrics and gynecology",
            "obstetrics gynecology",
            "physician obstetrics and gynecology",
            "physician obstetrics gynecology",
            "207v00000x",
            "207vc0200x",
            "207ve0102x",
            "207vg0400x",
            "207vm0101x",
            "207vx0000x",
            "gynecologic oncology",
            "gynecology",
            "maternal and fetal medicine",
            "maternal fetal medicine",
            "obstetrics",
            "reproductive endocrinology",
            "womens health",
            "women's health",
        ),
    ),
    SpecialtyFamily(
        family_id="dermatology",
        label="Dermatology",
        aliases=("dermatology", "dermatologist"),
    ),
    SpecialtyFamily(
        family_id="cardiology",
        label="Cardiology",
        aliases=(
            "cardiology",
            "cardiologist",
            "cardiovascular disease",
            "physician internal medicine cardiovascular disease",
            "207rc0000x",
        ),
    ),
    SpecialtyFamily(
        family_id="gastroenterology",
        label="Gastroenterology",
        aliases=("gastroenterology", "gastroenterologist", "gi", "digestive health"),
    ),
    SpecialtyFamily(
        family_id="neurology",
        label="Neurology",
        aliases=("neurology", "neurologist"),
    ),
    SpecialtyFamily(
        family_id="endocrinology",
        label="Endocrinology",
        aliases=("endocrinology", "endocrinologist"),
    ),
    SpecialtyFamily(
        family_id="ent",
        label="ENT / Otolaryngology",
        aliases=("ent", "otolaryngology", "otorhinolaryngology", "ear nose throat"),
    ),
    SpecialtyFamily(
        family_id="psychiatry-behavioral-health",
        label="Psychiatry / Behavioral Health",
        aliases=(
            "psychiatry",
            "psychiatrist",
            "mental health",
            "behavioral health",
            "psychology",
            "psychotherapy",
            "therapy",
            "counseling",
        ),
    ),
    SpecialtyFamily(
        family_id="orthopedics",
        label="Orthopedics",
        aliases=(
            "orthopedics",
            "orthopaedics",
            "orthopedic",
            "orthopaedic",
            "orthopedic surgery",
            "sports medicine",
        ),
    ),
    SpecialtyFamily(
        family_id="eye-care",
        label="Eye Care",
        aliases=("ophthalmology", "optometry", "eye care", "vision care"),
    ),
    SpecialtyFamily(
        family_id="urology",
        label="Urology",
        aliases=("urology", "urologist"),
    ),
    SpecialtyFamily(
        family_id="nephrology",
        label="Nephrology",
        aliases=("nephrology", "kidney care"),
    ),
    SpecialtyFamily(
        family_id="pulmonology",
        label="Pulmonology",
        aliases=("pulmonology", "pulmonary", "lung care"),
    ),
    SpecialtyFamily(
        family_id="allergy-immunology",
        label="Allergy & Immunology",
        aliases=("allergy", "allergies", "immunology", "allergy immunology"),
    ),
    SpecialtyFamily(
        family_id="rheumatology",
        label="Rheumatology",
        aliases=("rheumatology", "rheumatologist"),
    ),
    SpecialtyFamily(
        family_id="oncology-hematology",
        label="Oncology / Hematology",
        aliases=("oncology", "oncologist", "hematology", "cancer care"),
    ),
    SpecialtyFamily(
        family_id="physical-therapy-rehab",
        label="Physical Therapy / Rehab",
        aliases=("physical therapy", "physiatry", "pm and r", "rehabilitation"),
    ),
    SpecialtyFamily(
        family_id="radiology-imaging",
        label="Radiology / Imaging",
        aliases=("radiology", "diagnostic radiology", "imaging"),
    ),
)

_PROFESSION_WRAPPER_PREFIXES = frozenset(
    {
        "physician",
        "nurse practitioner",
        "physician assistant",
        "physician assistants",
        "clinical nurse specialist",
        "advanced practice registered nurse",
        "registered nurse",
    }
)

def normalize_specialty_family_id(value: object) -> Optional[str]:
    normalized_value = _normalize_lookup_value(value)
    if normalized_value is None:
        return None

    exact_match = _SPECIALTY_FAMILY_LOOKUP.get(normalized_value)
    if exact_match is not None:
        return exact_match

    for candidate in _canonical_lookup_candidates(value):
        family_id = _SPECIALTY_FAMILY_LOOKUP.get(candidate)
        if family_id is not None:
            return family_id
    return None


def derive_specialty_family_ids(
    values: Optional[Iterable[object]],
) -> tuple[str, ...]:
    if values is None:
        return ()

    family_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        family_id = normalize_specialty_family_id(value)
        if family_id is None or family_id in seen:
            continue
        seen.add(family_id)
        family_ids.append(family_id)
    return tuple(family_ids)


def derive_request_specialty_family_ids(
    specialties: Optional[Iterable[object]],
    explicit_family_ids: Optional[Iterable[object]] = None,
) -> tuple[str, ...]:
    return _merge_specialty_family_ids(
        derive_specialty_family_ids(explicit_family_ids),
        derive_specialty_family_ids(specialties),
    )


def normalize_query_specialty_family_id(value: object) -> Optional[str]:
    normalized_value = _normalize_lookup_value(value)
    if normalized_value is None:
        return None
    return _QUERY_SPECIALTY_FAMILY_LOOKUP.get(normalized_value)


def derive_query_specialty_family_ids(
    values: Optional[Iterable[object]],
) -> tuple[str, ...]:
    if values is None:
        return ()

    family_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        family_id = normalize_query_specialty_family_id(value)
        if family_id is None or family_id in seen:
            continue
        seen.add(family_id)
        family_ids.append(family_id)
    return tuple(family_ids)


def derive_provider_specialty_family_ids(
    specialties: Optional[Iterable[object]],
    taxonomy: object = None,
    explicit_family_ids: Optional[Iterable[object]] = None,
) -> tuple[str, ...]:
    evidence_values: list[object] = []
    if specialties is not None:
        evidence_values.extend(list(specialties))
    if taxonomy is not None:
        evidence_values.append(taxonomy)
    return _merge_specialty_family_ids(
        derive_specialty_family_ids(explicit_family_ids),
        derive_specialty_family_ids(evidence_values),
    )


def _merge_specialty_family_ids(*family_id_groups: Iterable[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for family_ids in family_id_groups:
        for family_id in family_ids:
            if family_id in seen:
                continue
            seen.add(family_id)
            merged.append(family_id)
    return tuple(merged)


def _normalize_lookup_value(value: object) -> Optional[str]:
    if value is None:
        return None

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    if not normalized:
        return None
    return normalized


def _canonical_lookup_candidates(value: object) -> tuple[str, ...]:
    raw_value = unicodedata.normalize("NFKC", str(value)).strip()
    if not raw_value:
        return ()

    normalized_candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate_value: Optional[str]) -> None:
        normalized_candidate = _normalize_lookup_value(candidate_value)
        if normalized_candidate is None or normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        normalized_candidates.append(normalized_candidate)

    raw_phrases = [raw_value]
    stripped_wrapper = _strip_profession_wrapper(raw_value)
    if stripped_wrapper is not None:
        raw_phrases.append(stripped_wrapper)

    for phrase in raw_phrases:
        for extracted_phrase in _extract_canonicalization_phrases(phrase):
            add_candidate(extracted_phrase)

    if stripped_wrapper is not None and "," not in stripped_wrapper:
        stripped_wrapper_candidate = _normalize_lookup_value(stripped_wrapper)
        if stripped_wrapper_candidate is not None:
            for suffix_candidate in _matching_suffix_candidates(stripped_wrapper_candidate):
                if suffix_candidate in seen:
                    continue
                seen.add(suffix_candidate)
                normalized_candidates.append(suffix_candidate)

    return tuple(normalized_candidates)


def _strip_profession_wrapper(value: str) -> Optional[str]:
    if "/" not in value:
        return None

    prefix, remainder = value.split("/", 1)
    if _normalize_lookup_value(prefix) not in _PROFESSION_WRAPPER_PREFIXES:
        return None

    cleaned_remainder = remainder.strip()
    if not cleaned_remainder:
        return None
    return cleaned_remainder


def _extract_canonicalization_phrases(value: str) -> tuple[str, ...]:
    phrases: list[str] = []
    seen: set[str] = set()

    def add_phrase(candidate: Optional[str]) -> None:
        if candidate is None:
            return
        cleaned_candidate = candidate.strip(" ,-/")
        if not cleaned_candidate:
            return
        normalized_candidate = _normalize_lookup_value(cleaned_candidate)
        if normalized_candidate is None or normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        phrases.append(cleaned_candidate)

    add_phrase(value)

    parenthetical_matches = re.findall(r"\(([^()]*)\)", value)
    without_parentheticals = re.sub(r"\([^()]*\)", " ", value)
    add_phrase(without_parentheticals)
    for parenthetical_phrase in parenthetical_matches:
        add_phrase(parenthetical_phrase)

    for candidate in tuple(phrases):
        comma_parts = [part.strip() for part in candidate.split(",")]
        if len(comma_parts) <= 1:
            continue

        head_phrase = comma_parts[0]
        add_phrase(head_phrase)

        normalized_head_candidates = [_normalize_lookup_value(head_phrase)]
        stripped_head_phrase = _strip_profession_wrapper(head_phrase)
        if stripped_head_phrase is not None:
            normalized_head_candidates.append(_normalize_lookup_value(stripped_head_phrase))

        if any(
            normalized_head_candidate and normalized_head_candidate in _SPECIALTY_FAMILY_LOOKUP
            for normalized_head_candidate in normalized_head_candidates
        ):
            continue

        for comma_phrase in comma_parts[1:]:
            add_phrase(comma_phrase)

    return tuple(phrases)


def _matching_suffix_candidates(normalized_value: str) -> tuple[str, ...]:
    tokens = normalized_value.split()
    if len(tokens) <= 1:
        return ()

    suffix_candidates: list[str] = []
    seen: set[str] = set()
    for start_index in range(1, len(tokens)):
        suffix = " ".join(tokens[start_index:])
        if suffix in seen or suffix not in _SPECIALTY_FAMILY_LOOKUP:
            continue
        seen.add(suffix)
        suffix_candidates.append(suffix)
    return tuple(suffix_candidates)


SPECIALTY_FAMILY_BY_ID = {
    family.family_id: family
    for family in SPECIALTY_FAMILY_CATALOG
}

QUERY_SPECIALTY_FAMILY_ALIASES_BY_ID = {
    "primary-care": (
        "primary care",
        "pcp",
        "family medicine",
        "family practice",
        "internal medicine",
    ),
    "pediatrics": (
        "pediatrics",
        "pediatric",
        "pediatrician",
        "child health",
    ),
    "dentistry": ("dentistry", "dentist", "dentista"),
    "obstetrics-gynecology": (
        "ob gyn",
        "obgyn",
        "obstetrics and gynecology",
        "obstetrics gynecology",
        "gynecology",
        "obstetrics",
    ),
    "dermatology": ("dermatology", "dermatologist"),
    "cardiology": ("cardiology", "cardiologist", "cardiovascular disease"),
    "gastroenterology": ("gastroenterology", "gastroenterologist"),
    "neurology": ("neurology", "neurologist"),
    "endocrinology": ("endocrinology", "endocrinologist"),
    "ent": ("ent", "otolaryngology", "otorhinolaryngology", "ear nose throat"),
    "psychiatry-behavioral-health": (
        "psychiatry",
        "psychiatrist",
    ),
    "orthopedics": (
        "orthopedics",
        "orthopaedics",
        "orthopedic",
        "orthopaedic",
        "orthopedic surgery",
    ),
    "eye-care": ("ophthalmology", "optometry", "eye care"),
    "urology": ("urology", "urologist"),
    "nephrology": ("nephrology",),
    "pulmonology": ("pulmonology", "pulmonary"),
    "allergy-immunology": ("allergy immunology",),
    "rheumatology": ("rheumatology", "rheumatologist"),
    "oncology-hematology": ("oncology", "oncologist", "hematology"),
    "physical-therapy-rehab": ("physical therapy", "physiatry", "pm and r"),
    "radiology-imaging": ("radiology", "diagnostic radiology"),
}

_SPECIALTY_FAMILY_LOOKUP = {
    normalized_alias: family.family_id
    for family in SPECIALTY_FAMILY_CATALOG
    for normalized_alias in (
        _normalize_lookup_value(family.family_id),
        _normalize_lookup_value(family.label),
        *(_normalize_lookup_value(alias) for alias in family.aliases),
    )
    if normalized_alias
}

_QUERY_SPECIALTY_FAMILY_LOOKUP = {
    normalized_alias: family_id
    for family_id, aliases in QUERY_SPECIALTY_FAMILY_ALIASES_BY_ID.items()
    for normalized_alias in (_normalize_lookup_value(alias) for alias in aliases)
    if normalized_alias
}
