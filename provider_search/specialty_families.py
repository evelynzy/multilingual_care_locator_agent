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
        aliases=("cardiology", "cardiologist"),
    ),
    SpecialtyFamily(
        family_id="gastroenterology",
        label="Gastroenterology",
        aliases=("gastroenterology", "gi", "digestive health"),
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
        aliases=("ent", "otolaryngology", "otorhinolaryngology"),
    ),
    SpecialtyFamily(
        family_id="psychiatry-behavioral-health",
        label="Psychiatry / Behavioral Health",
        aliases=(
            "psychiatry",
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
        aliases=("orthopedics", "orthopaedics", "orthopedic surgery", "sports medicine"),
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
        aliases=("oncology", "hematology", "cancer care"),
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

def normalize_specialty_family_id(value: object) -> Optional[str]:
    normalized_value = _normalize_lookup_value(value)
    if normalized_value is None:
        return None
    return _SPECIALTY_FAMILY_LOOKUP.get(normalized_value)


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


SPECIALTY_FAMILY_BY_ID = {
    family.family_id: family
    for family in SPECIALTY_FAMILY_CATALOG
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
