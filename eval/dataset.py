from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

LANGUAGES = ("en", "zh", "es", "ar", "ko")


@dataclass(frozen=True)
class GoldLabels:
    expected_specialty: Optional[str]
    expected_state: Optional[str]
    expect_followup: bool
    expect_nonzero_providers: bool
    expect_emergency_routing: bool
    expected_preferred_language: Optional[str]


@dataclass(frozen=True)
class LanguageVariant:
    language: str
    turns: List[str]
    source: str
    verification_status: str
    verified_by: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    dimension: str
    gold: GoldLabels
    variants: Dict[str, LanguageVariant]


def _parse_variant(language: str, raw: dict) -> LanguageVariant:
    if language not in LANGUAGES:
        raise ValueError("Unknown language key: {0}".format(language))
    turns = raw.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Variant {0} must have a non-empty turns list".format(language))
    return LanguageVariant(
        language=language,
        turns=[str(t) for t in turns],
        source=str(raw.get("source", "mt")),
        verification_status=str(raw.get("verification_status", "mt_only")),
        verified_by=raw.get("verified_by"),
        notes=raw.get("notes"),
    )


def _parse_scenario(raw: dict) -> Scenario:
    try:
        gold_raw = raw["gold"]
        gold = GoldLabels(
            expected_specialty=gold_raw.get("expected_specialty"),
            expected_state=gold_raw.get("expected_state"),
            expect_followup=bool(gold_raw["expect_followup"]),
            expect_nonzero_providers=bool(gold_raw["expect_nonzero_providers"]),
            expect_emergency_routing=bool(gold_raw["expect_emergency_routing"]),
            expected_preferred_language=gold_raw.get("expected_preferred_language"),
        )
        scenario_id = raw["id"]
        category = raw["category"]
        dimension = raw["dimension"]
        variants = {
            lang: _parse_variant(lang, variant_raw)
            for lang, variant_raw in raw["variants"].items()
        }
    except KeyError as exc:
        raise ValueError("Malformed scenario, missing required key: {0}".format(exc)) from exc
    if "en" not in variants:
        raise ValueError("Scenario {0} missing english seed".format(scenario_id))
    return Scenario(
        id=str(scenario_id),
        category=str(category),
        dimension=str(dimension),
        gold=gold,
        variants=variants,
    )


def load_scenarios(path: str = "eval/data/scenarios.json") -> List[Scenario]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = [_parse_scenario(item) for item in payload["scenarios"]]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario ids detected")
    return scenarios
