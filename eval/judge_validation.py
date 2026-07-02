from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from eval.judge import DIMENSIONS


def load_human_labels(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["labels"])


def cohens_kappa(judge_labels: List[bool], human_labels: List[bool]) -> Optional[float]:
    n = len(judge_labels)
    if n == 0 or n != len(human_labels):
        return None
    agree = sum(1 for j, h in zip(judge_labels, human_labels) if bool(j) == bool(h))
    po = agree / n
    p_judge = sum(1 for j in judge_labels if j) / n
    p_human = sum(1 for h in human_labels if h) / n
    pe = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def dimension_agreement(
    judge_by_cell: Dict[Tuple[str, str], dict], human_labels: List[dict], dimension: str
) -> dict:
    judge_vals: List[bool] = []
    human_vals: List[bool] = []
    languages = set()
    for row in human_labels:
        human_val = row.get(dimension)
        if human_val is None:
            continue
        key = (row["scenario_id"], row["language"])
        verdict = judge_by_cell.get(key)
        if verdict is None or dimension not in verdict:
            continue
        judge_vals.append(bool(verdict[dimension]))
        human_vals.append(bool(human_val))
        languages.add(row["language"])

    n = len(judge_vals)
    observed = (sum(1 for j, h in zip(judge_vals, human_vals) if j == h) / n) if n else None
    return {
        "dimension": dimension,
        "n": n,
        "observed_agreement": observed,
        "kappa": cohens_kappa(judge_vals, human_vals),
        "languages": sorted(languages),
    }


def agreement_report(
    judge_by_cell: Dict[Tuple[str, str], dict], human_labels: List[dict]
) -> Dict[str, dict]:
    return {dim: dimension_agreement(judge_by_cell, human_labels, dim) for dim in DIMENSIONS}


def judge_by_cell_from_rows(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    by_cell: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        # An errored judge verdict writes judge_<dim>=False for every dimension.
        # Those are not real labels; letting them through would score a judge
        # crash as a confident "fail" and bias kappa. Drop the whole cell.
        if row.get("judge_error") is not None:
            continue
        key = (row["scenario_id"], row["language"])
        dims = {}
        for dim in DIMENSIONS:
            field = "judge_{0}".format(dim)
            if field in row and row[field] is not None:
                dims[dim] = bool(row[field])
        if dims:
            by_cell[key] = dims
    return by_cell
