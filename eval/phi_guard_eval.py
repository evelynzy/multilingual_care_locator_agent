"""Layer-1 eval of the PHI input guard: detection rates per language x type.

Deterministic, offline, no LLM. Run from the repo root:
    .venv/bin/python -m eval.phi_guard_eval
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Tuple

from care.privacy import scan_phi

DEFAULT_CORPUS_PATH = "eval/data/phi_corpus.json"


def evaluate_corpus(path: str = DEFAULT_CORPUS_PATH) -> Dict[Tuple[str, str], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)["entries"]
    stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "detected": 0, "false_positives": 0}
    )
    for entry in entries:
        language = entry["language"]
        detected = bool(scan_phi(entry["text"]))
        if entry["expected_detected"]:
            key = (language, entry["phi_type"])
            stats[key]["total"] += 1
            stats[key]["detected"] += int(detected)
        else:
            key = (language, "negatives")
            stats[key]["total"] += 1
            stats[key]["false_positives"] += int(detected)
    return dict(stats)


def main() -> None:
    stats = evaluate_corpus()
    languages = sorted({lang for lang, _ in stats})
    phi_types = ["ssn", "phone", "email", "date", "id_number"]
    print("| language | " + " | ".join(phi_types) + " | false positives |")
    print("|---" * (len(phi_types) + 2) + "|")
    for language in languages:
        cells = []
        for phi_type in phi_types:
            cell = stats.get((language, phi_type))
            cells.append("-" if not cell else "{0}/{1}".format(cell["detected"], cell["total"]))
        neg = stats.get((language, "negatives"))
        neg_cell = "-" if not neg else "{0}/{1}".format(neg["false_positives"], neg["total"])
        print("| " + language + " | " + " | ".join(cells) + " | " + neg_cell + " |")


if __name__ == "__main__":
    main()
