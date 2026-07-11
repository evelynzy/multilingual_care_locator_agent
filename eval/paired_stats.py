"""Paired statistics for the fairness matrix (case-study numbers).

Pairs every non-English core-15 cell's applicable deterministic checks with
the same scenario's English cell, then reports per language: discordant-pair
counts, a McNemar exact test, and a SCENARIO-level cluster-bootstrap CI on the
pass-rate gap. Cluster resampling (not per-check) because checks within a
scenario are correlated — the effective sample size is closer to the 15
scenarios than to the 42 checks.

Reproduce the case-study table:
    PYTHONPATH=. python -m eval.paired_stats [run_jsonl]
(default run: eval/runs/2026-07-10-multilingual-judged-v4.jsonl)
"""
from __future__ import annotations

import json
import math
import random
import sys
from typing import Dict, List, Tuple

DEFAULT_RUN = "eval/runs/2026-07-10-multilingual-judged-v4.jsonl"
CORE = {"s%02d" % i for i in range(1, 16)}
METRICS = (
    "specialty",
    "state",
    "followup",
    "nonzero_providers",
    "emergency_routing",
    "preferred_language",
    "phi_redacted",
)


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _cells(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["scenario_id"], r["language"]): r for r in rows}


def pair_checks(rows: List[dict], language: str) -> List[Tuple[str, str, bool, bool]]:
    cells = _cells(rows)
    pairs: List[Tuple[str, str, bool, bool]] = []
    for (sid, lang), row in sorted(cells.items()):
        if lang != language or sid[:3] not in CORE:
            continue
        en = cells.get((sid, "en"))
        if en is None:
            continue
        for metric in METRICS:
            if row.get(metric + "_applicable") and en.get(metric + "_applicable"):
                pairs.append(
                    (sid, metric, bool(en.get(metric + "_passed")), bool(row.get(metric + "_passed")))
                )
    return pairs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs.

    b = pairs where English passes and the other language fails;
    c = the reverse. Under H0 each discordant pair is a fair coin.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _scenario_counts(pairs: List[Tuple[str, str, bool, bool]]) -> List[Tuple[int, int, int]]:
    by_sid: Dict[str, List[Tuple[bool, bool]]] = {}
    for sid, _metric, en_ok, lang_ok in pairs:
        by_sid.setdefault(sid, []).append((en_ok, lang_ok))
    counts = []
    for sid in sorted(by_sid):
        checks = by_sid[sid]
        counts.append(
            (sum(1 for _, l in checks if l), sum(1 for e, _ in checks if e), len(checks))
        )
    return counts


def cluster_bootstrap_gap(
    rows: List[dict], language: str, n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float, float]:
    counts = _scenario_counts(pair_checks(rows, language))
    total = sum(n for _, _, n in counts)
    point = (sum(l for l, _, _ in counts) - sum(e for _, e, _ in counts)) / total
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        sample = [counts[rng.randrange(len(counts))] for _ in range(len(counts))]
        n = sum(x for _, _, x in sample)
        if n == 0:
            continue
        gaps.append((sum(l for l, _, _ in sample) - sum(e for _, e, _ in sample)) / n)
    gaps.sort()
    lo = gaps[int(0.025 * len(gaps))]
    hi = gaps[min(len(gaps) - 1, int(0.975 * len(gaps)))]
    return point, lo, hi


def main(argv: List[str]) -> int:
    path = argv[0] if argv else DEFAULT_RUN
    rows = load_rows(path)
    languages = sorted({r["language"] for r in rows} - {"en"})
    print("run:", path)
    print("lang  pairs  b(en+/L-)  c(en-/L+)  McNemar-p  gap      95% cluster CI")
    for lang in languages:
        pairs = pair_checks(rows, lang)
        b = sum(1 for _, _, e, l in pairs if e and not l)
        c = sum(1 for _, _, e, l in pairs if l and not e)
        p = mcnemar_exact(b, c)
        gap, lo, hi = cluster_bootstrap_gap(rows, lang)
        print(
            "{0:4}  {1:5}  {2:9}  {3:9}  {4:9.4f}  {5:+7.1%}  [{6:+7.1%}, {7:+7.1%}]".format(
                lang, len(pairs), b, c, p, gap, lo, hi
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
