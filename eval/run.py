from __future__ import annotations

import json
import os
from typing import Dict, List

from eval.dataset import Scenario, load_scenarios
from eval.judge import DIMENSIONS, JudgeVerdict
from eval.scoring import MetricResult, score_trace
from eval.trace import Trace, run_trace

_METRIC_NAMES = (
    "specialty",
    "state",
    "followup",
    "nonzero_providers",
    "emergency_routing",
    "preferred_language",
)


def result_row(trace: Trace, results: List[MetricResult]) -> dict:
    row: Dict[str, object] = {
        "scenario_id": trace.scenario_id,
        "language": trace.language,
        "error": trace.error,
    }
    by_name = {r.name: r for r in results}
    for name in _METRIC_NAMES:
        metric = by_name[name]
        row["{0}_applicable".format(name)] = metric.applicable
        row["{0}_passed".format(name)] = metric.passed
    return row


def judge_fields(verdict: JudgeVerdict) -> dict:
    row = {}
    for dim in DIMENSIONS:
        row["judge_{0}".format(dim)] = getattr(verdict, dim)
    row["judge_error"] = verdict.error
    return row


def summarize(rows: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for row in rows:
        lang = row["language"]
        bucket = summary.setdefault(lang, {"applicable": 0, "passed": 0})
        for name in _METRIC_NAMES:
            applicable_key = "{0}_applicable".format(name)
            if row.get(applicable_key):
                bucket["applicable"] += 1
                if row.get("{0}_passed".format(name)):
                    bucket["passed"] += 1
    for bucket in summary.values():
        applicable = bucket["applicable"]
        bucket["pass_rate"] = (bucket["passed"] / applicable) if applicable else 0.0
    return summary


def run_matrix(
    agent,
    client,
    settings: dict,
    scenarios: List[Scenario] = None,
    out_path: str = "eval/results.jsonl",
    cache_dir: str = "eval/.cache",
    judge=None,
) -> List[dict]:
    if scenarios is None:
        scenarios = load_scenarios()

    rows: List[dict] = []
    for scenario in scenarios:
        for language in scenario.variants:
            trace = run_trace(scenario, language, agent, client, settings, cache_dir=cache_dir)
            results = score_trace(trace, scenario.gold)
            row = result_row(trace, results)
            if judge is not None:
                row.update(judge_fields(judge.score(scenario.id, language, trace)))
            rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    from dotenv import load_dotenv
    from huggingface_hub import InferenceClient
    from config_loader import get_chat_model_settings
    from provider_search.service import ProviderSearchService
    from provider_search.sources.clinicaltables import ClinicalTablesSource
    from eval.instrumented_agent import TracingAgent
    from eval.judge import JUDGE_MODEL, QwenJudge
    from eval.judge_validation import agreement_report, judge_by_cell_from_rows, load_human_labels

    load_dotenv()
    settings = get_chat_model_settings()
    token = os.environ["HF_TOKEN"]
    client = InferenceClient(model=settings["model_id"], token=token)
    judge = QwenJudge(InferenceClient(model=JUDGE_MODEL, token=token))
    agent = TracingAgent(ProviderSearchService(clinicaltables_source=ClinicalTablesSource()))

    rows = run_matrix(agent, client, settings, judge=judge)
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))

    labels_path = "eval/data/human_labels.json"
    if os.path.exists(labels_path):
        report = agreement_report(judge_by_cell_from_rows(rows), load_human_labels(labels_path))
        print(json.dumps(report, indent=2, ensure_ascii=False))
