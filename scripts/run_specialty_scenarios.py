#!/usr/bin/env python3
"""Run specialty-resolution scenarios through the real app.respond(...) path.

This script is meant for local manual validation. It does not score outcomes;
it just records what the assistant returned for each case. Some scenarios are
multi-turn; scripted follow-up turns are only sent when the assistant appears
to request them.
"""

from __future__ import annotations

import argparse
import importlib
import io
import re
import sys
import traceback
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@dataclass(frozen=True)
class Scenario:
    category: str
    name: str
    turns: tuple[str, ...]
    expected: str


@dataclass(frozen=True)
class RuntimeContext:
    respond: Callable[[str, list[dict]], str] | None
    login_message: str | None
    error_message: str | None
    import_error: str | None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Clear Specialist Intent", "cardiovascular disease 95051", ("cardiovascular disease 95051",), "resolves to cardiology, not PCP"),
    Scenario("Clear Specialist Intent", "heart doctor near 95051", ("heart doctor near 95051",), "resolves to cardiology"),
    Scenario("Clear Specialist Intent", "ear nose throat 10001", ("ear nose throat 10001",), "resolves to ENT"),
    Scenario("Clear Specialist Intent", "sinus specialist san jose ca", ("sinus specialist san jose ca",), "resolves to ENT, not generic primary care"),
    Scenario("Clear Specialist Intent", "obgyn 95051", ("obgyn 95051",), "resolves to OB/GYN"),
    Scenario("Clear Specialist Intent", "women's health doctor 95051", ("women's health doctor 95051",), "leans OB/GYN only if mapping is clear; otherwise asks"),
    Scenario("Clear Specialist Intent", "pediatrician 94110", ("pediatrician 94110",), "resolves to pediatrics"),
    Scenario("Clear Specialist Intent", "child health doctor 94110", ("child health doctor 94110",), "resolves to pediatrics"),
    Scenario("Clear Specialist Intent", "psychiatrist 60614", ("psychiatrist 60614",), "resolves to psychiatry"),
    Scenario("Clear Specialist Intent", "physical therapy 02139", ("physical therapy 02139",), "resolves to physical therapy/rehab, not PCP"),
    Scenario("Layperson / Condition Phrasing", "heart problems 95051", ("heart problems 95051",), "likely cardiology or clarification"),
    Scenario("Layperson / Condition Phrasing", "irregular heartbeat near me", ("irregular heartbeat near me", "95051"), "preserves cardiology intent after location follow-up if needed"),
    Scenario("Layperson / Condition Phrasing", "knee rehab 10001", ("knee rehab 10001",), "physical therapy/rehab"),
    Scenario("Layperson / Condition Phrasing", "tooth pain dentist 30309", ("tooth pain dentist 30309",), "dentistry"),
    Scenario("Layperson / Condition Phrasing", "hearing loss doctor 10001", ("hearing loss doctor 10001",), "likely ENT unless ambiguity triggers follow-up"),
    Scenario("Ambiguity Should Trigger Follow-Up", "pain 95051", ("pain 95051",), "follow-up, not broad misleading search"),
    Scenario("Ambiguity Should Trigger Follow-Up", "mental health 95051", ("mental health 95051",), "does not auto-resolve to psychiatry; clarifies"),
    Scenario("Ambiguity Should Trigger Follow-Up", "behavioral health near san jose", ("behavioral health near san jose",), "does not auto-resolve to psychiatry; clarifies"),
    Scenario("Ambiguity Should Trigger Follow-Up", "therapy 95051", ("therapy 95051",), "clarifies; does not silently convert to psychiatry or PT"),
    Scenario("Ambiguity Should Trigger Follow-Up", "specialist near me", ("specialist near me", "95051"), "asks which specialty; follow-up ZIP path included"),
    Scenario("Ambiguity Should Trigger Follow-Up", "doctor for my child and allergies 95051", ("doctor for my child and allergies 95051",), "clarifies if family is unclear"),
    Scenario("Fallback Must Not Erase Specialist Intent", "cardiology 95051", ("cardiology 95051",), "fallback stays cardiology-oriented"),
    Scenario("Fallback Must Not Erase Specialist Intent", "obstetrics gynecology 95051", ("obstetrics gynecology 95051",), "fallback does not downgrade to PCP"),
    Scenario("Fallback Must Not Erase Specialist Intent", "ent 95051", ("ent 95051",), "fallback does not collapse to generic directory behavior"),
    Scenario("Fallback Must Not Erase Specialist Intent", "pediatrics 95051", ("pediatrics 95051",), "retry/fallback preserves pediatrics"),
    Scenario("Provider-Side Taxonomy / Variant Survival", "obstetrics gynecology 95051 (provider-side)", ("obstetrics gynecology 95051",), "valid OB/GYN providers survive taxonomy variants"),
    Scenario("Provider-Side Taxonomy / Variant Survival", "cardiology 95051 (provider-side)", ("cardiology 95051",), "cardiology subtype/provider_type variants survive"),
    Scenario("Provider-Side Taxonomy / Variant Survival", "ear nose throat 95051", ("ear nose throat 95051",), "provider taxonomy/label variants still match ENT family"),
    Scenario("Negative Boundary Cases", "urgent care 95051", ("urgent care 95051",), "care setting, not specialist-family rescue"),
    Scenario("Negative Boundary Cases", "imaging 95051", ("imaging 95051",), "does not over-infer specialty unless clearly supported"),
    Scenario("Negative Boundary Cases", "gi 95051", ("gi 95051",), "asks rather than over-guesses if too ambiguous"),
    Scenario("Negative Boundary Cases", "allergy 95051", ("allergy 95051",), "resolves only if query-safe mapping clearly supports it"),
    Scenario("Broad Specialist Browse Reference", "附近专科医生 -> 95051", ("附近专科医生", "95051"), "current browse-style behavior reference case"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run specialty-resolution scenarios through app.respond(...) and print "
            "a per-case report. Some scenarios are multi-turn, and scripted "
            "follow-up turns are only sent when the assistant requests them. "
            "Requires a locally runnable app environment."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to also write the report to.",
    )
    parser.add_argument(
        "--contains",
        help="Run only scenarios whose name or category contains this case-insensitive substring.",
    )
    return parser.parse_args()


def select_scenarios(contains: str | None) -> tuple[Scenario, ...]:
    if not contains:
        return SCENARIOS
    needle = contains.casefold()
    return tuple(
        scenario
        for scenario in SCENARIOS
        if needle in scenario.name.casefold() or needle in scenario.category.casefold()
    )


def load_runtime() -> RuntimeContext:
    try:
        app_module = importlib.import_module("app")
    except Exception as exc:  # noqa: BLE001
        return RuntimeContext(
            respond=None,
            login_message=None,
            error_message=None,
            import_error=f"{type(exc).__name__}: {exc}",
        )
    return RuntimeContext(
        respond=getattr(app_module, "respond"),
        login_message=getattr(app_module, "LOGIN_MESSAGE", None),
        error_message=getattr(app_module, "ERROR_MESSAGE", None),
        import_error=None,
    )


def looks_like_zip_follow_up(message: str) -> bool:
    return bool(re.fullmatch(r"\d{5}(?:-\d{4})?", message.strip()))


def assistant_requested_follow_up(reply: str, follow_up: str) -> bool:
    lowered = reply.casefold()
    if looks_like_zip_follow_up(follow_up):
        location_markers = (
            "zip",
            "zipcode",
            "zip code",
            "postal code",
            "city",
            "state",
            "location",
            "where are you located",
            "where are you based",
            "what area",
            "what's your zip",
            "what is your zip",
            "what zip",
            "nearby area",
            "where should i search",
            "where should i look",
        )
        return any(marker in lowered for marker in location_markers)
    return False


def classify_reply(reply: str, runtime: RuntimeContext) -> str | None:
    normalized = reply.strip()
    if normalized.startswith("⚠️"):
        return "runtime_error"
    if runtime.login_message and normalized == runtime.login_message.strip():
        return "app_error"
    if runtime.error_message and runtime.error_message in normalized:
        return "runtime_error"
    return None


def run_scenario(runtime: RuntimeContext, scenario: Scenario) -> tuple[str, str]:
    if runtime.respond is None:
        return (
            scenario.name,
            "\n".join(
                [
                    f"Category: {scenario.category}",
                    f"Expected: {scenario.expected}",
                    "Status: error",
                    "",
                    "Runtime import failed before scenario execution.",
                    f"Exception: {runtime.import_error}",
                ]
            ),
        )

    history: list[dict[str, str]] = []
    lines: list[str] = [
        f"Category: {scenario.category}",
        f"Expected: {scenario.expected}",
        "Status: completed",
        "",
    ]

    for turn_index, user_message in enumerate(scenario.turns, start=1):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                assistant_reply = runtime.respond(user_message, history)
        except Exception as exc:  # noqa: BLE001
            lines[2] = "Status: error"
            lines.extend(
                [
                    f"Turn {turn_index} user: {user_message}",
                    f"Turn {turn_index} assistant: ERROR",
                    f"Exception: {type(exc).__name__}: {exc}",
                    "",
                    "Diagnostics - Traceback:",
                    traceback.format_exc().rstrip(),
                ]
            )
            debug_output = "\n".join(
                part for part in (stdout_buffer.getvalue().strip(), stderr_buffer.getvalue().strip()) if part
            )
            if debug_output:
                lines.extend(["", "Diagnostics - Captured output:", debug_output])
            return scenario.name, "\n".join(lines)

        reply_status = classify_reply(assistant_reply, runtime)
        if reply_status is not None:
            lines[2] = f"Status: {reply_status}"

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})
        lines.extend(
            [
                f"Turn {turn_index} user: {user_message}",
                f"Turn {turn_index} assistant: {assistant_reply}",
            ]
        )

        debug_output = "\n".join(
            part for part in (stdout_buffer.getvalue().strip(), stderr_buffer.getvalue().strip()) if part
        )
        if debug_output:
            lines.extend([f"Turn {turn_index} diagnostics:", debug_output])
        lines.append("")

        if reply_status is not None:
            return scenario.name, "\n".join(lines).rstrip()

        if turn_index < len(scenario.turns):
            next_turn = scenario.turns[turn_index]
            if not assistant_requested_follow_up(assistant_reply, next_turn):
                lines[2] = "Status: follow_up_not_requested"
                lines.extend(
                    [
                        (
                            "Configured follow-up was not sent because the assistant "
                            "did not appear to request the expected follow-up."
                        ),
                        f"Skipped user turn: {next_turn}",
                        "",
                    ]
                )
                return scenario.name, "\n".join(lines).rstrip()

    return scenario.name, "\n".join(lines).rstrip()


def render_report(results: Iterable[tuple[str, str]]) -> str:
    sections = [
        "# Specialty Scenario Report",
        "",
        (
            "This report was generated by `scripts/run_specialty_scenarios.py` "
            "using the real `app.respond(...)` path."
        ),
        (
            "Some scenarios are multi-turn. Scripted follow-up turns are only "
            "sent when the assistant appears to request them."
        ),
        "",
    ]
    for index, (name, body) in enumerate(results, start=1):
        sections.append(f"## {index}. {name}")
        sections.append("")
        sections.append(body)
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    scenarios = select_scenarios(args.contains)
    if not scenarios:
        print("No scenarios matched the provided filter.", file=sys.stderr)
        return 1

    runtime = load_runtime()
    results = [run_scenario(runtime, scenario) for scenario in scenarios]
    report = render_report(results)
    print(report, end="")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote report to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
