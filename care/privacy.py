"""Deterministic PHI input guard: scan and redact structured identifiers.

Pure functions, no I/O, no LLM. Applied by CareLocatorAgent.handle_request
before any text reaches the inference service. Placeholder tokens match no
detector, so redaction is idempotent (history is safely re-redacted every
turn). ZIP codes (5-digit and hyphenated ZIP+4) are never flagged — the app
needs them for provider search.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

PHI_PLACEHOLDERS: Dict[str, str] = {
    "ssn": "[REDACTED: SSN]",
    "phone": "[REDACTED: PHONE]",
    "email": "[REDACTED: EMAIL]",
    "date": "[REDACTED: DATE]",
    "id_number": "[REDACTED: ID NUMBER]",
}

# Detector order matters: earlier patterns claim their spans first and the
# claimed spans are masked before later patterns run.
_DETECTORS = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # 3-2-4 digit groups; shape is distinct from ZIP+4 (5-4).
    ("ssn", re.compile(r"(?<!\d)\d{3}[-. ]\d{2}[-. ]\d{4}(?!\d)")),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:"
            r"(?:\+?1[-. ]?)?(?:\(\d{3}\)[-. ]?|\d{3}[-. ])\d{3}[-. ]\d{4}"  # US shapes
            r"|\d{3}[-. ]\d{4}[-. ]\d{4}"  # 3-4-4 grouping (CN/KR mobiles)
            r"|\d{10,11}"  # bare 10-11 digit runs
            r")(?!\d)"
        ),
    ),
    # Numeric date triplets with a plausible year (year-first or year-last).
    (
        "date",
        re.compile(
            r"(?<!\d)(?:(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
            r"|\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2})(?!\d)"
        ),
    ),
    # Contiguous digit runs of 6+ not claimed above (member IDs / MRNs).
    # 5-digit ZIPs never match (>=6); hyphenated ZIP+4 splits into runs of
    # 5 and 4, so it never matches either. Bare 9-digit runs land here.
    ("id_number", re.compile(r"(?<!\d)\d{6,}(?!\d)")),
)


def fold_digits(text: str) -> str:
    """Fold every Unicode decimal digit to its ASCII equivalent (length-preserving).

    Arabic-Indic ٠-٩, Extended Arabic ۰-۹, fullwidth ０-９, etc. Python's ``\\d``
    already matches these, but ASCII digit literals inside patterns (e.g. the
    date detector's ``(?:19|20)`` year anchor) and consumers of matched VALUES
    (e.g. ZIP extraction feeding an English-only API) do not — folding closes
    both gaps (FINDINGS F9). Deliberately limited to DECIMAL digits
    (``isdecimal``, Unicode Nd): superscripts/circled digits never matched
    ``\\d``, so folding them would create false positives that never existed.
    Everything else passes through unchanged — never raises.
    """
    out = []
    for ch in str(text or ""):
        if ch.isdecimal() and not ("0" <= ch <= "9"):
            value = unicodedata.digit(ch, None)
            out.append(str(value) if value is not None else ch)
        else:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class PHIMatch:
    phi_type: str
    span: Tuple[int, int]
    matched_text: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    matches: Tuple[PHIMatch, ...]


def scan_phi(text: str) -> Tuple[PHIMatch, ...]:
    """Return PHI matches in ``text``, earliest first, non-overlapping."""
    source = str(text or "")
    if not source:
        return ()
    shadow = fold_digits(source)  # length-preserving: spans map back to the original
    matches: List[PHIMatch] = []
    for phi_type, pattern in _DETECTORS:
        for found in pattern.finditer(shadow):
            start, end = found.span()
            matches.append(
                PHIMatch(phi_type=phi_type, span=(start, end), matched_text=source[start:end])
            )
        # Mask claimed spans so later detectors cannot double-claim.
        for match in matches:
            start, end = match.span
            shadow = shadow[:start] + ("\x00" * (end - start)) + shadow[end:]
    matches.sort(key=lambda m: m.span)
    return tuple(matches)


def redact_phi(text: str) -> RedactionResult:
    """Replace every PHI match with its placeholder token."""
    source = str(text or "")
    matches = scan_phi(source)
    if not matches:
        return RedactionResult(text=source, matches=())
    pieces: List[str] = []
    cursor = 0
    for match in matches:
        start, end = match.span
        pieces.append(source[cursor:start])
        pieces.append(PHI_PLACEHOLDERS[match.phi_type])
        cursor = end
    pieces.append(source[cursor:])
    return RedactionResult(text="".join(pieces), matches=matches)
