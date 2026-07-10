"""Load committed locale files (care/locales/*.json) produced by
care.generate_locales. English masters live in code; these files carry every
other known language and deploy with the code (restart-proof, reviewable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

LOCALES_DIR = Path(__file__).parent / "locales"


def load_locales() -> Dict[str, dict]:
    locales: Dict[str, dict] = {}
    if not LOCALES_DIR.is_dir():
        return locales
    for path in sorted(LOCALES_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for required in ("copy", "sentences", "trust_guidance", "auto_translated_mark"):
            if required not in data:
                raise ValueError("locale file {0} missing '{1}'".format(path.name, required))
        locales[path.stem] = data
    return locales
