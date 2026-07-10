"""Load committed locale files (care/locales/*.json) produced by
care.generate_locales. English masters live in code; these files carry every
other known language and deploy with the code (restart-proof, reviewable).

Fail-fast by design: a malformed or section-missing locale file raises at
import time (rendering/safety load at module top). The files are committed
data validated by tests/test_locale_files.py, so a bad file can only arrive
via a commit the suite already rejects — failing loudly at startup beats
silently serving English."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

LOCALES_DIR = Path(__file__).parent / "locales"


def load_locales(locales_dir: Optional[Path] = None) -> Dict[str, dict]:
    base_dir = Path(locales_dir) if locales_dir is not None else LOCALES_DIR
    locales: Dict[str, dict] = {}
    if not base_dir.is_dir():
        return locales
    for path in sorted(base_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for required in ("copy", "sentences", "trust_guidance", "auto_translated_mark"):
            if required not in data:
                raise ValueError("locale file {0} missing '{1}'".format(path.name, required))
        locales[path.stem] = data
    return locales
