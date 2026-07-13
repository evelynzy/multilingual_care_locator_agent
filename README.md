---
title: Multilingual Care Locator Agent
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
hf_oauth: false
license: apache-2.0
short_description: Multilingual healthcare locator agent
sdk_version: 5.47.2
---

This Space hosts a multilingual care navigation assistant built with [Gradio](https://gradio.app) and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index), searching public US healthcare-provider registries.

## What it does
- Understands care requests written in any language via a multilingual LLM, and extracts structured intent (specialty, location, insurance, language preferences)
- Searches the public [NPI registry](https://clinicaltables.nlm.nih.gov/apidoc/npi_idv/v3/) (ClinicalTables + NPPES enrichment) through a curated specialty-taxonomy bridge
- Replies deterministically localized in seven languages — English, Español, 简体中文, العربية, 한국어, Tiếng Việt, Tagalog — with machine-translated copy visibly marked; other languages get a best-effort translation pass
- Redacts personal identifiers (SSNs, phone numbers, member IDs) from the input before any text reaches the LLM, and says so
- Falls back to trusted public resources when no direct matches are found

## Evaluation & case study
This project ships its own fairness evaluation, documented end to end:
- **[eval/CASE_STUDY.md](eval/CASE_STUDY.md)** — start here: how a statistically significant cross-language quality gap was measured, attributed layer by layer, engineered away, and verified (with paired statistics)
- [eval/RUNS.md](eval/RUNS.md) — the run log: numbers and caveats for every evaluation snapshot
- [eval/FINDINGS.md](eval/FINDINGS.md) — fourteen numbered findings, from taxonomy gaps to serving-stack nondeterminism
- [eval/paired_stats.py](eval/paired_stats.py) — reproduce the statistics from the committed run archives

This is an independent, non-commercial open-source project developed on personal time and personal devices. It is not affiliated with, endorsed by, or sponsored by my employer. 
The application uses only publicly available data and APIs (e.g., federal/state public directories). It does not process Protected Health Information (PHI) and is provided for informational purposes only, not as medical, legal, or insurance advice. 
Users should verify provider participation and coverage directly with the provider and plan. 
The code and documentation are released under the Apache License 2.0 (see [LICENSE](LICENSE)).