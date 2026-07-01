# Harness Run Log

Per-run summaries of the fairness-eval harness. Raw per-cell results for each run
are preserved under `eval/runs/<date>-<name>.jsonl` (the live `eval/results.jsonl`
is overwritten every run and gitignored). Rates are **deterministic-check** pass
rates; the LLM-as-judge scoring arrives in Milestone 2.

---

## 2026-07-01 — multilingual 15×5 (first full matrix)

- **System under test:** `openai/gpt-oss-20b` (HF Inference).
- **Translations:** `Qwen2.5-72B-Instruct` (mt_only; Chinese author-verifiable). Numeral-edge variants (s02 glued ZIP `儿科10013`, s13 Arabic-Indic `٠٢١٣٩`) hand-adjusted.
- **Dataset:** 15 scenarios × 5 languages, on branch `fairness-eval-harness` after the F1 (primary-care) fix.
- **Raw cells:** `eval/runs/2026-07-01-multilingual-15x5.jsonl`

### Per-language pass rate (deterministic checks)

| lang | pass rate | checks | scenarios fully pass | errors |
|------|-----------|--------|----------------------|--------|
| en   | **92%**   | 39/42  | 13/15                | 0      |
| es   | **92%**   | 39/42  | 13/15                | 0      |
| ko   | **85%**   | 36/42  | 11/15                | 0      |
| zh   | **83%**   | 35/42  | 11/15                | 0      |
| ar   | **69%**   | 29/42  | 8/15                 | 0      |

**Headline:** a clear equity gap — **Arabic 69% vs English/Spanish 92% (−23 pts)**. Chinese 83%, Korean 85%.

### Scenario × language (OK / x = fail)

```
scenario                        en  zh  es  ar  ko
s01-cardiology                  OK  OK  OK  OK  OK
s02-pediatrics-gluedzip         OK  OK  OK  OK  OK
s03-mentalhealth-ambiguous      x   x   x   x   x
s04-obgyn-fallback              OK  OK  OK  OK  OK
s05-emergency-chestpain         OK  OK  OK  OK  OK
s06-emergency-baby              OK  OK  OK  OK  OK
s07-location-sanjose            OK  x   OK  x   OK
s08-location-la                 OK  x   OK  x   x
s09-langconcordant-spanish      OK  OK  OK  x   x
s10-langconcordant-korean       OK  OK  OK  OK  OK
s11-colloquial-heart            OK  OK  OK  x   OK
s12-colloquial-dizzy            x   x   x   x   x
s13-numeralzip-ophthalmology    OK  OK  OK  x   OK
s14-multiturn-specialist        OK  OK  OK  OK  OK
s15-multiturn-children          OK  OK  OK  OK  OK
```

### Attribution (observational, vs the EN control)

- **NOT translation (fail in every language including EN):** s03, s12. These are the
  pre-localized app findings (F2/F3 — ambiguous-query handling). Correctly *excluded*
  from the fairness signal — a naive reader would blame the LLM/translation; the EN
  control proves they are downstream (Layer B) and language-independent.
- **LLM language layer (A1/A2) — EN passes, non-English fail:**
  - s07, s08 (location phrased in-language), s09 (language-concordant), s11 (colloquial), s13 (native-numeral ZIP).
  - **Arabic hit hardest** (5 of these). Notably **s13**: the Arabic-Indic ZIP `٠٢١٣٩` fails *only* in Arabic → the model doesn't parse native-script digits.

### Caveats
- Deterministic checks only (no LLM-judge yet). Small self-made dataset. Translations mt_only except Chinese.
- Attribution here is *observational*; counterfactual confirmation (inject EN intent to split A1 vs A2) is Milestone 3.
