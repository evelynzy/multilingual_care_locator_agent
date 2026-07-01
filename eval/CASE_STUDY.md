# Attribution-Aware Multilingual Fairness Evaluation — Case Study

**One line:** I built an evaluation harness for a multilingual healthcare-provider
locator, measured a large cross-language quality gap (Arabic **69%** vs English
**92%**), and — the point of the exercise — **localized each failure to the layer
that caused it**, showing the gap is the LLM's Arabic *location/numeral* handling,
not medical translation, while three other "failures" trace to our own code or to
scoring strictness rather than to language at all.

---

## 1. Problem

The app takes a free-text care request *in any language* (`"儿科 10013"`,
`"طب العيون ٠٢١٣٩"`), has an LLM extract a structured intent, searches an
English-only US provider registry (NPI), and renders results in the user's
language. The obvious question for a responsible-AI review: **do non-English
users get worse results?** The harder, more useful question: **when they do,
whose fault is it — the LLM, our deterministic code, or the English-only API?**
"Arabic looks worse" is not actionable; "Arabic fails because the model doesn't
convert native-script digits" is.

## 2. Method

**The pipeline has a natural language pivot.** The intent schema forces the
extracted `specialties` to be *English*. So once the request becomes a
`ParsedCareQuery`, everything downstream (search, ranking, API) is
English and **language-invariant by construction**. That yields four attributable
layers:

- **A1** — LLM input understanding (translate user text → English intent)
- **B** — our deterministic code (taxonomy mapping, query build, ranking)
- **C** — the English-only NPI API
- **A2** — LLM output rendering (results → user's language)

**Design:** 15 scenarios × 5 languages (en/zh/es/ar/ko). English is the
**control**. Each cell runs the *real* app path while capturing the intermediate
artifacts (intent, search request, results). Scoring is **deterministic** against
per-scenario gold labels (specialty resolved? provider in the right state?
follow-up asked? nonzero results? emergency routed? language-concordant?). No
LLM-judge yet (that's future work).

**Attribution rule (English-pivot invariant):** if a non-English cell produces the
*same English intent* as its English twin but a *different result*, the divergence
**must** be upstream in the parse (A1) — because B and C are language-invariant.
This turns attribution into a claim you can defend, not a guess.

## 3. Result — the equity gap

| lang | deterministic pass rate | scenarios fully passing |
|------|-------------------------|-------------------------|
| en   | **92%** | 13/15 |
| es   | **92%** | 13/15 |
| ko   | **85%** | 11/15 |
| zh   | **83%** | 11/15 |
| ar   | **69%** | 8/15 |

**Arabic is 23 points below English/Spanish.** (Raw data:
`eval/runs/2026-07-01-multilingual-15x5.jsonl`; run log: `eval/RUNS.md`.)

## 4. Attribution — what the gap actually is

**The Arabic gap is the LLM's Arabic location/numeral handling (A1), not medical
translation.** Every Arabic specialty parsed correctly (pediatrics, dentistry,
ophthalmology, cardiology). The failures, captured directly:

| scenario | specialty parsed | what diverged (AR vs EN) | layer |
|---|---|---|---|
| s13 numeral ZIP | ✅ ophthalmology | `location='٠٢١٣٩'` — native digits never converted to `02139`; the ZIP regex is Western-only → no location → 0 results | **A1** (with a B-side robustness gap) |
| s08 Los Angeles | ✅ dentistry | `location='Los Angeles'` — the **state was dropped** (EN kept `Los Angeles, CA`) | **A1** |
| s07 San Jose | ✅ pediatrics | `location='San Jose, California'` — full state name instead of `CA`, which the pipeline doesn't normalize | **A1** |
| s11 colloquial heart | ✅ cardiology | Arabic colloquial phrasing never reached a search | **A1** |
| s09 language-concordant | — | Arabic returned `family medicine` (10 providers, arguably *more* NPI-correct); the substring scorer wanted `primary care` → **false fail** | **scoring artifact** |

**Failures that are NOT about language at all** (they fail in *every* language,
including English — so the English control pre-localizes them):

- **s03 "mental health", s12 "dizzy"**: the app searches an ambiguous umbrella /
  multi-specialty term and returns 0 instead of asking a clarifying question. This
  is a **Layer-B** app behavior, language-independent. A naive reading of the
  multilingual grid would blame the LLM; the control proves otherwise.

## 5. Findings (and one fix)

- **F1 (fixed):** `"primary care"` returned **0 providers** — NPI has no such
  taxonomy; those doctors are under Family Medicine / Internal Medicine. Our code
  sent the umbrella term literally. Fixed by mapping it to `family medicine`
  (NPI's own suggest endpoint unhelpfully echoes the umbrella back, so the map has
  to win). English baseline 88% → 92%; verified live.
- **F2:** the model's self-reported `needs_clarification` flag is **unreliable** —
  `gpt-oss-20b` sets it `True` for essentially every query. Wiring app behavior to
  it collapsed the baseline 92% → 48%; reverted. *You cannot trust a model's
  self-reported uncertainty here.*
- **F3 (Arabic A1):** the cross-language gap is native-numeral + state-format
  handling in the Arabic parse (section 4).
- **F4 (method):** deterministic checks produce **false negatives** on
  semantically-correct-but-differently-worded outputs (s09 `family medicine` vs
  `primary care`). So the raw 69% Arabic figure slightly *overstates* the true
  disparity — which is exactly why an **LLM-judge with human validation** is the
  right next layer.

## 6. Honest limitations

- Small, self-made dataset (15 scenarios). Illustrative, not a benchmark.
- Deterministic scoring only; no LLM-judge yet (and F4 shows why one is needed).
- Non-English variants are machine-translated (Qwen, `mt_only`); Chinese is
  author-verifiable, the rest are not human-verified.
- Attribution here is **observational** (trace-diff + the English-pivot invariant).
  Counterfactual confirmation (inject the English intent and watch the failure
  vanish) is the stronger form and is scoped as the next milestone.

## 7. What I'd do next

1. **LLM-judge + human-labeled validation (Cohen's κ)** — an independent judge
   (cross-lineage from the system's model) scoring output quality/safety, to catch
   the F4 class of false negatives and add a quality/safety dimension.
2. **Counterfactual attribution** — inject the English twin's intent into the
   Arabic failures to upgrade A1 claims from *shown* to *proven*.
3. **Fix the Arabic A1 failures** — normalize native-script digits and state
   names (a prompt or a normalization step), then re-run and measure the gap close.
4. **Scale** the dataset across more specialties/languages once the above lands.
