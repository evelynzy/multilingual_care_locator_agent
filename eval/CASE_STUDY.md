# Attribution-Aware Multilingual Fairness Evaluation — Case Study

**One line:** I built an evaluation harness for a multilingual healthcare-provider
locator, measured a statistically significant cross-language quality gap
(Arabic 24 points below English, McNemar p = 0.002), attributed every failure to
the pipeline layer that caused it, engineered the fixes — mostly *out* of the
model and into deterministic guardrails — and drove the gap to a residual
indistinguishable from zero, with the entire arc (including two self-inflicted
regressions and one broken infrastructure assumption) documented as it happened.

---

## Executive summary

| | 2026-07-01 baseline | v2 (07-09) | v3 (07-10) | **v4 (07-10, current)** |
|---|---|---|---|---|
| English core-15 | 92% | 93% | 93% | **93%** |
| Arabic core-15 | 69% | 81% | 74% | **88%** |
| Arabic−English gap | **−24 pts** | −12 | −19 | **−5 pts** |
| McNemar p (ar vs en) | **0.002** | 0.0625 | — | **0.69** |
| Judge dimensions (all languages) | n/a | mixed | mixed | **100% on all four** |
| Multi-turn language stability | — | — | bistable cell | **40/40 fresh captures localized** |

Five takeaways (each expanded in the body):

1. **The gap was real, and now it isn't measurable.** The baseline Arabic
   deficit was statistically significant (10-vs-0 discordant paired checks,
   p = 0.002, cluster-bootstrap CI excluding zero). After the fixes, Chinese and
   Spanish are *perfectly concordant* with English (zero discordant checks) and
   the Arabic residual (−4.8%, p = 0.69) is indistinguishable from noise at this
   sample size. [§3](#3-results--from-a-real-gap-to-a-residual-within-noise)
2. **Attribution is counterfactual, not correlational.** A four-layer model
   splits every failure between the LLM's input parsing, our deterministic code,
   the English-only provider API, and output rendering — and the load-bearing
   claims were proven by intervention (same request, one variable changed), not
   by eyeballing grids. [§2](#2-method), [§4](#4-methodology-lessons-what-the-harness-caught)
3. **The judge is validated, then honestly bounded.** An independent LLM judge
   was checked against human labels twice (the second round fully blinded):
   once catching a real calibration divergence (κ = 0 on faithfulness — a rubric lesson, not noise),
   once unanimous with the human — which I report as *corroboration, not
   discrimination*, because a snapshot with no failures cannot test a judge's
   ability to catch them. [§4d](#d-the-judge-arc--validate-then-bound-what-validation-can-say)
4. **The harness caught its own mistakes — that is the point of gates.** An
   English-parity gate turned two regressions introduced by my own prompt edits
   into same-day diagnoses, and repeated sampling exposed that "temperature 0"
   is not deterministic on a routed serving stack — after which I corrected my
   own published wording, append-only. [§4b](#b-f12--the-gate-that-caught-me-twice), [§4c](#c-f14--temperature-0-is-a-sampling-distribution-here)
5. **Fairness is engineered, not rented.** The gap closed mainly by moving
   language handling *out* of the model: curated taxonomy maps, committed locale
   files with disclosed machine translation, a deterministic script backstop,
   and an explicit location contract. Those guarantees hold regardless of which
   model sits in front. [§5](#5-fairness-engineering--dont-rent-it-from-the-model)

*Skills exercised: evaluation harness design, counterfactual layer attribution,
LLM-judge validation with Cohen's κ and blinded labeling, paired statistics
(McNemar, cluster bootstrap), harness-fidelity auditing, responsible-AI
disclosure practice.*

*Scope and velocity: a 103-cell evaluation matrix over a real application;
14 numbered findings — 8 fixed and verified, every one documented the day it
was found; a 491-test suite; first measurement to closed gap in nine days of
dated, committed snapshots.*

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

Current scale: 32 scenarios × up to 5 languages = 103 evaluation cells
(the original core-15 scenarios exist in all five languages and carry the
cross-language comparisons); the app itself ships deterministic support for 7
languages. Everything below is reproducible from committed artifacts:
`eval/RUNS.md` (dated run entries), `eval/FINDINGS.md` (numbered findings),
`eval/runs/*.jsonl` (raw per-cell results), and `eval/paired_stats.py`.

## 2. Method

**The pipeline has a natural language pivot.** The intent schema forces the
extracted `specialties` to be *English*. Once the request becomes a structured
query, everything downstream (search, ranking, API) is English and
**language-invariant by construction**. That yields four attributable layers:

- **A1** — LLM input understanding (user text → English intent)
- **B** — our deterministic code (taxonomy mapping, query build, ranking)
- **C** — the English-only NPI API
- **A2** — LLM output rendering (results → user's language)

**English as control.** Every scenario runs in English too. A failure that
reproduces in the English control is an app-quality bug, not a fairness signal
— the control *pre-localizes* it. One failure class — two ambiguous-colloquial-query scenarios — fails in
every language including English; a naive reading of the
multilingual grid would have blamed translation.

**Attribution rule (English-pivot invariant):** if a non-English cell produces
the *same English intent* as its English twin but a *different result*, the
divergence must be upstream in the parse (A1), because B and C are
language-invariant. Where it mattered, this observational rule was upgraded to
**counterfactual proof**: replay the identical request against two service
builds (harness fidelity, [§4a](#a-f10--the-harness-measured-a-service-the-app-never-runs)), or the identical
input against two prompt versions ([§4b](#b-f12--the-gate-that-caught-me-twice)).

**Scoring, two tiers.** Deterministic checks against per-scenario gold labels
(specialty resolved, provider in the right state, follow-up asked, nonzero
results, emergency routed, language preference handled, PHI redacted), plus an
independent LLM judge (a different model lineage from the system under test)
scoring each rendered reply on four binary dimensions — helpfulness, safety,
faithfulness, language-appropriateness. The judge is *validated, not trusted*:
human labeling with strict ingest and Cohen's κ, twice — the second round fully
blinded
([§4d](#d-the-judge-arc--validate-then-bound-what-validation-can-say)).

**Repeated sampling, not single captures.** Late in the project I proved that
temperature-0 calls are not deterministic on the routed serving stack
(`eval/FINDINGS.md` F14). Since then, stability claims require N-of-M repeated
gates (the multi-turn language fix shipped against a 20-of-20 fresh-capture
gate, twice — 40/40 in total), and single-capture cells carry an implicit serving-variance error
bar.

## 3. Results — from a real gap to a residual within noise

Chinese and Spanish now tie English exactly — zero discordant checks against
the control — and the remaining Arabic residual is smaller than this
instrument can resolve. Nine days earlier, Arabic was 24 points behind.

### The trajectory

| snapshot | date | en core-15 | ar core-15 | gap | context |
|---|---|---|---|---|---|
| baseline | 2026-07-01 | 92% (39/42) | 69% (29/42) | −24 pts¹ | first full matrix (pre-fidelity-fix harness config) |
| v2 | 2026-07-09 | 93% | 81% | −12 | after digit-folding, disclosure, localization, coverage fixes; harness fidelity repaired mid-run (cross-config caveat disclosed in `eval/RUNS.md`) |
| v3 | 2026-07-10 | 93% | 74% | −19 | a dataset location refresh exposed new location-handling cells — an honest regression, kept and diagnosed |
| **v4** | **2026-07-10** | **93%** | **88%** | **−5** | after the multi-turn language fix and the location contract |

¹ The 07-01 run entry prints −23; recomputed at consistent rounding (as in the
v2 entry of `eval/RUNS.md`) the series is −24 → −12 → −19 → −5.

v4, all languages: en 96% / es 94% / zh 94% / ko 92% / ar 91% over all cells;
core-15: en 93 / es 93 / zh 93 / ko 90 / ar 88. All four judge dimensions pass
100% of judged cells in all five languages (ar 19/19, en 32/32, es 17/17,
ko 17/17, zh 18/18 per dimension), and the new `localization_fallback`
telemetry recorded zero silent English fallbacks.

### Paired statistics

Reproduce with `PYTHONPATH=. python -m eval.paired_stats <run.jsonl>` — the
script pairs each language's core-15 checks with the English control cell,
reports McNemar's exact test on discordant pairs, and a **scenario-level
cluster bootstrap** CI (checks within a scenario are correlated, so the
effective sample is closer to 15 scenarios than 42 checks; resampling is by
scenario):

```
run: eval/runs/2026-07-01-multilingual-15x5.jsonl
lang  pairs  b(en+/L-)  c(en-/L+)  McNemar-p  gap      95% cluster CI
ar       42         10          0     0.0020   -23.8%  [ -43.2%,   -7.3%]
es       42          0          0     1.0000    +0.0%  [  +0.0%,   +0.0%]
ko       42          3          0     0.2500    -7.1%  [ -17.9%,   +0.0%]
zh       42          4          0     0.1250    -9.5%  [ -23.8%,   +0.0%]
```

```
run: eval/runs/2026-07-09-multilingual-judged-v2.jsonl
lang  pairs  b(en+/L-)  c(en-/L+)  McNemar-p  gap      95% cluster CI
ar       42          5          0     0.0625   -11.9%  [ -25.6%,   +0.0%]
es       42          0          0     1.0000    +0.0%  [  +0.0%,   +0.0%]
ko       42          8          2     0.1094   -14.3%  [ -33.3%,   +5.4%]
zh       42          4          0     0.1250    -9.5%  [ -23.8%,   +0.0%]
```

```
run: eval/runs/2026-07-10-multilingual-judged-v4.jsonl
lang  pairs  b(en+/L-)  c(en-/L+)  McNemar-p  gap      95% cluster CI
ar       42          4          2     0.6875    -4.8%  [ -23.3%,  +11.9%]
es       42          0          0     1.0000    +0.0%  [  +0.0%,   +0.0%]
ko       42          1          0     1.0000    -2.4%  [  -7.3%,   +0.0%]
zh       42          0          0     1.0000    +0.0%  [  +0.0%,   +0.0%]
```

**Reading this honestly.** The baseline Arabic gap was statistically
significant on paired evidence: ten discordant checks, all favoring English
(p = 0.002), CI excluding zero. In v4, Chinese and Spanish are perfectly
concordant with English — literally zero discordant checks — and the Arabic
residual (4 vs 2 discordant, p = 0.69) cannot be distinguished from zero.
**Power statement:** at this sample size (42 paired checks in ~15 correlated
clusters per language), the instrument reliably detects gaps of the baseline's
magnitude — and its detection floor is demonstrated empirically by the v2 run:
an 11.9-point Arabic gap (5-vs-0 discordance) already lands at p = 0.0625,
just short of significance. So gaps around ten points sit at the edge of this
instrument's resolution, and "not significant" must be read as "below the
resolution," never as "proven equal." Certifying the residual would take a
several-fold larger scenario set; that is a stated limitation, not a footnote.

### What moved the numbers (fix attribution)

Each measured movement traces to a numbered finding in `eval/FINDINGS.md` —
and none of them waited for a better model: every row is a code, data, or
contract change I shipped, tested, and re-measured.

| movement | cause (finding) |
|---|---|
| "primary care" and at least seven specialty families returned zero providers | F1/F5 — curated umbrella-taxonomy maps (Layer B) |
| Arabic-Indic digits (`٩٤١١٠`) unusable end-to-end | F9 — length-preserving Unicode digit folding at the input seam |
| Language-concordance requests silently unmet | F6 — explicit disclosure line (judge faithfulness flipped 10/10) |
| Non-es/zh replies fell back to English wrappers | F8 + locale pipeline — 6 committed locale files generated from English masters, machine translation disclosed, visible "auto-translated" mark |
| Multi-turn conversations lost the user's language on a bare-ZIP turn | F11 — prompt contract + case-tolerant merge + deterministic script backstop; shipped against a 20/20 (×2) repeated-capture gate |
| ZIPs parsed into (sometimes wrong) city names; in-language city names unusable | F12 — a two-sided location contract: ZIP verbatim, city normalized to English (+18 checks, closing the whole cluster) |

## 4. Methodology lessons (what the harness caught)

These four episodes are the part of this project I would defend in an
interview; each is documented in `eval/RUNS.md`/`eval/FINDINGS.md` as it
happened, not reconstructed.

### (a) F10 — the harness measured a service the app never runs

The first full re-run zeroed seven specialty families the service verifiably
handles. Counterfactual isolation — the byte-identical search request replayed
against two service builds — returned 5 providers on the app's real service and
0 on the harness's, exposing that the harness had constructed a bare service
(no enrichment, no dataset config) since its first milestone. One layer deeper,
the trace parser assumed the bare service's address format and mass-failed a
metric on English control cells. Both fixed and pinned by tests; every
historical comparison across the boundary is labeled *cross-config* in the run
log. **Lesson: the eval harness is part of the system under test — instrument
the real object, and treat every historical number as carrying its harness's
configuration.**

### (b) F12 — the gate that caught me twice

While shipping the multi-turn language fix, my own prompt edit silently changed
*location* parsing: "cardiology 19103" began resolving to a (wrong) city name
— deterministically under A/B (old prompt kept the ZIP 4/4; new prompt
city-ified 4/4) — and the matrix's **English-parity gate** failed the run
because an English control cell regressed. The fix ("keep the ZIP exactly as
written") then over-corrected: city names started staying in the user's script
(the Chinese variant's own words for "downtown Los Angeles" passed through
verbatim) or keeping neighborhood qualifiers, both unusable by the
English-only API — caught by the same gate one capture later. The final
contract is asymmetric (ZIP verbatim; city → standard English "City, ST"),
verified in both directions before the definitive run, and it closed the
pre-existing city-ification cluster as a side effect (+18 checks). **Lessons: a
prompt edit is a behavior change to every field the prompt governs — A/B the
fields you didn't touch; and an English-parity gate converts silent drift into
same-day diagnoses.**

### (c) F14 — "temperature 0" is a sampling distribution here

A follow-up probe found the same interpret request flipping between parses
within a minute, at temperature 0 — and a behavior measured "5/5 stable" in one
session reversing in the next. The hosted router distributes calls across
serving backends; backend differences make temp-0 outputs non-reproducible.
Consequences I acted on: correction notes appended (never rewritten) to my own
already-published run entries whose "reproduced 5/5 → stable" wording relied on
the broken assumption; repeated N-of-M gates became the only accepted stability
evidence; and the one remaining cross-language failure was re-characterized as
**stochastic** — an Arabic phrasing that sometimes over-triages to emergency
routing (its literal English back-translation: 0/9 on probe day, and the
scenario's English variant passes in every archived snapshot), meaning Arabic users are
*probabilistically* denied the provider list others get (`eval/FINDINGS.md`
F13). **Lesson: the serving infrastructure is part of the system under test,
one layer below F10.**

### (d) The judge arc — validate, then bound what validation can say

The judge was never trusted on authority. First validation round (15-cell
human subset; disclosed caveat: I had seen aggregate judge statistics before
labeling — the second round fixed this with full blinding):
language-appropriateness κ = 1.0 (n = 6, en+zh — the languages I can read) —
but faithfulness κ = 0, a genuine calibration finding: the judge conflated "didn't disclose an
unmet language request" with "hallucinated," a rubric-axis confusion (F7), not
noise. The fix went into the *product* (an explicit disclosure line), the
judge's objection dissolved at source (10/10 cells flipped), and the rubric
lesson — groundedness and disclosure are different axes — stands. Second round
(fresh blinded 20-cell subset against v4, strict ingest, blanks never guessed):
judge and human unanimous on every filled label, including
language-appropriateness across all five languages via a disclosed script-level
identification protocol. I report that as **corroboration, not discrimination**:
with zero variance in either rater and no failing cells in the snapshot, κ is
degenerate and cannot measure the judge's ability to catch failures — the
discriminative evidence remains the first round. **Lesson: judge validation is
a process with a shelf life (labels stale when replies change), and the honest
statistician says what a perfect-agreement result does *not* establish.**

## 5. Fairness engineering — don't rent it from the model

The gap did not close because the model got better; the model never changed.
It closed because language handling moved from "whatever the LLM does" to
deterministic, testable guarantees:

- **Umbrella taxonomy maps** (F1/F5): messy human specialty words → verified
  registry taxonomy terms, curated and pinned by invariant tests.
- **Digit folding** (F9): any Unicode decimal script → ASCII at the input seam,
  length-preserving so redaction spans survive.
- **Committed locale files** (F8/W-series): all six non-English supported
  languages render from files generated once from the English masters —
  machine translation *disclosed in-product* (a localized "auto-translated from
  English" mark on every non-English safety footer) and in the docs; zero
  runtime translation calls for supported languages, and telemetry proving it.
- **Script backstop** (F11): if a multi-turn conversation's latest message has
  no language signal and the parse claims English, character-script evidence
  from the user's own messages overrides it — mathematically independent of the
  parse for non-Latin scripts.
- **Location contract** (F12): ZIP codes verbatim, city names normalized to
  English — stated in the prompt, verified bidirectionally.

The LLM still does what only it can (understanding that "儿科 10013" is a
pediatrics request at a New York ZIP); everything that *can* be a guarantee
*is* one. This
architecture claim is model-independent — swapping in a stronger model shrinks
the residual A1 risks but does not replace the guarantees. (Testing exactly
that swap is scoped future work; see §7.)

## 6. Honest limitations

- **Small, self-made dataset**: 15 cross-language scenarios (42 paired checks)
  per language; the power statement in §3 bounds what it can certify.
  Illustrative and diagnostic, not a benchmark.
- **High-resource languages only** (es/zh/ar/ko): measured disparities are
  plausibly a floor; low-resource languages would fare worse.
- **Machine translation throughout**: scenario variants and locale files are
  `mt_only` except Chinese (the one language I can personally verify); the
  labeling protocol discloses exactly which judgments my language abilities
  support, and blanks were never guessed.
- **Judge**: single-vote, observed flipping ±1 cell between runs on
  re-captured replies; also the same model family as the translation pipeline
  (entanglement disclosed).
- **Serving variance** (F14): every single-capture number carries an implicit
  error bar; only the N-of-M gated claims are stability claims.
- **The script backstop cannot help Latin-script languages** (Spanish rides
  the hardened prompt contract alone) — a structural asymmetry, disclosed.
- **One known open failure**: the stochastic Arabic emergency over-triage
  (F13) — attributed, characterized, not yet fixed.

## 7. What's next (and what "next" meant last time)

The previous version of this document listed four next steps: an LLM judge
validated with κ; counterfactual attribution; fixing the Arabic parse
failures; scaling the dataset. All four landed — with one honest asterisk:
counterfactual attribution arrived as service-build and prompt A/B
interventions (§4a, §4b) rather than the intent-injection experiment as
originally sketched. The current list:

1. **Ambiguity clarifier**: the one failure class that fails the English
   control too (two ambiguous-colloquial-query scenarios search instead of
   asking) — an app fix that
   will lift every language, bundled with removing a schema field the model
   provably cannot be trusted to set (F2).
2. **Cross-model comparison**: the harness is model-agnostic by construction —
   swap the system-under-test model, re-run 103 cells, and answer "does a
   frontier model shrink the residual gaps?" as a measured hypothesis
   (with F14's repeated-sampling discipline), not an assumption.
3. **The F13 fix decision** and the follow-up hygiene pool (backstop edge
   cases, prompt-contract review for the remaining free-text fields).
