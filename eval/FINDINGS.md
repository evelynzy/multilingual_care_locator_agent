# Fairness-Eval Findings

Running log of what the harness surfaced. Each finding is localized to a layer
(A1 = LLM input understanding, B = our deterministic code, C = English-only NPI
API, A2 = LLM output rendering) per the attribution model in the design spec.

## English control baseline

- 15 scenarios, single-turn + multi-turn, run through the real `app` path.
- Baseline: **39/42 deterministic checks pass (92%)**, 0 errors.
- The English run is the *control*: a failure here is an app-quality issue, not a
  fairness issue. Crucially, it **pre-localizes** the failure — if the same
  scenario also fails in another language, we already know which layer owns it.

## Findings

### F1 — "primary care" returns zero providers  (Layer B, FIXED)
The LLM correctly extracts `primary care` and it maps to the `primary-care`
family, but the search sends the literal umbrella term to NPI, which has **no
"primary care" taxonomy** — those doctors are filed under Family Medicine /
Internal Medicine. Result: 0 providers for a common request.
- Evidence: `primary care @ 90011 → 0`, `family medicine @ 90011 → 5`.
- Fix: `suggest_specialty_terms` maps the umbrella term to `family medicine`
  (NPI's suggest endpoint echoes the umbrella back, so the map must win over it).
- Verified: s09 flipped 0→10 providers; baseline 88%→92%.

### F2 — the model's `needs_clarification` signal is unreliable  (Layer A1)
`gpt-oss-20b` returns `needs_clarification=True` for **essentially every query**,
including unambiguous ones (cardiology, ob/gyn, pediatrics, ophthalmology all
flagged True). The flag is noise and **cannot drive app behavior** — wiring
clarification to it made the app clarify on every query and collapsed the
baseline 92%→48%. This is why the app was designed to ignore it. Documented and
reverted rather than shipped.

### F3 — ambiguous queries are searched, not clarified  (open)
`s03` ("mental health") and `s12` ("dizzy…room spins") are genuinely ambiguous.
The app searches an umbrella / multi-specialty term → 0 providers → generic
fallback, instead of asking a clarifying question. Because of F2, this can **not**
be fixed by trusting `needs_clarification`. The correct fix is a curated
"ambiguous umbrella → clarify" detector (independent of the LLM flag), analogous
to the F1 umbrella map. Deferred.

### F4 — deterministic scoring produces false negatives  (method)
Semantically-correct-but-differently-worded outputs fail the substring scorer
(e.g. Arabic returned `family medicine`, arguably more NPI-correct than the gold
`primary care`, yet scored a fail). So raw pass rates slightly *overstate*
disparities — the reason an independent LLM-judge with human validation is the
right next layer.

### F5 — most advertised specialties return zero (Layer B; the "AI-bridge" gap, FIXED)
Systematically verifying the app's 22 specialty families found that **search
returns 0 providers at every ZIP** for at least seven — `orthopedics,
endocrinology, pulmonology, rheumatology, nephrology, oncology, physical therapy`
— while `dermatology, gastroenterology, neurology, urology, psychiatry, allergy,
radiology` work. Same root cause as F1: the LLM understands the request
correctly, but the deterministic layer sends the literal term to NPI, which files
these under different taxonomy classifications (NPI has no bare "endocrinology";
it lives under "Internal Medicine, Endocrinology, …").

**This is the project's core value stated as a bug.** An LLM-fronted locator
exists to *bridge* messy human requests → the rigid English-only NPI taxonomy. F5
is exactly where that bridge is unbuilt: the LLM did its half; the last mile
(specialty → NPI-taxonomy search terms) is missing. The fix generalizes F1's
umbrella map — give each `specialty_families` entry its NPI-taxonomy synonyms
(deterministic + reliable), with the LLM as a long-tail fallback. Deferred; would
lift usable coverage from ~14 → 22 families. (These broken specialties are kept
OUT of the fairness dataset so they don't muddy the cross-language signal.)

**FIXED (2026-07-08).** The fix needed both halves of the pipeline, verified by a
live before/after probe through the real service at ZIP 94110:
- *Retrieval:* `_UMBRELLA_TAXONOMY_TERMS` (clinicaltables.py) now rewrites every
  affected family's query terms to live-verified NPI taxonomy names. Internal-
  medicine subspecialties only match under their full display form — e.g.
  "internal medicine, hematology & oncology"; the bare subspecialty
  ("hematology & oncology", "pulmonary disease") returns zero.
- *Ranking gate:* retrieval alone was not enough — candidates fetched under the
  rewritten term were then dropped as `specialty_mismatch`, because
  `specialty_families` couldn't classify NUCC display names: "Orthopaedic
  Surgery" (ae-spelling) mapped to no family, and "Internal Medicine,
  Rheumatology" mapped to primary-care (the comma head wins). Each affected
  family now carries its NUCC display names as aliases; the exact full-string
  match beats the comma-head heuristic.
- Before → after: orthopedics/endocrinology/pulmonology/rheumatology/nephrology
  0 → 5; oncology 2 → 5; physical therapy 3 → 5; controls unchanged at 5.
  Mapping and classification pinned by `tests/test_specialty_umbrella_terms.py`
  and `tests/test_provider_search_foundation.py`.
- Coverage sweep (2026-07-08): all 22 specialty families probed live through
  the real service at ZIP 94110 (dense) and 68508 (low-density). The sweep
  caught a second gap class: practitioner forms ("orthopedist",
  "pulmonologist", "urologist", …) failed either at retrieval (not in the
  umbrella map, and NPI's suggest endpoint does not convert them) or at the
  ranking gate (term missing from the family-alias catalog — even
  "primary care physician", an original F1 map key, had this gap). The LLM
  parse usually normalizes practitioner forms to field names, but captured
  traces show leaks (an "allergist" parse). Fixed: 17 practitioner-form
  entries added to the umbrella map (values live-verified) plus 15
  family-catalog aliases; two invariant tests now pin that every umbrella key
  classifies to a family, and to the same family as its value. After: every
  core phrasing nonzero at 94110. Known limitations recorded: colloquial
  forms ("eye doctor", "cancer doctor") rely on the LLM parse; low-density
  ZIPs (68508) return sparse-to-zero results at exact-ZIP granularity
  regardless of specialty — a location-handling behavior, not a mapping gap.

### F6 — language-concordance requests silently return non-matching providers (Layer B; disclosure gap, FIXED)
When the user states a language requirement the app cannot satisfy — `s09` ("Spanish-speaking
primary care"), `s10` ("Korean-speaking pediatrician") — and no NPI record carries language
data (`languages` is empty across results), the app returned correctly-specialtied providers
**without disclosing that the language requirement was not met**. The reply is *not* unfaithful:
it never claims a provider speaks the language; it simply dropped the stated need silently.
**Fixed:** `CareLocatorAgent._unverified_preferred_languages` flags any requested language no
returned provider is confirmed to speak; `_compose_result_card_response` then renders a localized
"⚠️ we could not confirm that any of these providers speak {languages}" note (en/es/zh copy; ar/ko
fall back to English like the rest of the deterministic card). Providers still show — the reply
just stops implying they meet the language need. Verified by
`tests/test_language_concordance_disclosure.py`. (Surfaced by the author during judge-validation
labeling.)
**Matrix-confirmed (2026-07-09 run):** `judge_faithfulness` flipped fail→pass on all
10 s09/s10 cells across every language — the disclosure resolved the judge's
objection exactly as predicted (see RUNS.md).

### F7 — the judge over-flags faithfulness on the F6 cells (judge-calibration; Layer A judge)
The Qwen judge scored `s09`/`s10` `faithfulness=fail` in every language, conflating the F6
*disclosure* gap with hallucination. Author labels mark them faithful → Cohen's κ on
faithfulness collapses to 0 while language-appropriateness κ = 1.0 (`eval/RUNS.md`, 2026-07-02).
Lesson surfaced by human validation: **faithfulness (are the claims grounded?) and disclosure
(did we admit the unmet need?) are distinct axes** — the judge rubric should split them, and an
aggregate faithfulness score shouldn't be trusted until it does (and until a larger, balanced
labeled set exists).

### F8 — provider-results reply lost any-language localization (Layer A2 regression, FIXED)
Commit `df2362c` ("restore clickable examples and provider cards") switched provider-card
rendering from an LLM pass — which localized the whole reply into the user's detected language
(e.g. a full Czech reply, per an older author screenshot) — to a deterministic template, for
reliable/structured/clickable cards. `2823e93` re-added localization but only for **en/es/zh**, so
**every other language (Czech, Korean, Arabic, Vietnamese, Tagalog, …) silently fell back to an
English wrapper.** This is a chunk of the A2 `language_appropriateness` misses the judge flagged,
and a genuine regression (not a design choice — confirmed by `git log -S` archaeology and a
dated screenshot of the earlier behavior).
**Fixed (hybrid — restores the pre-regression behavior without the old unreliability):** the
provider table stays deterministic and verbatim; for any non-native language,
`_reply_localization_target` detects the target and `_localize_reply_via_llm` runs one LLM pass
that translates the wrapper copy (intro, labels, guidance, safety) while keeping provider names,
addresses, phones, ZIPs, and URLs exactly as written. Falls back to the original reply on any LLM
error (never worse than the English fallback). Verified live: a Czech `68502` query now returns a
fully-Czech reply (incl. the localized "911" line) with the Medicare URL verbatim. Tests:
`tests/test_reply_localization.py`. (Note: re-running the fairness matrix should now lift the ar/ko
`language_appropriateness` scores — deferred.)
**Matrix-confirmed, partially (2026-07-09 run):** single-turn language-appropriateness
is now high everywhere (ar 18/19, zh 17/18 overall), but all four s15 multi-turn cells
still render English — the turn-2 location-only parse ("94110") resets
`response_language` to English before rendering. That is a distinct, still-open
multi-turn language-context gap, not an F8 regression (see RUNS.md).

### F9 — PHI redaction misses Arabic-script dates: the gap hides in ASCII literals (guard fairness, FIXED)
The new deterministic PHI input guard (`care/privacy.py`) was evaluated offline over a
synthetic multilingual corpus (`eval/data/phi_corpus.json`, en/zh/es/ar/ko × five PHI
types plus per-language negatives; `eval/phi_guard_eval.py`). The headline is a
double surprise:
- **Most cross-script detection came free**: Python's `\d` is Unicode-aware, so
  Arabic-Indic SSNs (`١٢٣-٤٥-٦٧٨٩`), phones, member IDs, and fullwidth-digit Chinese
  phones all matched the ASCII-authored patterns — 1/1, 2/2, 1/1 for ar; 2/2 for the
  zh fullwidth phone variant.
- **The residual gap is exactly where an ASCII literal snuck into a pattern**: the
  date detector anchors plausible years with the literal `(?:19|20)`, which cannot
  match `١٩` — ar date detection 0/1 while every other language scored 1/1. The same
  failure class caused the known Arabic-Indic ZIP issue (`eval/CASE_STUDY.md` §4):
  there, Unicode `\d` *matched* the ZIP but returned the Arabic-Indic string, which
  the English-only provider API cannot use.
The lesson generalizes: digit-class regexes are script-neutral in Python, but any
literal digit inside a pattern — and any consumer of the *matched value* — silently
reintroduces an ASCII assumption. Negatives were 0 false positives in every language
(including the Arabic-Indic ZIP `٩٤١١٠`).

**FIXED (2026-07-09).** `fold_digits` (care/privacy.py) folds every Unicode decimal
digit to ASCII before pattern matching. The fold is length-preserving, so match spans
map back to the original text and redaction preserves the user's script (the reply
shows `[REDACTED: …]` in place of `١٩٨٥-٠١-٠٢`, everything else untouched). Before →
after on the committed corpus: ar date 0/1 → 1/1; every other cell was already
complete; negatives (including the Arabic-Indic ZIP `٩٤١١٠`) 0 false positives both
before and after. Full per-language parity is pinned by
`tests/eval/test_phi_guard_eval.py`. The same helper closes the CASE_STUDY §4 ZIP
issue on both of its paths: message-side extraction (`_extract_zip_code` now returns
`94110` for `٩٤١١٠`) and the LLM-returned `location` field (folded at the
`ParsedCareQuery` seam, since that value bypasses extraction entirely and flows to
the English-only provider API). Folding is deliberately limited to decimal digits —
superscript/circled characters never matched `\d`, and folding them would have
created false positives that never existed.
**Matrix-confirmed (2026-07-09 run):** `phi_redacted` passed in all 5 live cells,
including the Arabic-Indic member-ID and SSN variants — no raw synthetic PHI reached
the intent LLM in any capture.

### F10 — the harness measured a service the app never runs (Layer: the eval itself, FIXED)
The 2026-07-09 matrix re-run's first capture zeroed all seven umbrella-family
scenarios — families the service verifiably handles. Counterfactual isolation
(the byte-identical `ProviderSearchRequest` replayed against two service builds:
5 results vs 0) located the fault in the harness, not the app: `eval/run.py`
constructed a bare `ProviderSearchService(clinicaltables_source=ClinicalTablesSource())`
— no NPPES enrichment, no YAML dataset configuration — so umbrella-family
candidates carried too little taxonomy evidence for the ranking gate and every
such cell silently zeroed. A second, subtler instance sat one layer deeper: the
trace capture's state parser assumed the bare source's address format and
returned empty states for every NPPES-enriched record (enriched addresses embed
the ZIP+4 unhyphenated — `CA 981015173` — which the old regex could not
terminate on), mass-failing the `state` metric including on English control
cells. The gap dated to Milestone 1 and stayed invisible for one reason: no
earlier scenario exercised a code path where the two configurations differ.

**FIXED (2026-07-09).** `eval/run.py::build_matrix_agent` now wraps the service
the app itself constructs (config-driven datasets, NPPES enrichment, cache), and
`eval/trace.py::_provider_state` reads the enriched record shapes; both are
pinned by tests so the drift cannot silently return. Lessons recorded: an eval
harness is part of the system under test — instrument the REAL object, not a
lookalike; and every historical number carries its harness's configuration
(the 2026-07-01/02 baselines are bare-config and all comparisons against them
are labeled cross-config in RUNS.md).

## Method note
Findings F1/F3 sit in Layer B (our code) and F2 in Layer A1 (the LLM). The LLM
understood the English inputs correctly in every case — the failures are
downstream of the parse. So any of these that recur in non-English variants are
**not** translation faults; the harness has already localized them.
