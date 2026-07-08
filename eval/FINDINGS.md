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

## Method note
Findings F1/F3 sit in Layer B (our code) and F2 in Layer A1 (the LLM). The LLM
understood the English inputs correctly in every case — the failures are
downstream of the parse. So any of these that recur in non-English variants are
**not** translation faults; the harness has already localized them.
