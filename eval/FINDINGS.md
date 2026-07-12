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
**DISSOLVED AT SOURCE (2026-07-10).** The F6 disclosure line removed the judge's
objection (10/10 faithfulness flips on the s09/s10 cells, 2026-07-09 run), and on
a fresh 20-cell blinded human subset against the v4 replies the judge and the
author are unanimous on faithfulness (20/20 agreement, all five languages, zero
disagreements — RUNS.md v4 addendum). The divergence no longer reproduces; the
lesson stands for future rubric design (keep groundedness and disclosure as
separate axes), but no calibration correction is currently needed.

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
**Update (2026-07-10 run):** the merge-layer reset is fixed — `_merge_parsed_queries`
now sources the conversation language from the full-history parse
(regression-tested), and s15 language-appropriateness flipped True in es (a genuinely
localized render) and in ar/ko (with a caveat: those replies still render the English
wrapper, which the judge passed anyway — see RUNS.md). The residual s15/zh failure is
upstream of the merge and is its own finding: F11.
**CLOSED (2026-07-10, v4).** The remaining coverage asymmetry F8 hid (3-language
in-code copy / 7-language footer / LLM-for-the-rest) is gone: all six non-English
known languages now render from locale files committed to the repo and generated
from the English masters (`care/generate_locales.py` → `care/locales/*.json`,
machine-translated and disclosed as such; zh is the one author-verifiable file;
each non-English footer carries a localized "auto-translated from English" mark).
The long-tail LLM pass (Czech, …) now retries once, rejects English echoes, and
records a `localization_fallback` trace field — the v4 matrix recorded zero
fallback events. English is the single hardcoded copy source; editing it and
re-running the generator regenerates every language.

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
the ZIP+4 unhyphenated — `CA 981011234` — which the old regex could not
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

### F11 — the full-history parse misdetects conversation language when the last turn is language-neutral (Layer A1, open)
With the merge fixed (F8 update), the conversation language comes from the
full-history interpret call — which turns out to be unreliable in exactly the
scenario the fix targets: a follow-up turn that is a bare ZIP code. Payload-level
instrumentation of the interpret call (identical parameters and prompts,
temperature 0) shows `detected_language` flipping between `zh` and `en` for the
same two user turns, depending only on the wording of the assistant's intervening
reply — which is itself sampled at temperature 0.3, so the s15/zh cell is
bistable from run to run (see RUNS.md 2026-07-10). Two contract gaps compound
here: (a) the interpret prompt never states that `detected_language` means the
*user's* conversation-level language, so nothing anchors it against a
language-neutral final turn; (b) the parse emits inconsistent language spellings
("zh", "Chinese", "en", "English") while the merge's absent-value sentinel is the
lowercase literal `"unknown"` only, so a differently-spelled sentinel would pass
as a real language. Candidate fix (deferred, W-item): harden the interpret prompt
contract — define `detected_language` as the user-conversation language with a
fixed value set and sentinel — then re-measure s15 stability across repeated
captures. Related judge observation: on the ar/ko s15 cells the judge passed
English-rendered replies (multi-turn leniency), so the B4 re-label deliberately
includes multi-turn cells.

**FIXED (2026-07-10, v4).** Three layers, because the prompt alone could not be
trusted (the F2 lesson): (a) the interpret prompt now defines
`detected_language` as the USER's conversation-level language with an
English-name vocabulary and an exact lowercase `unknown` sentinel, plus an
explicit rule that numeric/ZIP-only messages carry no language signal; (b) the
merge judges language absence case/variant-tolerantly
(`_is_unknown_response_language`, catching "Unknown"/"N/A"/"undetected"); (c) a
deterministic script backstop overrides the parse when a multi-turn
conversation's latest message has no letters and the merged language resolved
to English/unknown — Han/Hangul/Arabic majority over the user's own messages
decides, so zh/ar/ko retention no longer depends on the parse at all. Measured
as stability, not a single pass: 20/20 fresh s15 captures render localized at
the shipped commit (40/40 including a mid-branch gate). Latin-script languages
(es/vi/tagalog) cannot be distinguished from English by script and ride the
hardened contract — disclosed limit.

### F12 — the location field's contract was underspecified in both directions (Layer A1, FIXED)
The interpret prompt's original location comment ("city/region or null") left
the model free to transform what the user wrote, and it failed in BOTH
directions, each exposed by a different scenario family:
- **ZIP → city ("city-ification", first diagnosed on s29 in the 2026-07-09
  run):** the parse sometimes returned a ZIP as a city name — sometimes the
  wrong city entirely ("cardiology 19103" → "Upper Darby, PA") — and the
  bare-city search path returns zero where the ZIP returns results. Which
  cells hit it shifted with unrelated prompt edits: this branch's
  conversation-language rewrite deterministically flipped s14 into the failure
  (old prompt kept the ZIP 4/4; new prompt city-ified 4/4) — caught by the
  matrix's English-parity gate, since s14/en regressed.
- **City → verbatim (over-correction, caught one capture later):** fixing the
  first direction with "exactly as the user wrote it" made city names stay in
  the user's script (or keep neighborhood qualifiers like "downtown …"), which
  the English-only provider API cannot search — s07/s08 regressed across
  languages including the English control.

**FIXED (2026-07-10, v4)** with an asymmetric contract: a ZIP code is
extracted verbatim and never replaced by a city or neighborhood name; a city
name is normalized to standard English "City, ST" (translated, abbreviations
expanded, qualifiers dropped). Verified in both directions before capture
(20/20 probes across five languages), then measured: +18 checks — the entire
city-ification cluster closed (s01/ko, s07 zh/ar, s08 zh/ar/ko, s29 zh/ar,
s12/ar). Two lessons recorded: a prompt edit is a behavior change to EVERY
field the prompt governs, so A/B the untouched fields too (on both the
full-history and latest-only parses); and an English-control parity gate in
the eval is what turned both regressions from silent production drift into
same-day diagnoses.

### F13 — Arabic symptom phrasing is STOCHASTICALLY over-triaged to emergency (Layer A1, OPEN)
s11-colloquial-heart ("my heart keeps skipping beats" + a ZIP; gold expects a
normal cardiology search) fails only in Arabic across the archived snapshots:
the Arabic variant parsed `urgency=emergency` in every measurement of two
earlier sessions (5/5, then the v4 capture), so the app emergency-routed and
never searched. Counterfactual probes were run with interpretations
pre-registered in a local spec BEFORE execution — and **none matched**, which
is reported as such rather than reinterpreted silently: on probe day the same
dataset phrase parsed emergency only 3/9, flipping within minutes at
temperature 0, while the literal English back-translation parsed emergency 0/9
(and the dataset's English variant passes in every archived snapshot). Three
assistant-generated Arabic paraphrases of the same complaint (disclosed: not
verified by a native speaker) parsed non-emergency 3/3 each on probe day —
small samples under F14's nondeterminism, so phrase-level conclusions are
deliberately not drawn. The defensible attribution: **the emergency/non-emergency
boundary is unstable specifically on the Arabic input** — the English side
sits firmly on the correct side (the literal back-translation 0/9 on probe
day; the dataset's own English variant, a distinct phrasing, non-emergency in
every archived snapshot), while the Arabic phrase oscillates (3/9 to 6/6
depending on the day). The fairness harm is
therefore probabilistic availability loss: an Arabic speaker asking the exact
question an English speaker asks receives, some fraction of the time, only 911
guidance instead of the cardiologist list — and which fraction varies with
serving conditions (F14). Status OPEN: attribution only, no fix shipped; the
fix decision is the owner's (if prompt-side, it carries the F12 verification
tax and should ride the planned clarifier work). Raw probe outputs archived
locally with the eval scratch records.

### F14 — "temperature 0" is not deterministic on this serving stack (Layer: the eval itself, OPEN as a standing caveat)
During the F13 probes, the identical interpret request (same prompt, same
input, same code, temperature 0) returned different parses within the same
minute — and a behavior measured as "5/5 at temperature 0" in one session
reversed to "0/3, then 3/6" in another. The inference route explains it: the
hosted router distributes requests across serving backends (a request captured
during probing was served by a third-party provider), and backend differences
(quantization, batching, samplers) make temperature-0 outputs
non-reproducible across calls, let alone across days. Consequences, recorded
honestly: (a) the "reproduced 5/5 at temperature 0 → stable behavior" wording
in the v3/v4 run entries relied on a determinism assumption the
infrastructure does not provide — correction notes are appended to those
entries (the originals are preserved, not rewritten); the underlying failures
were real in their snapshots, but "stable vs transient" classifications made
from single-session repetition are weaker than stated. (b) Repeated sampling
is the only defensible stability evidence on this stack — the 20/20
fresh-capture gate used for the multi-turn language fix is the pattern to
reuse, and single-capture cells in any snapshot carry an implicit ±
serving-variance error bar. (c) This is the F10 lesson extended one layer
down: not just the harness's service configuration but the SERVING
INFRASTRUCTURE is part of the system under test.

## Method note
Findings F1/F3 sit in Layer B (our code) and F2 in Layer A1 (the LLM). The LLM
understood the English inputs correctly in every case — the failures are
downstream of the parse. So any of these that recur in non-English variants are
**not** translation faults; the harness has already localized them.
