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

### F5 — most advertised specialties return zero (Layer B; the "AI-bridge" gap)
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

## Method note
Findings F1/F3 sit in Layer B (our code) and F2 in Layer A1 (the LLM). The LLM
understood the English inputs correctly in every case — the failures are
downstream of the parse. So any of these that recur in non-English variants are
**not** translation faults; the harness has already localized them.
