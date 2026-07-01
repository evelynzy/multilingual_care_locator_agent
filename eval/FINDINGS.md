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

## Method note
Findings F1/F3 sit in Layer B (our code) and F2 in Layer A1 (the LLM). The LLM
understood the English inputs correctly in every case — the failures are
downstream of the parse. So any of these that recur in non-English variants are
**not** translation faults; the harness has already localized them.
