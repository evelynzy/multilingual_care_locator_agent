# Architecture

A multilingual healthcare-provider locator: users describe what care they need in **any
language** ("儿科 10013", "Najděte mi pediatra v 68502"); the app extracts a structured
intent, searches the English-only US provider registries (NPI), and answers in the user's
language. Runs as a Gradio app on Hugging Face Spaces; the chat model is `openai/gpt-oss-20b`
via the HF Inference API.

## The pipeline and the language pivot

```
user text (any language)
   │  A1 — intent extraction: LLM fills a structured form
   ▼
ParsedCareQuery          ← specialties are forced to ENGLISH by the schema
   │  B — deterministic retrieval (provider_search/)
   ▼
ProviderSearchRequest → ranked provider results
   │  C — English-only external APIs (ClinicalTables NPI, NPPES)
   ▼
provider records
   │  A2 — reply rendering: deterministic cards + localization
   ▼
reply in the user's language
```

**The load-bearing property:** the intent schema forces extracted `specialties` into English,
so everything downstream of `ParsedCareQuery` (search, ranking, external APIs) is
English-only and **language-invariant by construction**. This is what makes cross-language
failures attributable to a specific layer — the basis of the fairness evaluation in `eval/`
(see `eval/CASE_STUDY.md`).

## Stage A1 — input understanding (`care/intent.py`, `care/language.py`)

**Intent extraction** (`_interpret_user_need`). One LLM call with a `care_intent` json_schema
forcing structured output (summary, English specialties, location, insurance, preferred
languages, urgency, care setting, …). Three-attempt armor: schema-enforced call → retry
without `response_format` if the inference provider rejects it → one "strict JSON only"
retry → deterministic rescue. The model's self-reported `needs_clarification` flag is
recorded but never drives behavior — measured to fire on essentially every query
(`eval/FINDINGS.md` F2); clarification decisions are made deterministically instead.

**Repair pipeline** (runs after every parse):
- *Rescue* — if no valid JSON survived, build a minimal payload from the raw message
  (summary = the message, location regex-rescued) so the app degrades instead of crashing.
- *Reconcile* — cross-check the model against the raw message. If the message names two or
  more specialty families, or matches known-ambiguous phrasings (e.g. "child allergy":
  pediatrics vs allergy/immunology), the app **abstains**: it clears the model's guessed
  specialties and asks a clarifying question rather than betting on the wrong specialist.
- *Sanitize (numeric trust boundary)* — numbers are trusted only if the user typed them.
  Procedure-code context is keyword-driven ("cpt", "procedure", "billing code"): in that mode
  a bare 5-digit number is never treated as a ZIP and the location may come only from an
  explicit "City, ST" pattern. In normal mode ZIPs are extracted with an exactly-five-digits
  rule (digit lookarounds, so CJK-glued input like `儿科10013` still parses). A model-returned
  bare ZIP that does not appear verbatim in the user's message is rejected.

**Multi-turn merge.** On any turn with prior history the app parses twice — full history and
latest-message-only — and merges field-by-field: scalars (location, urgency, care setting)
prefer the latest turn; list fields union without duplicates; specialties take the latest
turn's answer when it names any. Answering a "which specialty?" question with only a ZIP keeps
the earlier turn's specialty. The app is stateless between turns: every turn re-derives intent
from the visible transcript, which lets later turns correct earlier misreads.

**Language determination.** The LLM reports `detected_language`/`response_language`; a guard
overrides "respond in English" whenever a non-English input was detected (small models drift
toward English). Downstream, an NFKD-normalized alias map (`"中文"`, `"한국어"`, `"es"`, …)
resolves the language for each rendering subsystem.

## Routing and guidance (`care/guidance.py`, `care/safety.py`)

`_build_navigation_guidance` decides the reply mode — `emergency`, `search`, or
`clarification` — via a priority ladder in `_classify_care_setting`:

1. **Emergency** (checked first, short-circuits before any search): two OR'd detectors —
   deterministic English keyword patterns (chest pain, can't breathe, stroke, 911 variants …)
   and the model's parsed urgency/care-setting. For non-English input, detection rides on the
   model's English-normalized parse; emergency parity across languages is exercised by the
   fairness eval (s05/s06).
2. Urgent-care / specialist / primary-care classification shapes an advisory "care route"
   line; specialist intent attaches a referral note (HMO/POS often require referrals; PPO may
   not — confirm with the insurer).
3. **Clarification**: a fixed bank of five curated questions (location, care type, insurance
   plan, preferred language, plan type), each with a deterministic trigger. Clarifying replies
   are LLM-composed so the questions render in the user's language.

**Location parsing** uses dedicated assets in `care/intent.py`: US state code and full-name tables, a noise-token
list (words like "plan"/"find"/"area" can never become city names), specialty-word rejection
("dermatology, CA" does not invent a city), and city/state regexes shared with the trust
boundary above.

## Stages B/C — retrieval (`provider_search/`)

A self-contained package with single-purpose modules:

- **`service.py`** — orchestration: normalize the request → PHI-free fingerprint → cache
  check → plan source-request variants → collect candidates → retry plans when results are
  thin (including nearby-location retries) → field-level dedupe/merge → rank → trace.
- **`sources/clinicaltables.py`** — the NPI ClinicalTables adapter. Includes the
  umbrella-taxonomy map: NPI has no taxonomy for terms like "primary care" (those providers
  file under Family Medicine), so umbrella terms are rewritten to NPI-recognized taxonomies
  before search (`eval/FINDINGS.md` F1; generalizing this map to all specialty families is
  planned — F5).
- **`sources/nppes.py`** — NPPES registry enrichment per provider (addresses, taxonomies,
  Medicare opt-out).
- **`ranking.py`** — weighted scoring. Specialty alignment dominates (4.0 per match);
  keywords 1.75; language and insurance alignment 1.5; location is binary with a hard rule
  (requested state ≠ provider state → zero); small bonuses for verified-trust metadata.
- **`specialty_families.py`** — the canonical catalog of 22 specialty families and aliases,
  shared by request normalization, message-evidence scanning, and provider gating (see
  `docs/specialty-resolution-architecture.md`).
- **`cache.py`** — PHI-free response cache keyed by request fingerprint, storing provider IDs
  only.

**Fallback chain** (agent-side): when search returns nothing and a location is known, a
curated resource list from `config/settings.yaml` (filtered by specialty/care-setting/region,
e.g. Medicare Care Compare) is offered instead; source failures and missing-location cases get
explicit explanatory notes. The service-level `fallback_resources` field is a reserved seam —
currently always empty (see the comment in `service.py`).

## Stage A2 — output (`care/rendering.py`, `app.py`)

**Deterministic card renderer.** Provider results render without an LLM: intro, care-route and
referral notes, one HTML provider card per result (address, phone, source, why-matched), and a
trust-badge row that never over-claims ("Informational", "Network unverified", "New patients
unknown", "Appointments unverified"). Provider data is inserted verbatim.

**Localization.** Card template copy is pre-written in English/Spanish/Chinese; for any other
detected language a single LLM pass (`care/language.py`) translates the wrapper text (headings, labels, guidance,
safety notes) while keeping provider names, addresses, phones, ZIPs, and URLs exactly as
written, falling back to the original reply on any failure. This restores full any-language
replies (`eval/FINDINGS.md` F8) on top of reliable structured cards.

**Language-concordance disclosure.** If the user asked for a provider who speaks language X
and no returned record confirms it (NPI rarely carries language data), the reply says so
explicitly — "we could not confirm that any of these providers speak X" — instead of silently
implying the need was met (`eval/FINDINGS.md` F6).

**Safety footer.** Every reply, on every composition path, ends with the safety block from
`care/safety.py` (care navigation only, nothing verified unless stated, call to confirm, don't
share personal health information, 911 guidance), pre-written in seven languages
(en/es/zh/vi/tl/ar/ko) with double-append guards.

**App shell (`app.py`).** A thin Gradio `ChatInterface`: builds the inference client, prepends
the configured system message, delegates to `CareLocatorAgent.handle_request`, and hosts the
provider-card CSS plus static safety/data-source notes. `ssr_mode=False` works around a Gradio
SSR reachability check that crash-loops inside Spaces.

## Trust and safety posture

- Navigation, not medicine: no diagnosis, no prescriptions; emergency queries route to 911/ER
  guidance before any search.
- Nothing is over-claimed: insurance/network participation, new-patient status, and
  appointments are labeled unverified unless a source says otherwise.
- Numbers are only trusted from the user, never from the model (the numeric trust boundary).
- Search requests are PHI-lean by design: only specialty/location/insurance/language/keyword
  fields ever reach the retrieval layer or its cache.
- Model self-reports (like `needs_clarification`) are validated behaviorally before being
  trusted — and rejected when they prove unreliable.

## Evaluation (`eval/`)

A fairness-evaluation harness runs the real request path over a scenario × language matrix
(en/zh/es/ar/ko), captures per-layer artifacts (intent → search request → results → rendered
reply), scores them against gold labels, and adds an independent LLM judge
(`Qwen2.5-72B-Instruct`, cross-lineage from the system model) validated against human labels
with Cohen's κ. Because layers B/C are language-invariant, cross-language divergence is
attributable to a specific layer. Results, findings (F1–F8), and the case study live in
`eval/RUNS.md`, `eval/FINDINGS.md`, and `eval/CASE_STUDY.md`.

## Known limitations and planned work

- **Specialty coverage** (F5): several advertised specialties still lack NPI-taxonomy synonym
  mappings and return no providers; generalizing the umbrella map to every specialty family is
  the highest-impact planned fix.
- **Ambiguity clarification** (F3): umbrella queries like "mental health" are searched (and
  return nothing) instead of triggering a clarifying question; the curated ambiguity list that
  already handles "child allergy" is planned to grow, matching on the language-invariant
  parsed specialties so all languages benefit equally.
- **Localization consolidation**: template copy currently lives in three hand-written language
  tables plus a seven-language footer; consolidating to a single English source with generated,
  clearly-labeled translations (shipped as locale files) is planned.
- **Input privacy guard**: a deterministic pre-LLM redaction pass for identifiers users
  shouldn't share (SSNs, member IDs, phone numbers) is planned, including a multilingual
  evaluation of the guard itself.
- **Intent schema cleanup**: the model-supplied `needs_clarification` field will be removed
  from the schema (it is already ignored); the app-computed signal will be renamed to make its
  provenance explicit.
- **Native-numeral locations**: Arabic-Indic ZIP digits (e.g. `٠٢١٣٩`) are not yet normalized
  (`eval/CASE_STUDY.md` §4) — open.

## Repository layout

| Path | Role |
|---|---|
| `app.py` | Gradio shell (Spaces entry point) |
| `care/` | A1 + routing + A2 package: language, safety, intent, guidance, rendering, agent |
| `provider_search/` | B: retrieval package (service, sources, ranking, families, cache) |
| `eval/` | fairness-evaluation harness (dataset, tracing, scoring, judge, κ) |
| `config/settings.yaml` | prompts, UI copy, fallback resources, search settings |
| `retriever.py`, `config_loader.py` | legacy retrieval helpers / config access |
| `tests/` | unit + gated live tests (`RUN_EVAL=1`) |

The A1/routing/A2 stages are now split into the `care/` package with single-purpose modules (mirroring `provider_search/`).
