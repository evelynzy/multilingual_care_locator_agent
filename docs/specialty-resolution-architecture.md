# Shared Specialty-Resolution Architecture

The specialty-resolution system uses one canonical specialty-family model across
request normalization, query-time rescue, provider normalization, and
ranking/gating. The shared catalog lives in
`provider_search/specialty_families.py` and defines stable family identifiers
such as `cardiology`, `obstetrics-gynecology`, and `primary-care`.

## Shared family model

The shared catalog drives three related normalization paths:

- `derive_request_specialty_family_ids(...)`
  Converts explicit request specialties and family ids into canonical family ids
  for search requests.
- `derive_provider_specialty_family_ids(...)`
  Converts provider-side evidence such as `provider_type`, taxonomy
  descriptions, taxonomy codes, and source-specific specialty strings into the
  same family ids.
- `derive_query_specialty_family_ids(...)`
  Converts only a narrower, query-safe subset of user wording into family ids
  when request-time rescue is needed.

This lets the system compare user intent and provider evidence in one shared
vocabulary while still keeping request-time inference conservative.

## Intentional trust boundary

The provider catalog is intentionally broader than the query-safe rescue
catalog.

- Provider normalization must be broad because live directories emit many
  specialty variants, wrapped taxonomy labels, and descendant taxonomy codes.
  Missing those forms would drop valid candidates before ranking.
- Query-time rescue must be narrower because broad, underspecified user wording
  should trigger clarification instead of silent specialty assignment.

Examples of phrases that are valid provider evidence but not query-safe rescue
aliases:

- `mental health`
- `behavioral health`
- `therapy`
- `counseling`
- `sports medicine`
- `imaging`
- `allergy`
- `gi`
- `urgent care`

These phrases can still participate in downstream provider normalization, but
request-time rescue should abstain so the app can ask a follow-up question when
intent is ambiguous.

## Invariants future changes must preserve

- Clear specialist wording should survive the full pipeline as a canonical
  specialty family.
- Broad provider evidence should continue to normalize into the shared family
  model.
- Query-safe rescue should stay narrower than provider normalization.
- Adding a new specialty family should update both catalogs deliberately rather
  than copying every provider alias into the query-safe surface.
