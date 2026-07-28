# Local input schema

This document defines column names required by the software. It contains no
study observations, source locators, publication identifiers, or summary
statistics.

## Hierarchy fields

| Field | Purpose |
|---|---|
| `record_id` | Unique observation identifier |
| `base_mix_id` | Base-mixture grouping identifier |
| `publication_family_id` | Publication-family grouping identifier |
| `experimental_campaign_id` | Experimental-campaign grouping identifier |
| `research_programme_id` | Programme or laboratory grouping identifier |

The model-ready CSV must contain all five fields. The hierarchy CSV supplied to
the split generator must contain `record_id`, `experimental_campaign_id`, and
`research_programme_id`; it may also repeat `publication_family_id` for local
audit purposes.

## Predictor fields

| Field | Unit or type |
|---|---|
| `curing_age_days` | days |
| `water_to_binder_ratio` | mass ratio |
| `portland_or_clinker_pct_binder` | % binder |
| `calcined_clay_pct_binder` | % binder |
| `limestone_pct_binder` | % binder |
| `portland_basis_harmonized` | categorical |
| `gypsum_pct_binder` | % binder |
| `calcination_temperature_c` | °C |
| `calcination_duration_h` | h |
| `sand_to_binder_ratio` | mass ratio |
| `gypsum_reporting_status` | categorical |
| `initial_cure_temperature_c` | °C |
| `post_demould_temperature_c` | °C |
| `compression_face_area_mm2_exact` | mm² |
| `initial_cure_class` | categorical |
| `post_demould_cure_class` | categorical |
| `specimen_shape_class` | categorical |
| `test_standard_family` | categorical |
| `aggregate_class` | categorical |

`compressive_strength_mpa` is the response. Predictor values may be missing;
imputation is fitted separately inside each training fold. The response and all
hierarchy identifiers must be complete, and `record_id` must be unique.

## Privacy boundary

The schema is public, but populated inputs and generated split manifests are
not part of this repository. Keep them in a private or otherwise lawfully
controlled location.
