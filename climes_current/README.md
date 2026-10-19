# Current CLIMES application

Run `python integrated.py` from the repository root. The launcher starts
`climes_current/integrated_final.py`; `heatwave_stats.py` supplies historical
calculations. See the root [README](../readme.md) for Colab setup and sources.

## Conversation repair (r5)

The application recognises heatwave plurals and natural cooling questions,
clarifies ambiguous temperature questions, and distinguishes month-specific
historical comparisons from future forecasts. A selected month includes only
runs whose counted consecutive days all fall within that calendar month.
The terms "above" and "at least" use strict and inclusive thresholds respectively.
Unsupported year, season or multi-month historical filters are disclosed.

Healthcare replies retain the context of dependent source sentences and filter
PDF headings. General CLIMES recommendations are labelled as system preparedness,
not a hazard-specific clinical protocol. Duplicate generator advice retains the
more complete source sentence. Authentication failures identify rejected weather
credentials or permissions without displaying the key.

## Validation

Local checks passed: 38 regression methods, 19 integration methods and 25
conversation methods. The 17-case synthetic harness also passed using test doubles.
Records are in `validation/conversation_r5_20261020/`. No live model, API, speech
or browser run of r5 is claimed by these local checks.

The earlier r4 cloud batch recorded 17 passing cases on 19 October 2026; its
original evidence remains in `validation/live_r4_20261019/`. That historical
result does not certify the changed r5 application. A separate r4 run returned
HTTP 401 for live weather requests, indicating rejected credentials or access.

`START.main(project=...)` runs all local checks before the standard 17-case live
batch, exports an evidence ZIP, and launches the UI. Live UI interactions are
logged to the run directory's `events.jsonl` after the initial ZIP is created.
Directly running `integrated.py` starts the app without an evidence ZIP; weather
failure reason codes are also printed in the runtime output.
