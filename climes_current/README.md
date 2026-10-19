# CLIMES Application

The repository entry point is `integrated.py`, which launches
`climes_current/integrated_final.py`. Historical temperature calculations are
implemented in `heatwave_stats.py` at the repository root.

See the [project README](../readme.md) for the application overview and setup.
`START.py` runs the acceptance batch, exports its evidence ZIP, and opens the UI.

The local checks comprise 38 regression test methods and 19 integration/output
checks. The Colab acceptance run on 19 October 2026 recorded 17 passing cases.
Its answers and acceptance report are retained in `validation/live_r4_20261019/`.
The synthetic harness is a separate test using service doubles.
Earlier result files in this folder document previous development versions.
