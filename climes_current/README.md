# Current CLIMES application

Run `python integrated.py` from the repository root. The launcher starts
`climes_current/integrated_final.py`; `heatwave_stats.py` at the root supplies
historical calculations. See the root [README](../readme.md) for Colab setup,
example questions, source coverage and limitations.

The r4 cloud batch recorded 17 passing cases on 19 October 2026. Its original
answers and acceptance report are in `validation/live_r4_20261019/`. The main
application script and relevant components were compared with the merged main
branch; this does not constitute a fresh cloud execution of that Git commit.
The local suite has 38 regression methods and 19 integration methods. The
synthetic harness uses service doubles and is separate from live evidence.

`START.main(project=...)` runs the live batch, exports its evidence ZIP and
launches the UI. Older loose result files in this folder describe earlier
candidates. They are retained as development history, not current acceptance.
