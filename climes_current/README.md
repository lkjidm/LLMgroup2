# Combined implementation

The canonical repository command is `python integrated.py`; it launches the
implementation in this folder. See the root `readme.md` for data sources,
historical counting semantics, setup, and live acceptance instructions.

Current offline results are in `validation/`; `version_manifest.json` identifies
the combined candidate and its component hashes. Older result files in this
folder are retained as historical evidence and apply to earlier code revisions.

Run `START.main(project=...)` for the 17-case live batch, evidence export and UI.
The live batch must be rerun after integration; old PR #3 passes do not certify
the newly added knowledge index, healthcare route or historical functionality.
