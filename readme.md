# CLIMES-Monitor

A student prototype for Uppsala weather observations, forecasts, source-based
preparedness excerpts, and descriptive historical temperature context.

## Current entry point

Run `python integrated.py` from this repository. It launches
`climes_current/integrated_final.py`; both commands use the same implementation.
The former application at the repository root has been replaced by this launcher.
`test.ipynb` runs the combined acceptance batch and then opens the same app.

The base language model remains **Qwen/Qwen3.5-2B**. This revision does not train
model weights or load a LoRA adapter. The model selects passage IDs; generated
prose is discarded. Weather values and historical statistics are rendered from
validated tool outputs.

## Integrated components

- Location matching prioritises recognised neighbourhoods such as Flogsta and
  Gränby over a city name. WeatherMetro failures/missing fields produce explicit
  unavailable responses; no fallback observations are fabricated.
- English resident preparedness queries return complete, vetted action sentences
  that are actually present in retrieved English FEMA leaflet passages. Cover
  pages, link lists and municipal-policy titles do not qualify as resident
  precautions. At least two distinct precautions are required for acceptance.
  The scope of these US leaflets is disclosed; they are not Swedish official
  guidance. Unverified model prose is never used as safety advice.
- The CLIMES healthcare preparedness brief is used for healthcare/hospital
  questions, with a substantive recommendation check. The full Swedish manuals
  remain in the knowledge store but are not substituted for English resident
  advice. Automatic translation is not implemented.
- `heatwave_stats.py` uses the bundled SMHI daily maximum CSV. A request for
  4 days at 31°C counts runs at **at least 31°C for at least 4 consecutive days**.
  It no longer silently substitutes the 30°C/3-day category. Missing dates or
  temperatures break runs. Each maximal run is counted once.
- Historical frequency is descriptive, not an official warning, future event
  probability, or neighbourhood-level forecast. A duration must be supplied;
  "32°C tomorrow" does not invent a three-day duration.
- Text is displayed before optional speech. Audio failure preserves the answer.

## Run and check

A CUDA GPU and the Python packages in `requirements.txt` are required for the
live model. Set `WEATHERMETRO_API_KEY` in the environment (or use the notebook's
Colab Secret/password prompt). Never commit a key.

```bash
python climes_current/check_final.py climes_current/integrated_final.py
python climes_current/check_integration.py
python climes_current/check_acceptance_harness.py
python integrated.py
```

The first three commands do not load the real model or contact WeatherMetro.
The harness uses synthetic retrieval/weather/audio/UI services and is explicitly
not a live acceptance result. For the live batch, use `test.ipynb` or:

```python
from climes_current.START import main
report, evidence_zip = main(project="/absolute/path/to/LLMgroup2")
```

The batch covers the original 12 prompts, healthcare guidance, an exact
historical query, a multi-day hypothetical, and two simulated API failures
(17 cases total). It exports answers, source passages, tool results, component
hashes, timings, and checks to `climes_final_acceptance_*.zip`, then launches the
UI. Inspect `UNAVAILABLE`/`FAIL`/`ERROR` results and the actual selected sources.
Structural passes do not establish medical accuracy or weather accuracy.

## Development provenance and limits

This candidate combines main `bc8b70bba4e91df284dcf7ce42a870efdc4967d8`
(including PR #4's knowledge index and historical dataset) with PR #3
`41c867e3417044ceaa6a3e8e3fa7f6543d9b3f7d` and the integration changes above.
The r4 revision renders historical results as one short paragraph using the exact tool values, retaining full statistical details in the event log. The r3 revision added concise conversational presentation, deduplicated bullet precautions, readable source names, and complete healthcare recommendation sentences. The preceding r2 revision addressed the observed cover/reference-list failure in the first combined live test. See `climes_current/version_manifest.json` and `climes_current/validation/`.
Earlier logs under `climes_current/` describe earlier candidates and do not
certify this integration. The manifest states whether a live rerun is recorded.

The app does not check an official warnings feed. Source relevance, Swedish
language accessibility, third-party service availability, and real-user usability
still require evaluation. Do not treat the app as an emergency service.
