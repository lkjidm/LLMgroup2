# CLIMES-Monitor

CLIMES-Monitor is a student project for weather information and climate-related
preparedness in Uppsala, Sweden. It combines weather-service data, retrieved
preparedness material, historical temperature calculations, and an optional
spoken response in a Gradio interface.

## Run the current application

Use the latest `main` branch. The application entry point is **`integrated.py`**.
It launches the implementation in **`climes_current/integrated_final.py`**.
Historical temperature calculations are implemented in **`heatwave_stats.py`**.

### Google Colab

1. Open [`test.ipynb`](test.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the first code cell. It clones the repository into a separate directory,
   checks GPU availability, and installs the required packages if needed.
4. Run the second code cell. Supply the WeatherMetro key through the Colab Secret
   `WEATHERMETRO_API_KEY` or the hidden input prompt.
5. The notebook runs the checks and acceptance batch, exports an evidence ZIP,
   and starts the app. Open the printed `gradio.live` link.

Keep the Colab runtime connected while using or sharing the app. The public
link depends on the running service and is not permanent hosting. If automatic
ZIP download stalls, use the Colab Files pane to download the generated
`climes_final_acceptance_*.zip` from `/content`.

### Local environment

The live application expects an NVIDIA CUDA GPU, Python 3.10 or later, and
network access for model downloads, WeatherMetro and optional speech synthesis.
The recorded cloud run used a Colab T4; other environments have not been
independently validated.

```bash
git clone https://github.com/lkjidm/LLMgroup2.git
cd LLMgroup2
python -m pip install -r requirements.txt
```

Set `WEATHERMETRO_API_KEY` in your environment, then run:

```bash
python integrated.py
```

Do not put API keys in source files or commit them to GitHub. Weather requests
require a valid key. The notebook requests it before the acceptance batch.

## What the app does

- **Current weather and forecasts:** Recognised Uppsala locations are matched
  to coordinates, prioritising neighbourhoods such as Flogsta and Gränby over
  the city name. Answers display validated weather-tool values and explicit
  forecast dates. Missing or invalid data produces an unavailable response.
- **Resident preparedness:** English questions about heat, cold or flooding
  retrieve the relevant FEMA leaflet. The app presents complete sentences from
  an approved set only when those sentences occur in the retrieved passage.
  The source and US context are disclosed.
- **Healthcare preparedness:** Healthcare or hospital questions retrieve
  substantive recommendations from the CLIMES healthcare preparedness brief.
  These are distinguished from personal resident guidance.
- **Historical temperature context:** The bundled SMHI series is queried using
  the requested Celsius threshold and minimum consecutive-day duration. Each
  maximal matching run counts once; missing dates or temperatures break runs.
  A short paragraph explains the result. Historical occurrence is not a future
  event probability, health-risk classification or official warning.
- **Speech:** Text is displayed first, followed by optional audio. Speech
  generation failure preserves the text answer.

The language model is **Qwen/Qwen3.5-2B**. This application uses the base model;
it does not train or load a LoRA adapter. The model selects retrieved passage
IDs. Routing, weather values, historical statistics and final response rendering
are controlled in Python; unrestricted model prose is not displayed as advice.
The embedding model is `all-MiniLM-L6-v2`.

## Example questions

```text
What is the current weather in Flogsta, Uppsala?
What is the weather forecast for Uppsala tomorrow?
What precautions should residents take during extreme cold in Uppsala?
Suppose it is 32°C in Uppsala tomorrow. What precautions should I take?
Historically how often has Uppsala had 4 consecutive days at 31°C?
What healthcare preparedness is recommended for hospitals during extreme heat?
```

Each question should include the relevant conditions explicitly. Context-dependent
follow-ups such as “What about older people?” are not currently supported.

## Project files and data

| Path | Purpose |
|---|---|
| `integrated.py` | Canonical app launcher |
| `climes_current/integrated_final.py` | Routing, validated weather access, retrieval, response rendering, speech and UI |
| `heatwave_stats.py` | Exact-threshold historical run counting |
| `test.ipynb` | Colab setup, acceptance batch and app launch |
| `climes_current/START.py` | Live acceptance runner and evidence export |
| `knowledge/` | Chroma index, source PDFs and Uppsala place dictionary |
| `uppsala_tmax_1985_2026.csv` | Bundled SMHI daily maximum series, 1 June 1985–30 June 2026 |
| `climes_current/validation/` | Offline checks and the recorded r4 cloud acceptance results |
| `climes_current/version_manifest.json` | Candidate provenance, component hashes and verification scope |
| `old/`, `experiment/`, `README_QWEN.md` | Earlier development material; not the final app entry point |

The historical CSV contains 15,004 observations, one missing calendar date and
quality flags G and Y. The current calculation retains both flags without
interpreting or filtering them; this limits claims about data quality.
The category tables in `data/` are earlier analysis outputs; the live historical
function calculates from the daily CSV rather than using those fixed categories.
Swedish manuals remain in the knowledge store but are not automatically
translated or substituted for English resident advice.

## Evaluation

The r4 Colab acceptance run on **19 October 2026** recorded **17 PASS, 0 FAIL,
0 UNAVAILABLE and 0 ERROR**. It covered 15 predefined prompts and two simulated
weather-provider failures. Speech-file generation and Gradio message
preprocessing also passed their checks. The merged application's main script,
launcher, historical tool, runner and CSV were compared with the recorded r4
components. This is not a new cloud run of the merged Git commit.

The local suite includes **38 regression test methods** (including the original
16 risk-rule boundary cases) and **19 integration/output checks**. A separate
17-case synthetic harness tests evidence export and scoring with service doubles.

```bash
python climes_current/check_final.py climes_current/integrated_final.py
python climes_current/check_integration.py
python climes_current/check_acceptance_harness.py
```

These commands do not run the real language model, WeatherMetro, speech service
or browser. Synthetic harness passes must not be reported as additional real
user tests or as an overall accuracy score. Cloud results are retained in
[`climes_current/validation/live_r4_20261019/`](climes_current/validation/live_r4_20261019/).
Older logs elsewhere in `climes_current/` concern earlier candidates.

## Limitations

This prototype does not check an official weather-warning feed, provide a
clinical assessment, or establish neighbourhood-level weather accuracy from
independent measurements. General US preparedness guidance may not fully fit
Swedish circumstances. Healthcare recommendations address services rather than
individual care. Current tests cover selected cases; they do not demonstrate
medical safety, general answer accuracy, accessibility or broad usability.
User feedback should be reported separately after it has been collected.
re evaluation. Do not treat the app as an emergency service.
