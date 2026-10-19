> **Archived development documentation (14 October 2026).** This describes the earlier HeatLens/Qwen3.5-4B experiment, not the final CLIMES application or its evaluation. For the current Qwen3.5-2B application, use [readme.md](readme.md) and run `integrated.py`. Historical paths and pending work below refer to that earlier experiment.

# HeatLens: Qwen3.5 Integration

This component develops the model and application portion of the shared HeatLens project. It uses Qwen3.5-4B on a Colab GPU to extract a temperature scenario, validates and routes the output in Python, executes a historical weather comparison, and generates an explanation from the tool result.

**Status:** Development snapshot for branch `br-2`. Model loading, selected extraction checks, routing checks and one complete interaction have user-reported Colab results. The UI is prepared but its execution has not yet been confirmed. Fresh-session reproducibility and the final evaluation remain pending.

## Files

| File | Purpose | Verification status |
| --- | --- | --- |
| `04_Qwen35_HeatLens_Demo.ipynb` | The contributor's actual Colab notebook containing the working model, routing and integration cells | Must be exported from the current Colab session and added separately; not included in this source package |
| `heatlens_weather_tool.py` | CSV loader and exact-threshold historical comparison using the agreed tool entry point | Locally checked with reference cases; a reference result was reproduced in the user's Colab session |
| `heatlens_colab_ui.py` | Optional interface cell using the existing `ask_heatlens` function | Prepared source; rendering and callbacks require Colab verification |
| `requirements-qwen.txt` | Observed model-library versions and proposed widget dependency | Complete fresh-session setup has not yet been verified |
| `evaluation/qwen35_progress.md` | Provenance, observed progress and remaining checks | Summary transcribed from supplied execution outputs; not a new evaluation run |

The historical CSV is shared project data. The component expects `uppsala_tmax_1985_2026.csv`. Use the repository's existing file when it is the agreed snapshot; if absent, add the dataset or acquisition instructions supplied by the data owner.

## Application Flow

1. Qwen extracts the scenario as application-defined JSON.
2. Python checks the required arguments and decides the effective application route.
3. Complete supported scenarios invoke `heatwave_risk_tool(days, temperature)`.
4. The tool counts qualifying observed continuous runs for the exact requested temperature threshold and minimum duration.
5. Qwen explains the returned calculation and limitations.
6. Missing information returns clarification without invoking the weather tool.

The extraction format is an application-defined protocol. It is not presented as a verified native Qwen function-calling implementation.

## Current Colab Workflow

Use a GPU runtime. The reported setup used a Tesla T4, PyTorch 2.11.0+cu130, Transformers 5.19.0 and Accelerate 1.15.0. Model weights were loaded as `torch.float16`. Some parameters were offloaded to CPU and reference kernels were used; inference speed remains a limitation.

The development notebook uses interactive uploads for the Python module and CSV. Its saved outputs establish particular execution checks, not a guarantee that another session can reproduce the notebook without adjustments.

Before finalizing the notebook, standardize file references to:

```python
from heatlens_weather_tool import load_weather_data, heatwave_risk_tool
metadata = load_weather_data("uppsala_tmax_1985_2026.csv")
```

Download-generated prefixes such as `1heatlens_weather_tool.py` and duplicate suffixes such as `uppsala_tmax_1985_2026 (1).csv` are temporary session filenames. They should not become the repository's canonical filenames. The standard import above requires the module and CSV to be available in the notebook's working directory and the module directory to be importable.

The optional UI source is designed to run as a notebook cell in the same session after `ask_heatlens` is defined. It does not independently load the model or data. Its explanation prompt uses “at least” for minimum duration and identifies quality review as a project data-owner task. The updated explanation and widget behaviour still require verification.

## Recorded Results and Limits

The original five-question model protocol checks passed 3/5 before and after the prompt revision. In the revised run, incomplete inputs generated clarification messages but incorrectly selected `call_tool`. Python routing subsequently handled all five retained outputs as intended. These are separate measurements: model protocol compliance remains 3/5, while recorded-output application routing passed 5/5.

Two additional missing-information questions produced correct raw `clarify` outputs and correct application routing. One complete scenario used 4 days and 31°C, invoked the Python tool and returned two qualifying observed runs in the supplied dataset. The generated explanation preserved the principal numerical results but used inaccurate “official review” wording, which prompted an explanation-prompt revision.

These checks are development evidence, not a held-out final evaluation. Their provenance and remaining work are documented in `evaluation/qwen35_progress.md`.

## Data Interpretation

The supplied CSV covers 1985-06-01 to 2026-06-30 and contains 15,004 rows, one missing calendar date and quality flags G and Y. The integration module retains these flags without interpreting or filtering them; the data owner must approve the final policy.

Missing observations break confirmed runs and can split an underlying event. Counts refer to maximal qualifying runs in available observations. The component does not calculate a calibrated health-risk probability, issue an official warning or forecast weather.

The exact 31°C/four-day comparison differs from the earlier fixed 30°C/three-day category. Results from these two definitions must not be compared as if they counted identical events.

## Remaining Submission Work

- Export and organize the actual working Colab notebook.
- Standardize filenames and verify execution from a fresh GPU session.
- Confirm that incomplete questions do not invoke the weather tool through the integrated agent.
- Verify UI rendering, complete-question processing, clarification and recovery after errors.
- Obtain data-owner review of quality flags, missing observations and comparison rules.
- Run a documented final evaluation using new questions; preserve failures and raw outputs.
- Complete the Architecture and experimental-setup report content and an individual contribution statement reflecting actual work.
- Merge reviewed project components into the agreed final submission version.

## Generative-AI Assistance

Generative AI assisted with code preparation, prompt revision, debugging and documentation. The contributor executed the Colab checks and supplied the observed outputs. Verification status in this document distinguishes those observations from prepared or pending work. The final project report should include the team's complete and accurate AI-use declaration.
