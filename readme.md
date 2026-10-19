# CLIMES-Monitor: Uppsala Climate & Public Health Agent

> An AI agent designed to provide weather information, historical temperature context, and source-based preparedness guidance for Uppsala, Sweden.

---

## 🌟 Overview

**CLIMES-Monitor** is a specialized AI assistant built to bridge the gap between meteorological data and practical preparedness information for local residents and international students in Uppsala.

The project adopts a **"Guardrail-First" hybrid architecture**: weather-data validation, project-defined risk thresholds, and historical temperature calculations are handled by a deterministic Python engine. The base **Qwen/Qwen3.5-2B** model selects relevant retrieved passages, while Python renders source-supported responses. The final application does not train or load a LoRA adapter.

The system supports current weather and forecasts, neighbourhood matching, preparedness guidance for heat, cold and flooding, and historical comparisons using the user's specified temperature and duration. English resident guidance uses FEMA leaflets; healthcare-related questions use the CLIMES healthcare preparedness brief. Swedish manuals remain in the knowledge base but are not automatically translated into English. Optional text-to-speech provides an additional way to access responses.

---

## 🏗️ Project Structure

The project is organized as follows:

```text
LLMgroup2/
├── integrated.py             # 🚀 Main application launcher
├── climes_current/           # 🛠️ Final application, checks and validation records
│   ├── integrated_final.py   # Application logic, retrieval, responses and UI
│   ├── START.py              # Acceptance batch, evidence export and UI launch
│   └── validation/           # Offline checks and recorded cloud validation
├── heatwave_stats.py         # 🌡️ Historical temperature calculations
├── test.ipynb                # 📓 Colab setup, acceptance checks and app launch
├── my_weather_lora_adapter/  # Earlier model/tokenizer files; not loaded by the final app
├── knowledge/               # 📚 RAG index, source manuals and Uppsala place data
├── data/                    # 📊 Processed climate and meteorological datasets
├── evaluation/              # 📈 Earlier development evaluation records
├── experiment/              # 🧪 Experimental notebooks and scripts
├── old/                     # 📁 Earlier application components
├── uppsala_tmax_1985_2026.csv # Historical Uppsala daily maximum temperature records
└── requirements.txt         # 📦 Python dependencies
```

---

## 🚀 Running the Application

Use the latest `main` branch. For Google Colab, open **`test.ipynb`**, select a **T4 GPU**, and run the two code cells in order. Provide the WeatherMetro API key through the Colab Secret `WEATHERMETRO_API_KEY` or the hidden input prompt. The notebook runs the acceptance checks, exports an evidence ZIP, and launches the Gradio interface. Keep the runtime connected while using or sharing the temporary interface link.

For a local CUDA-enabled environment, install the dependencies, set the `WEATHERMETRO_API_KEY` environment variable, and run:

```bash
python -m pip install -r requirements.txt
python integrated.py
```

`integrated.py` launches `climes_current/integrated_final.py`. 
---

## 🧪 Evaluation and Limitations

The local suite contains **38 regression test methods**, including the original 16 risk-rule boundary cases, and **19 integration/output checks**. The r4 Colab batch on **19 October 2026** recorded **17 passing cases**: 15 predefined prompts and two simulated weather-provider failures. These selected checks do not establish overall accuracy, medical safety or usability. Detailed results are discussed in the project report.

The application does not check an official weather-warning feed. FEMA guidance may not fully reflect Swedish circumstances, and healthcare guidance is intended for services rather than individual care. Historical comparisons count matching runs in available station observations; they do not predict future events or determine health risk. Context-dependent follow-up questions are not currently supported.
