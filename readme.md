# CLIMES-Monitor: Uppsala Climate & Public Health Agent

> A robust, locally-executable AI agent designed to monitor climate conditions, assess extreme weather risks, and deliver public-health guidance tailored specifically for Uppsala, Sweden.

---

## 🌟 Overview

**CLIMES-Monitor** is a specialized, privacy-focused AI assistant built to bridge the gap between complex meteorological data and actionable public health safety for local residents and international students in Uppsala. 

Instead of relying purely on unconstrained large language models (which are prone to hallucinations in critical scenarios), this project adopts a **"Guardrail-First" hybrid architecture**: objective meteorological thresholds and safety logic are enforced via a deterministic Python engine, while fine-tuned LLMs handle empathetic, context-aware natural language synthesis.

---

## 🏗️ Project Structure

The project is structured into a clean, easy-to-maintain layout optimized for local deployment and rapid demonstration:

```text
LLMgroup2/
├── integrated.py            # 🚀 Main entry point (monolithic execution script)
├── my_weather_lora_adapter/ # 🎯 Fine-tuned LoRA weights for domain adaptation
├── knowledge/               # 📚 Local RAG safety manuals and emergency guides
├── data/                    # 📊 Processed climate and meteorological datasets
├── evaluation/              # 📈 System evaluation metrics and logs
├── experiment/              # 🧪 Experimental notebooks and scripts
├── uppsala_tmax_1985_2026.csv # 🌡️ Historical Uppsala maximum temperature records
└── equirements.txt         # 📦 Python dependencies

## Report progress - 17 October 2026

The shared Prism project is the working version of the report.

- Section 3 (Architecture) is complete.
- Section 4 (Results) is complete except for user feedback.
- User feedback will be added after testing.
- Figure placement will be adjusted after the full report is complete.

