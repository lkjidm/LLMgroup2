import json
import os
import re
from threading import Thread
import time
import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
from heatwave_stats import historical_heatwave_stats
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    logging,
)
import collections
import edge_tts
import asyncio

async def text_to_speech(text: str, output_file: str = "output.mp3", voice: str = "en-US-AvaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def generate_audio(text: str):
    # only convert the first 250 characters to speech to avoid long audio generation
    short_text = text[:250] if len(text) > 250 else text
    asyncio.run(text_to_speech(short_text, "response_audio.mp3"))
    return "response_audio.mp3"

logging.set_verbosity_warning()

# =====================================================================
# 1. data loading and initialization
# =====================================================================
try:
  df_hist = pd.read_csv('./uppsala_tmax_1985_2026.csv', parse_dates=['date'])
  print('Historical weather data loaded successfully! Shape:', df_hist.shape)
except Exception as e:
  print('Warning: CSV file not found. Creating a dummy dataset for demonstration.')
  dates = pd.date_range('1985-06-01', '2026-06-30')
  df_hist = pd.DataFrame({'date': dates, 'tmax': np.random.uniform(10, 32, len(dates))})

WEATHERMETRO_API_KEY = 'uXQyXC4iqxCVdzGSCbQ0uGmir0CzRFtU'
DEFAULT_LAT = 59.8586
DEFAULT_LON = 17.6389
DEFAULT_PLACE_NAME = 'Uppsala Centrum'
_weather_cache = {}

UPPSALA_PLACES_JSON = './knowledge/uppsala_all_places.json'
def load_uppsala_places():
  if os.path.exists(UPPSALA_PLACES_JSON):
    with open(UPPSALA_PLACES_JSON, 'r', encoding='utf-8') as f:
      return json.load(f)
  print('⚠️ warning: Uppsala places JSON file not found. Using city center as default.')
  return []

UPPSALA_PLACES = load_uppsala_places()

# =====================================================================
# 2. location extraction and API utility functions
# =====================================================================
def extract_location_coords(user_text: str):
  text_lower = user_text.lower()
  sorted_places = sorted(UPPSALA_PLACES, key=lambda x: len(x['name']), reverse=True)
  for place in sorted_places:
    name = place['name']
    if len(name) > 1 and name.lower() in text_lower:
      return {'found': True, 'name': name, 'lat': place['lat'], 'lon': place['lon'], 'type': place.get('type', 'landmark')}
  return {'found': False, 'name': DEFAULT_PLACE_NAME, 'lat': DEFAULT_LAT, 'lon': DEFAULT_LON, 'type': 'city_center'}

def detect_weather_intent(user_text: str):
    text = user_text.lower()
    forecast_keywords = ["tomorrow", "forecast", "next week", "future", "will it", "predict"]
    is_forecast = any(kw in text for kw in forecast_keywords)
    days_match = re.search(r'(\d+)\s*(?:days|day)', text)
    days = int(days_match.group(1)) if days_match else (3 if is_forecast else 1)
    return "forecast" if is_forecast else "current", days

def fetch_weathermetro_data(lat: float, lon: float, place_name: str = "Uppsala", user_text: str = "") -> dict:
    intent, forecast_days = detect_weather_intent(user_text)
    now = time.time()
    cache_key = f"{intent}_{round(lat, 3)}_{round(lon, 3)}"

    if cache_key in _weather_cache:
        cached_data, timestamp = _weather_cache[cache_key]
        if now - timestamp < 600:
            return cached_data

    headers = {"User-Agent": "UppsalaClimateBot/1.0"}
    url = "https://weathermetro.com/api/v2/forecast" if intent == "forecast" else "https://weathermetro.com/api/v2/current"
    params = {"api_key": WEATHERMETRO_API_KEY, "lat": lat, "lng": lon, "units": "metric"}
    if intent == "forecast":
        params["days"] = forecast_days

    try:
        response = requests.get(url, params=params, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            resp_data = data.get("response", {}).get("data", {})
            if intent == "forecast":
                hourly = resp_data.get("hourly", {})
                times, temps, precips = hourly.get("time", []), hourly.get("temperature_2m", []), hourly.get("precipitation", [])
                daily_grouped = collections.defaultdict(lambda: {"temps": [], "precips": []})
                for t, temp, precip in zip(times, temps, precips):
                    daily_grouped[t.split("T")[0]]["temps"].append(temp)
                    daily_grouped[t.split("T")[0]]["precips"].append(precip)
                parsed_daily = [
                    {"date": k, "temp_max": round(max(v["temps"]), 1), "temp_min": round(min(v["temps"]), 1), "precip_total": round(sum(v["precips"]), 1)}
                    for k, v in daily_grouped.items() if v["temps"]
                ]
                result = {"status": "success", "intent": "forecast", "place_name": place_name, "lat": lat, "lng": lon, "forecast_days": forecast_days, "daily": parsed_daily}
            else:
                current = data.get("response", {}).get("current", {})
                result = {"status": "success", "intent": "current", "place_name": place_name, "lat": lat, "lng": lon, "temp": current.get("temperature_2m", 7.4), "humidity": current.get("relative_humidity_2m", 80), "wind_speed": current.get("wind_speed_10m", 10), "precipitation": current.get("precipitation", 0), "condition": "Live Weather"}
            _weather_cache[cache_key] = (result, now)
            return result
    except Exception as e:
        print(f"WeatherMetro API Exception: {e}")

    return {"status": "fallback", "intent": intent, "place_name": place_name, "lat": lat, "lng": lon, "temp": 7.4, "condition": "Fallback Weather"}

def compute_extreme_weather_risk(
    temp: float = None, wind_speed: float = None, precip: float = None
):
  risks = []

  # 1. heatwave identification
  if temp is not None:
    if temp >= 30:
      risks.append({
          "type": "HEATWAVE",
          "level": "HIGH",
          "desc": (
              f"Extreme Heat Warning: Max temp reached {temp}°C (≥30°C"
              " threshold)."
          ),
      })
    elif temp >= 27:
      risks.append({
          "type": "HEATWAVE",
          "level": "MODERATE",
          "desc": (
              f"Health-relevant Heat: Temp reached {temp}°C (≥27°C threshold)."
          ),
      })

  # 2. Extreme Cold
  if temp is not None:
    if temp <= -25:
      risks.append({
          "type": "EXTREME_COLD",
          "level": "HIGH",
          "desc": (
              f"Severe Cold Hazard: Temp dropped to {temp}°C (≤-25°C"
              " threshold)."
          ),
      })
    elif temp <= -15:
      risks.append({
          "type": "EXTREME_COLD",
          "level": "MODERATE",
          "desc": (
              f"Extreme Cold Caution: Temp dropped to {temp}°C (≤-15°C"
              " threshold)."
          ),
      })

  # 3. Strong Wind Identification - Units in km/h as per WeatherMetro
  if wind_speed is not None:
    if wind_speed >= 70:
      risks.append({
          "type": "STRONG_WIND",
          "level": "HIGH",
          "desc": (
              f"Severe Wind Warning: Wind speed reached {wind_speed} km/h"
              " (≥70 km/h)."
          ),
      })
    elif wind_speed >= 50:
      risks.append({
          "type": "STRONG_WIND",
          "level": "MODERATE",
          "desc": (
              f"Strong Wind Caution: Wind speed reached {wind_speed} km/h"
              " (≥50 km/h)."
          ),
      })

  # 4. Heavy Rainfall
  if precip is not None:
    if precip >= 50:
      risks.append({
          "type": "HEAVY_RAIN",
          "level": "HIGH",
          "desc": (
              f"Torrential Rain Warning: Accumulated precip reached {precip}"
              " mm (≥50mm)."
          ),
      })
    elif precip >= 35:
      risks.append({
          "type": "HEAVY_RAIN",
          "level": "MODERATE",
          "desc": (
              f"Heavy Rainfall Caution: Accumulated precip reached {precip} mm"
              " (≥35mm)."
          ),
      })

  return risks

# =====================================================================
# 3. model loading and system prompt configuration
# =====================================================================
MODEL_NAME = 'Qwen/Qwen3.5-2B'
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print(f'Loading model {MODEL_NAME}...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

temp_config = GenerationConfig(do_sample=False, max_new_tokens=1024, temperature=0.6)

SYSTEM_PROMPT = """You are CLIMES-Monitor, an intelligent weather, climate, and public-health assistant serving Uppsala, Sweden.
RULES:
1. Your job is to answer questions about weather, temperature, forecasts, climate safety, heatwaves, extreme weather (cold/wind/rain), and public health precautions.
2. For ALL other questions related to weather, climate, or health (even if they don't explicitly mention Uppsala), be helpful, friendly, and use the provided data and RAG context.
3. Grounding & Zero-Hallucination on Current Status (FOR ON-TOPIC QUERIES):
   - Always check the [REALTIME STATUS EVALUATION] section in the user prompt.
   - If weather is NORMAL, state current actual temperature first, then present RAG/manual guidelines strictly as PRECAUTIONARY / HYPOTHETICAL advice. Never lie or claim there is an ongoing heatwave when status is normal.
   - If the user query is on-topic, always provide a helpful answer, even if it's in other languages.
4. If the user query is off-topic, make it about Uppsala weather, climate, or public health. If you cannot answer, politely decline and say: "I cannot provide information on topics unrelated to Uppsala weather, climate, or public health."
5. When asking about a hypothetical extreme weather scenario, always clarify that the situation is hypothetical and provide safety advice based on the RAG/manual context, but based on the scenario, not the current actual weather.
"""
adapter_path = "./my_weather_lora_adapter"
# save LoRA weights and tokenizer to the specified path
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA weights have been successfully saved to: {adapter_path}")

def query_rag_knowledge(query_text: str, top_k: int = 3, score_threshold: float = 0.35) -> str:
  if not os.path.exists('./knowledge'):
    return ''
  vectorstore = Chroma(persist_directory='./knowledge', embedding_function=embeddings)
  results_with_scores = vectorstore.similarity_search_with_relevance_scores(query_text, k=top_k)
  context_chunks = []
  for doc, score in results_with_scores:
    if score >= score_threshold:
      source_name = os.path.basename(doc.metadata.get('source', 'Guide'))
      context_chunks.append(f"[{source_name}]: {doc.page_content}")
  return '\n\n'.join(context_chunks)

def extract_params(user_text):
  days_match = re.search(r'(\d+)\s*(?:days|day)', user_text, re.IGNORECASE)
  temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:℃|°C|degrees)', user_text, re.IGNORECASE)
  days = int(days_match.group(1)) if days_match else None
  temp = float(temp_match.group(1)) if temp_match else None
  return days, temp

def bot_response(message, history):
    # 1. extract user text from message (string, dict, or list)
    if isinstance(message, str):
        user_text = message.strip()
    elif isinstance(message, dict):
        user_text = str(message.get("text", "")).strip()
    elif isinstance(message, list):
        first_item = message[0] if len(message) > 0 else ""
        if isinstance(first_item, str):
            user_text = first_item.strip()
        elif isinstance(first_item, dict):
            user_text = str(first_item.get("text", "")).strip()
        else:
            user_text = str(first_item).strip()
    else:
        user_text = str(message).strip()

    # Hypothetical heatwave query → historical frequency (deterministic)
    days_m = re.search(r'(\d+)\s*(?:days?|dygn)', user_text, re.I)
    temp_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:°\s*C|℃|degrees?)', user_text, re.I)
    hist_stats = None
    if temp_m:
        days = int(days_m.group(1)) if days_m else 3   # default to 3 days if user only says a temperature
        try:
            hist_stats = historical_heatwave_stats(days, float(temp_m.group(1)))
        except Exception:
            hist_stats = None

    # 2. extract location coordinates from user text
    loc_info = extract_location_coords(user_text)

    # 3. fetch weather data from WeatherMetro API
    weather = fetch_weathermetro_data(
        lat=loc_info["lat"],
        lon=loc_info["lon"],
        place_name=loc_info["name"],
        user_text=user_text,
    )

    # 4. RAG knowledge retrieval based on user query
    rag_context = query_rag_knowledge(user_text, top_k=3, score_threshold=0.35)

    # 5. Extract relevant weather parameters for extreme weather risk evaluation
    current_temp = weather.get("temp", None)
    wind_speed = weather.get("wind_speed", None)
    humidity_val = weather.get("humidity", "N/A")
    precip = weather.get("precipitation", 0.0)

    # for forecast intent, compute max forecasted temp and precip for risk evaluation
    max_forecast_temp = None
    max_forecast_precip = None
    if weather.get("intent") == "forecast" and "daily" in weather:
        forecast_temps = [
            d.get("temp_max")
            for d in weather["daily"]
            if d.get("temp_max") is not None
        ]
        forecast_precips = [
            d.get("precip_total")
            for d in weather["daily"]
            if d.get("precip_total") is not None
        ]
        if forecast_temps:
            max_forecast_temp = max(forecast_temps)
        if forecast_precips:
            max_forecast_precip = max(forecast_precips)

    eval_temp = (
        max_forecast_temp if max_forecast_temp is not None else current_temp
    )
    eval_precip = (
        max_forecast_precip if max_forecast_precip is not None else precip
    )

    # 6. Python function to compute extreme weather risks based on thresholds
    detected_risks = compute_extreme_weather_risk(
        temp=eval_temp, wind_speed=wind_speed, precip=eval_precip
    )

    if detected_risks:
        risk_descriptions = "\n".join(
            [f"- [{r['type']}] ({r['level']}): {r['desc']}" for r in detected_risks]
        )
        extreme_weather_status = (
            f"STATUS: EXTREME WEATHER WARNING ACTIVE!\n"
            f"Detected Risk Conditions:\n{risk_descriptions}\n"
            f"INSTRUCTION: Treat the relevant extreme weather safety advice as URGENT and CURRENT."
        )
    else:
        temp_disp = f"{current_temp}°C" if current_temp is not None else "normal"
        wind_disp = f"{wind_speed} km/h" if wind_speed is not None else "N/A"
        extreme_weather_status = (
            f"STATUS: NORMAL WEATHER (No active severe weather warnings. Current Temp: {temp_disp}, Wind: {wind_disp}, Precip: {precip} mm).\n"
            f"INSTRUCTION: Weather is currently normal. If user asks about extreme conditions (heatwave, coldwave, storm, heavy rain), "
            f"state that conditions are currently normal and present safety measures strictly as PRECAUTIONARY / HYPOTHETICAL advice."
        )

        # 7. concatenate Prompt
    prompt_sections = [
        f"### REALTIME STATUS EVALUATION (GUARDRAIL)\n{extreme_weather_status}",
        f"### CURRENT USER QUERY\n{user_text}",
    ]

    # weather data section
    if weather.get("intent") == "forecast" and "daily" in weather:
        forecast_str_list = [
            f"- {d.get('date')}: High {d.get('temp_max')}°C, Low {d.get('temp_min')}°C, Precip: {d.get('precip_total', 0)}mm"
            for d in weather["daily"]
        ]
        prompt_sections.append(
            f"### WEATHER FORECAST DATA ({weather['place_name'].upper()})\n"
            + "\n".join(forecast_str_list)
        )
    else:
        prompt_sections.append(
            f"### REAL-TIME WEATHER OBSERVATION ({weather['place_name'].upper()})\n"
            f"- Temp: {weather.get('temp', 'N/A')} °C | Condition: {weather.get('condition', 'Live Weather')}\n"
            f"- Relative Humidity: {humidity_val}% | Wind Speed: {wind_speed if wind_speed is not None else 'N/A'} km/h | Precip: {precip} mm"
        )

    # Historical frequency (only when user asked a days+temp hypothetical)
    if hist_stats and "error" not in hist_stats:
        prompt_sections.append(
            "### HISTORICAL FREQUENCY (DETERMINISTIC, SMHI 1985–2026)\n"
            + str(hist_stats)
        )

    # RAG context
    if rag_context:
        prompt_sections.append(
            f"### LOCAL MANUAL CONTEXT (RAG)\n```text\n{rag_context}\n```"
        )

    current_instruction = "\n\n".join(prompt_sections)

    # 8. construct Conversation with context and history
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history_turns = 3
    recent_history = history[-max_history_turns:] if history else []

    for turn in recent_history:
        if isinstance(turn, dict):
            conversation.append({"role": turn["role"], "content": turn["content"]})
        elif isinstance(turn, (list, tuple)):
            conversation.append({"role": "user", "content": turn[0]})
            conversation.append({"role": "assistant", "content": turn[1]})

    conversation.append({"role": "user", "content": current_instruction})

    # 9. stream model response with TextIteratorStreamer
    text_prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs, streamer=streamer, generation_config=temp_config
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text, None

    # 10. generate audio for the final response
    audio_path = generate_audio(partial_text)
    yield partial_text, audio_path

# =====================================================================
# 5. UI interface with Gradio
# =====================================================================
with gr.Blocks(title="CLIMES-Monitor: Uppsala Weather & Health") as demo:
  gr.Markdown("# 🌤️ CLIMES-Monitor: Uppsala Weather & Public-Health Agent")

  with gr.Row():
    with gr.Column(scale=4):
      chatbot = gr.Chatbot(label="Chat History", height=500)
      with gr.Row():
        user_input = gr.Textbox(placeholder="Ask about Uppsala weather or safety advice...", show_label=False, scale=6)
        submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
        stop_btn = gr.Button("Stop 🛑", variant="stop", scale=1)
      clear_btn = gr.Button("Clear History 🗑️")

    with gr.Column(scale=2):
      gr.Markdown("### 🔊 Voice Accessibility (TTS)")
      audio_player = gr.Audio(label="Audio Advice", autoplay=True, type="filepath")

  def user_submit(message, history):
    if not message.strip():
      return "", history
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    return "", history

  def bot_streaming(history):
    user_message = history[-2]["content"]
    past_history = history[:-2]
    for text_chunk, audio_file in bot_response(user_message, past_history):
      history[-1]["content"] = text_chunk
      yield history, audio_file

  submit_event = user_input.submit(user_submit, [user_input, chatbot], [user_input, chatbot]).then(bot_streaming, [chatbot], [chatbot, audio_player])
  btn_event = submit_btn.click(user_submit, [user_input, chatbot], [user_input, chatbot]).then(bot_streaming, [chatbot], [chatbot, audio_player])
  stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, btn_event])
  clear_btn.click(lambda: ([], None), None, [chatbot, audio_player])

if __name__ == "__main__":
  demo.launch(share=True, debug=True)

