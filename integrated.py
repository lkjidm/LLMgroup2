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

# 避免 transformers 打印过多日志
logging.set_verbosity_warning()

# =====================================================================
# 1. 数据准备与工具函数
# =====================================================================
try:
  df_hist = pd.read_csv('./uppsala_tmax_1985_2026.csv', parse_dates=['date'])
  print('Historical weather data loaded successfully! Shape:', df_hist.shape)
except Exception as e:
  print(
      'Warning: CSV file not found. Creating a dummy dataset for demonstration.'
  )
  dates = pd.date_range('1985-06-01', '2026-06-30')
  df_hist = pd.DataFrame(
      {'date': dates, 'tmax': np.random.uniform(10, 32, len(dates))}
  )

WEATHERMETRO_API_KEY = 'uXQyXC4iqxCVdzGSCbQ0uGmir0CzRFtU'
UPPSALA_PLACES_JSON = './knowledge/uppsala_all_places.json'

DEFAULT_LAT = 59.8586
DEFAULT_LON = 17.6389
DEFAULT_PLACE_NAME = 'Uppsala Centrum'

_weather_cache = {}


def load_uppsala_places():
  """读取本地 2800+ 地名节点 JSON"""
  if os.path.exists(UPPSALA_PLACES_JSON):
    with open(UPPSALA_PLACES_JSON, 'r', encoding='utf-8') as f:
      return json.load(f)
  print('⚠️ 警告: 地名 JSON 文件不存在，将使用默认市中心坐标。')
  return []


UPPSALA_PLACES = load_uppsala_places()


# ==========================================
# 2. 地名动态识别与坐标提取函数
# ==========================================
def extract_location_coords(user_text: str):
  """从用户输入中提取乌普萨拉具体地名并返回对应的经纬度"""
  text_lower = user_text.lower()

  sorted_places = sorted(
      UPPSALA_PLACES, key=lambda x: len(x['name']), reverse=True
  )

  for place in sorted_places:
    name = place['name']
    if len(name) > 1 and name.lower() in text_lower:
      return {
          'found': True,
          'name': name,
          'lat': place['lat'],
          'lon': place['lon'],  # OpenStreetMap 导出的键名为 lon
          'type': place.get('type', 'landmark'),
      }

  return {
      'found': False,
      'name': DEFAULT_PLACE_NAME,
      'lat': DEFAULT_LAT,
      'lon': DEFAULT_LON,
      'type': 'city_center',
  }


# ==========================================
# 3. 动态 WeatherMetro API 查询（适配 v2 结构 + 兼容 lng 参数）
# ==========================================
def detect_weather_intent(user_text: str):
    """判断用户是在查询实时天气还是未来预测"""
    text = user_text.lower()
    
    # 预测关键词正则匹配
    forecast_keywords = [
        "tomorrow", "forecast", "next week", "future", "will it", "predict"    
    ]
    is_forecast = any(kw in text for kw in forecast_keywords)
    
    # 提取查询天数（如 "3 days forecast" -> 3 天，默认预测 3 天）
    days_match = re.search(r'(\d+)\s*(?:days|day|天)', text)
    days = int(days_match.group(1)) if days_match else (3 if is_forecast else 1)
    
    return "forecast" if is_forecast else "current", days


def fetch_weathermetro_data(
    lat: float, lon: float, place_name: str = "Uppsala", user_text: str = ""
) -> dict:
    """适配 WeatherMetro v2 按小时 (hourly) 返回的真实 forecast 数据结构"""
    intent, forecast_days = detect_weather_intent(user_text)

    now = time.time()
    cache_key = f"{intent}_{round(lat, 3)}_{round(lon, 3)}"

    # 1. 检查缓存
    if cache_key in _weather_cache:
        cached_data, timestamp = _weather_cache[cache_key]
        if now - timestamp < 600:
            return cached_data

    headers = {"User-Agent": "UppsalaClimateBot/1.0"}

    # 2. 根据意图决定 Endpoint
    if intent == "forecast":
        url = "https://weathermetro.com/api/v2/forecast"
        params = {
            "api_key": WEATHERMETRO_API_KEY,
            "lat": lat,
            "lng": lon,  # WeatherMetro 参数名必须是 lng
            "days": forecast_days,
            "units": "metric",
        }
    else:
        url = "https://weathermetro.com/api/v2/current"
        params = {
            "api_key": WEATHERMETRO_API_KEY,
            "lat": lat,
            "lng": lon,
            "units": "metric",
        }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            resp_data = data.get("response", {}).get("data", {})

            # -----------------------------------------------------------
            # 分支 A: 处理 forecast (逐小时 hourly 列表聚合为按天 daily)
            # -----------------------------------------------------------
            if intent == "forecast":
                hourly = resp_data.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                precips = hourly.get("precipitation", [])

                # 按日期分组储存数据: {"2026-10-14": {"temps": [...], "precips": [...]}}
                daily_grouped = collections.defaultdict(
                    lambda: {"temps": [], "precips": []}
                )

                for t, temp, precip in zip(times, temps, precips):
                    date_key = t.split("T")[0]  # 提取 YYYY-MM-DD
                    daily_grouped[date_key]["temps"].append(temp)
                    daily_grouped[date_key]["precips"].append(precip)

                # 计算每天的最高温、最低温和总降水量
                parsed_daily = []
                for date_key, values in daily_grouped.items():
                    if values["temps"]:
                        parsed_daily.append({
                            "date": date_key,
                            "temp_max": round(max(values["temps"]), 1),
                            "temp_min": round(min(values["temps"]), 1),
                            "precip_total": round(sum(values["precips"]), 1),
                        })

                result = {
                    "status": "success",
                    "intent": "forecast",
                    "place_name": place_name,
                    "lat": lat,
                    "lng": lon,
                    "forecast_days": forecast_days,
                    "daily": parsed_daily,  # 返回聚合好的每天预报列表
                    "timestamp": time.strftime("%H:%M:%S"),
                }

            # -----------------------------------------------------------
            # 分支 B: 处理 current 实时数据
            # -----------------------------------------------------------
            else:
                current = data.get("response", {}).get("current", {})
                result = {
                    "status": "success",
                    "intent": "current",
                    "place_name": place_name,
                    "lat": lat,
                    "lng": lon,
                    "temp": current.get("temperature_2m", 7.4),
                    "humidity": current.get("relative_humidity_2m", 80),
                    "wind_speed": current.get("wind_speed_10m", 10),
                    "condition": "Live Weather",
                    "timestamp": time.strftime("%H:%M:%S"),
                }

            _weather_cache[cache_key] = (result, now)
            return result
        else:
            print(
                f"WeatherMetro API Error {response.status_code}:"
                f" {response.text}"
            )
    except Exception as e:
        print(f"WeatherMetro API Exception: {e}")

    # 降级兜底方案
    return {
        "status": "fallback",
        "intent": intent,
        "place_name": place_name,
        "lat": lat,
        "lng": lon,
        "temp": 7.4,
        "condition": "Fallback Weather",
        "timestamp": time.strftime("%H:%M:%S"),
    }


def find_heatwaves(df, threshold, min_duration=3):
  temp = df.copy()
  temp['is_hot'] = temp['tmax'] >= threshold
  temp['group'] = (temp['is_hot'] != temp['is_hot'].shift()).cumsum()
  temp.loc[~temp['is_hot'], 'group'] = 0

  hw = (
      temp[temp['is_hot']]
      .groupby('group')
      .agg(
          start=('date', 'min'),
          end=('date', 'max'),
          duration=('date', 'count'),
          max_tmax=('tmax', 'max'),
          mean_tmax=('tmax', 'mean'),
      )
      .reset_index(drop=True)
  )
  return hw[hw['duration'] >= min_duration].reset_index(drop=True)


def compute_heatwave_risk(days: int, temp: float, df_hist=df_hist):
  """确定性计算乌普萨拉高温热浪风险的核心函数"""
  if not isinstance(days, (int, np.integer)) or days < 1:
    return {
        'level': 'invalid',
        'comment': 'Number of days must be an integer ≥ 1.',
    }
  if not isinstance(temp, (int, float, np.floating)) or temp < -40 or temp > 45:
    return {'level': 'invalid', 'comment': 'Temperature must be realistic (°C).'}

  record_max = float(df_hist['tmax'].max())
  record_date = str(df_hist.loc[df_hist['tmax'].idxmax(), 'date'].date())

  if temp >= 30 and days >= 3:
    hist = find_heatwaves(df_hist, threshold=30.0, min_duration=3)
    count = len(hist)
    score = min(0.95, 0.60 + 0.025 * max(0, 15 - count))
    return {
        'level': 'level2',
        'description': 'Extreme heat (Level 2): ≥30°C for ≥3 days',
        'historical_count': count,
        'risk_score': float(round(score, 2)),
        'comment': (
            'Matches SMHI yellow-warning level. Rare but high health impact in'
            ' Uppsala.'
        ),
        'record_max': record_max,
        'record_date': record_date,
    }
  if temp >= 33 and days >= 1:
    extreme_days = int((df_hist['tmax'] >= 33).sum())
    score = min(0.75, 0.45 + 0.02 * extreme_days)
    return {
        'level': 'extreme_single',
        'description': 'Extreme single day: ≥33°C (even for 1 day)',
        'historical_count': extreme_days,
        'risk_score': float(round(score, 2)),
        'comment': (
            'Very high temperature for Uppsala. Notable even as a single day.'
        ),
        'record_max': record_max,
        'record_date': record_date,
    }
  if temp >= 27 and days >= 3:
    hist = find_heatwaves(df_hist, threshold=27.0, min_duration=3)
    count = len(hist)
    score = min(0.80, 0.30 + 0.006 * count)
    return {
        'level': 'level1',
        'description': 'Health-relevant heat (Level 1): ≥27°C for ≥3 days',
        'historical_count': count,
        'risk_score': float(round(score, 2)),
        'comment': (
            'Temperature range where Swedish studies show clear increases in'
            ' mortality risk.'
        ),
        'record_max': record_max,
        'record_date': record_date,
    }
  if temp >= 25 and days >= 5:
    hist = find_heatwaves(df_hist, threshold=25.0, min_duration=5)
    count = len(hist)
    score = min(0.65, 0.18 + 0.004 * count)
    return {
        'level': 'level3',
        'description': 'Classic SMHI heatwave (Level 3): ≥25°C for ≥5 days',
        'historical_count': count,
        'risk_score': float(round(score, 2)),
        'comment': 'Official SMHI climatological definition of a heatwave.',
        'record_max': record_max,
        'record_date': record_date,
    }

  return {
      'level': 'none',
      'description': 'Below defined heatwave thresholds',
      'historical_count': 0,
      'risk_score': 0.05,
      'comment': 'Does not meet any defined heat criteria.',
      'record_max': record_max,
      'record_date': record_date,
  }


# =====================================================================
# 4. 模型加载
# =====================================================================
MODEL_NAME = 'Qwen/Qwen3.5-2B'
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print(f'Loading model {MODEL_NAME}...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

temp_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
    temperature=0.6,
)


SYSTEM_PROMPT = """You are CLIMES-Monitor, an intelligent weather, climate, and public-health assistant specifically serving Uppsala, Sweden.

Your core guiding rules:
1. Intent Recognition & Topic Scope:
   - Topic Scope: Weather, temperature, heatwaves, extreme climate events, localized geography in Uppsala, and public health / heat safety guidelines.
   - Any query asking about weather, heatwave safety tips, cooling measures, or health precautions IS ON-TOPIC and directly relevant. Treat general climate/safety questions as naturally applied to Uppsala residents.
   - ONLY refuse queries that are completely unrelated to weather, climate, or public health (e.g., movies, programming code troubleshooting, sports, cooking recipes, gaming).

2. Direct & Helpful Advice:
   - When asked for safety tips during heatwaves or extreme weather, provide clear, practical, and location-aware recommendations (e.g., indoor cooling, hydration, vulnerable group precautions) try to use RAG knowledge provided, still use genral public health guidance if no RAG knowledge is available.

3. Grounding & Accuracy:
   - Cite exact metrics (measured temperatures, risk levels, historical counts) when tool results or RAG contexts are provided. Never invent fictional weather data.

4. Refusal Trigger (Strictly Constrained):
   - ONLY if the query is strictly off-topic (e.g., "write python code", "who is Harry Potter"), reply with:
     "I cannot provide information on topics unrelated to Uppsala weather, climate, or public health."
     - Avoid JSON format in your final response.
"""

EXTRACTION_PROMPT = (
    'Extract duration (days) and temperature (°C) for Uppsala heatwave'
    ' assessment from user query.\nReturn ONLY a valid JSON with keys: "days"'
    ' (int or null), "temperature" (float or null).\nExample user: "4 days at 31'
    ' degrees" -> {"days": 4, "temperature": 31.0}\nExample user: "31 degrees"'
    ' -> {"days": null, "temperature": 31.0}'
)


def query_rag_knowledge(
    query_text: str, top_k: int = 3, score_threshold: float = 0.35
) -> str:
  if not os.path.exists('./knowledge'):
    return ''

  vectorstore = Chroma(
      persist_directory='./knowledge', embedding_function=embeddings
  )
  results_with_scores = vectorstore.similarity_search_with_relevance_scores(
      query_text, k=top_k
  )

  context_chunks = []
  for doc, score in results_with_scores:
    if score >= score_threshold:
      page_num = doc.metadata.get('page', 0) + 1
      source_name = os.path.basename(doc.metadata.get('source', '官方指南'))
      context_chunks.append(
          f'[{source_name} 第 {page_num} 页] (相关度:'
          f' {score:.2f}):\n{doc.page_content}'
      )

  if not context_chunks:
    return ''

  return '\n\n'.join(context_chunks)


def extract_params(user_text):
  days_match = re.search(r'(\d+)\s*(?:天|days|day)', user_text, re.IGNORECASE)
  temp_match = re.search(
      r'(\d+(?:\.\d+)?)\s*(?:度|℃|°C|degrees)', user_text, re.IGNORECASE
  )

  days = int(days_match.group(1)) if days_match else None
  temp = float(temp_match.group(1)) if temp_match else None

  if days is None or temp is None:
    messages = [
        {'role': 'system', 'content': EXTRACTION_PROMPT},
        {'role': 'user', 'content': user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
    with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    output_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )

    try:
      json_str = re.search(r'\{.*\}', output_text, re.DOTALL).group(0)
      data = json.loads(json_str)
      if days is None:
        days = data.get('days')
      if temp is None:
        temp = data.get('temperature')
    except Exception:
      pass

  return days, temp


# =====================================================================
# 5. Gradio 对话响应逻辑
# =====================================================================
def bot_response(message, history):
  # 1. 提取用户输入的文本内容（兼容字符串与 Gradio 字典输入）
  user_text = (
      message.strip()
      if isinstance(message, str)
      else message.get("text", "").strip()
  )

  # 2. 提取地名与坐标（OpenStreetMap 数据返回 lat 与 lon）
  loc_info = extract_location_coords(user_text)

  # 3. 智能获取 WeatherMetro 天气数据（自动识别实时 current 或 预测 forecast）
  weather = fetch_weathermetro_data(
      lat=loc_info["lat"],
      lon=loc_info["lon"],
      place_name=loc_info["name"],
      user_text=user_text,
  )

  # 4. RAG 官方防灾知识库检索
  rag_context = query_rag_knowledge(user_text, top_k=3, score_threshold=0.35)

  # 5. 参数提取与热浪风险评估
  days, temp_param = extract_params(user_text)
  eval_temp = temp_param if temp_param is not None else weather.get("temp")

  tool_result = None
  if days is not None and eval_temp is not None:
    try:
      tool_result = compute_heatwave_risk(
          days=int(days), temp=float(eval_temp)
      )
    except Exception as e:
      print("Tool execution error:", e)

  # =========================================================================
  # 6. 关键修复：优先在最外层初始化 prompt_sections 列表！
  # =========================================================================
  prompt_sections = [f"### CURRENT USER QUERY\n{user_text}"]

  # 6a. 天气信息区块：根据意图区分【预报】与【实时】
  if weather.get("intent") == "forecast" and "daily" in weather:
    forecast_str_list = []
    for day in weather["daily"]:
      date_str = day.get("date", "Upcoming")
      max_t = day.get("temp_max", "N/A")
      min_t = day.get("temp_min", "N/A")
      precip = day.get("precip_total", 0)
      forecast_str_list.append(
          f"- {date_str}: High {max_t}°C, Low {min_t}°C, Total Precip:"
          f" {precip}mm"
      )

    forecast_text = (
        "\n".join(forecast_str_list)
        if forecast_str_list
        else "No daily forecast available."
    )
    prompt_sections.append(
        f"### WEATHER FORECAST DATA ({weather['place_name'].upper()})\n"
        f"- Location: {weather['place_name']} (lat: {weather['lat']}, lng:"
        f" {weather['lng']})\n"
        f"{forecast_text}"
    )
  else:
    # 使用 .get() 安全防御 KeyError
    prompt_sections.append(
        f"### REAL-TIME WEATHER OBSERVATION"
        f" ({weather['place_name'].upper()})\n"
        f"- Location: {weather['place_name']} (lat: {weather['lat']}, lng:"
        f" {weather['lng']})\n"
        f"- Measured Temp: {weather.get('temp', 'N/A')} °C | Condition:"
        f" {weather.get('condition', 'Live Weather')}\n"
        f"- Humidity: {weather.get('humidity', 'N/A')}% | Wind:"
        f" {weather.get('wind_speed', 'N/A')} m/s"
    )

  # 6b. RAG 知识库区块
  if rag_context:
    prompt_sections.append(
        "### LOCAL GEOGRAPHY & MANUAL CONTEXT"
        f" (RAG)\n```text\n{rag_context}\n```"
    )

  # 6c. 热浪风险评估结果区块
  if tool_result and tool_result.get("level") != "invalid":
    prompt_sections.append(
        "### EXTREME HEATWAVE RISK METRICS\n"
        f"- Risk Level: {tool_result.get('level')}"
        f" ({tool_result.get('description')})\n"
        f"- Historical Frequency (1985-2026):"
        f" {tool_result.get('historical_count')} times\n"
        f"- Uppsala Record High: {tool_result.get('record_max')} °C\n"
        f"- Note: {tool_result.get('comment')}"
    )

  # 6d. 指导指令
  prompt_sections.append(
      "### INSTRUCTIONS\n"
      "1. This query is directly relevant to Uppsala weather, climate, or"
      " health safety.\n"
      "2. State the target location and weather condition/forecast naturally"
      " in the first sentence.\n"
      "3. Provide practical, location-aware recommendations grounded in the"
      " RAG context if applicable.\n"
      "4. Maintain a warm, authoritative, and friendly tone."
  )

  # 拼装最终的 User Instruction
  current_instruction = "\n\n".join(prompt_sections)

  # 7. 构建包含多轮对话历史上下文（Context History）的 Conversation 结构
  conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

  # 保留最近 3 轮历史对话
  max_history_turns = 3
  recent_history = history[-max_history_turns:] if history else []

  for turn in recent_history:
    if isinstance(turn, dict):
      conversation.append({"role": turn["role"], "content": turn["content"]})
    elif isinstance(turn, (list, tuple)):
      conversation.append({"role": "user", "content": turn[0]})
      conversation.append({"role": "assistant", "content": turn[1]})

  conversation.append({"role": "user", "content": current_instruction})

  # 8. 应用 Chat Template 并生成回复
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

  thread = Thread(target=model.generate, kwargs=generation_kwargs)
  thread.start()

  partial_text = ""
  for new_text in streamer:
    partial_text += new_text
    yield partial_text


demo = gr.ChatInterface(
    bot_response,
    multimodal=False,
    title='CLIMES-Monitor: Uppsala Weather & Heatwave Assistant',
)

if __name__ == '__main__':
  demo.launch()