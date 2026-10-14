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
def fetch_weathermetro_data(
    lat: float, lon: float, place_name: str = 'Uppsala'
) -> dict:
  """根据经纬度实时查询 WeatherMetro 天气接口"""
  now = time.time()
  cache_key = f'{round(lat, 3)}_{round(lon, 3)}'

  # 1. 检查 10 分钟缓存
  if cache_key in _weather_cache:
    cached_data, timestamp = _weather_cache[cache_key]
    if now - timestamp < 600:
      return cached_data

  # 2. WeatherMetro 必须接收 lng 作为经度键名
  url = 'https://weathermetro.com/api/v2/current'
  params = {
      'api_key': WEATHERMETRO_API_KEY,
      'lat': lat,
      'lng': lon,  # 将 OpenStreetMap 的 lon 传入 WeatherMetro 的 lng 参数
      'units': 'metric',
  }
  headers = {'User-Agent': 'UppsalaClimateBot/1.0'}

  try:
    response = requests.get(url, params=params, headers=headers, timeout=8)
    if response.status_code == 200:
      data = response.json()
      # 根据测试单元格解析真实的 JSON 层级: response -> current
      current = data.get('response', {}).get('current', {})

      temp = current.get('temperature_2m', 7.4)
      humidity = current.get('relative_humidity_2m', 80)
      wind_speed = current.get('wind_speed_10m', 10)

      result = {
          'status': 'success',
          'place_name': place_name,
          'lat': lat,
          'lng': lon,
          'temp': temp,
          'humidity': humidity,
          'wind_speed': wind_speed,
          'condition': 'Live Weather',
          'timestamp': time.strftime('%H:%M:%S'),
      }
      _weather_cache[cache_key] = (result, now)
      return result
    else:
      print(f'WeatherMetro API Error {response.status_code}: {response.text}')
  except Exception as e:
    print(f'WeatherMetro API Call Exception: {e}')

  # 降级兜底方案
  return {
      'status': 'fallback',
      'place_name': place_name,
      'lat': lat,
      'lng': lon,
      'temp': 7.4,
      'humidity': 85,
      'wind_speed': 10,
      'condition': 'Fallback Weather',
      'timestamp': time.strftime('%H:%M:%S'),
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
MODEL_NAME = 'Qwen/Qwen3.5-0.8B'
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print(f'Loading model {MODEL_NAME}...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

temp_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
    temperature=0.3,
)

SYSTEM_PROMPT = """You are CLIMES-Monitor, an intelligent weather, climate, and public-health assistant for Uppsala, Sweden (including neighborhoods like Flogsta, Luthagen, and Gottsunda).

Your core guiding rules:
1. Flexible & Friendly Intent Handling:
   - For general everyday weather queries or local inquiries, provide warm, professional, and natural responses. Do NOT forcefully fabricate heatwave warnings or extreme weather risks when none are implied.
   - For questions about historical comparison, extreme temperatures, or multi-day heat scenarios, use precise numbers and tool calculations.
   - Identify the user's specific location (e.g., Flogsta, Luthagen) if mentioned.

2. Strict Grounding (Zero Hallucination):
   - Never invent temperatures, historical occurrences, or risk numbers.
   - When background tool calculation results or historical database results are provided in the prompt, you MUST explicitly cite those exact numbers (e.g., historical count, warning level, record high).

3. Actionable & Localized Health Guidance:
   - Provide clear, practical public-health advice relevant to Uppsala residents (e.g., hydration, cooling, vulnerable groups) grounded in the provided reference materials or official guidelines.
   - If a user's query lacks specific parameter details (e.g., mentions 31°C but not duration), provide a reasonable response based on a standard 1-day or 3-day scenario, while gently inviting them to specify the duration for more precise calculations.

4. Formatting:
   - Use clean Markdown (including Markdown tables when summarizing metrics).
   - Avoid JSON format in your final response.
   - DO NOT hallucinate rules not found in the RAG or tool output.
   - If the user asks about off-topic subjects (including movies, entertainment, sports, code troubleshooting, general chit-chat, cooking, or general history outside Uppsala climate context), YOU MUST REFUSE TO ANSWER.
   - DO NOT attempt to answer, summarize, or chat about off-topic themes under any circumstances.
   - Answer "I cannot provide information on topics unrelated to Uppsala weather, climate, or public health." if the user asks about off-topic subjects.
   
Always remebmer: Your responses must be grounded in the provided RAG knowledge and tool outputs. If no relevant data is available, respond naturally without fabricating information.
Some knowledge is U.S. related, but you must adapt it to Uppsala's context.
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
  user_text = (
      message.strip()
      if isinstance(message, str)
      else message.get('text', '').strip()
  )

  # 提取地名与坐标
  loc_info = extract_location_coords(user_text)

  # ✅ 关键修正：传入 loc_info['lon']
  weather = fetch_weathermetro_data(
      lat=loc_info['lat'], lon=loc_info['lon'], place_name=loc_info['name']
  )

  # RAG 知识检索
  rag_context = query_rag_knowledge(user_text, top_k=3, score_threshold=0.35)

  # 参数提取与评估
  days, temp_param = extract_params(user_text)
  eval_temp = temp_param if temp_param is not None else weather.get('temp')

  tool_result = None
  if days is not None and eval_temp is not None:
    try:
      tool_result = compute_heatwave_risk(
          days=int(days), temp=float(eval_temp)
      )
    except Exception as e:
      print('Tool execution error:', e)

  prompt_sections = [f'### USER QUERY\n{user_text}']

  # 实时天气
  prompt_sections.append(
      f"### REAL-TIME WEATHER OBSERVATION ({weather['place_name'].upper()})\n"
      f"- Location: {weather['place_name']} (lat: {weather['lat']}, lng:"
      f" {weather['lng']})\n"
      f"- Measured Temp: {weather['temp']} °C | Condition:"
      f" {weather['condition']}\n"
      f"- Humidity: {weather['humidity']}% | Wind: {weather['wind_speed']} m/s"
  )

  if rag_context:
    prompt_sections.append(
        f'### LOCAL GEOGRAPHY & MANUAL CONTEXT (RAG)\n```text\n{rag_context}\n```'
    )

  if tool_result and tool_result.get('level') != 'invalid':
    prompt_sections.append(
        f"### EXTREME HEATWAVE RISK METRICS\n- Risk Level:"
        f" {tool_result.get('level')} ({tool_result.get('description')})\n-"
        ' Historical Frequency (1985-2026):'
        f" {tool_result.get('historical_count')} times\n- Uppsala Record"
        f" High: {tool_result.get('record_max')} °C\n- Note:"
        f" {tool_result.get('comment')}"
    )

  prompt_sections.append(
      '### INSTRUCTIONS\n'
      '1. State the exact target location and current measured weather in the'
      ' first sentence.\n'
      '2. If an extreme weather scenario was calculated, summarize risk'
      ' metrics in a brief table.\n'
      '3. Provide 3 actionable, location-aware safety recommendations based on'
      ' RAG context.\n'
      '4. Keep response direct, authoritative, and within 250 words.'
  )

  prompt_instruction = '\n\n'.join(prompt_sections)

  conversation = [
      {'role': 'system', 'content': SYSTEM_PROMPT},
      {'role': 'user', 'content': prompt_instruction},
  ]

  text_prompt = tokenizer.apply_chat_template(
      conversation, tokenize=False, add_generation_prompt=True
  )
  inputs = tokenizer([text_prompt], return_tensors='pt').to(model.device)

  streamer = TextIteratorStreamer(
      tokenizer, skip_prompt=True, skip_special_tokens=True
  )
  generation_kwargs = dict(
      **inputs, streamer=streamer, generation_config=temp_config
  )

  thread = Thread(target=model.generate, kwargs=generation_kwargs)
  thread.start()

  partial_text = ''
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