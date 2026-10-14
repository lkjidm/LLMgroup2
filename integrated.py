import json
import re
from threading import Thread
import torch
import pandas as pd
import numpy as np
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    logging
)

# 避免 transformers 打印过多日志
logging.set_verbosity_warning()

# =====================================================================
# 1. 数据准备与工具函数 (移植自 agent.py / heatwave_tool.py)
# =====================================================================
# 尝试读取历史数据（如果在本地没有，请确保 uppsala_tmax_1985_2026.csv 在同一目录下）
try:
    df_hist = pd.read_csv("uppsala_tmax_1985_2026.csv", parse_dates=["date"])
    print("Historical weather data loaded successfully! Shape:", df_hist.shape)
except Exception as e:
    print("Warning: CSV file not found. Creating a dummy dataset for demonstration.")
    # 生成测试用模拟数据
    dates = pd.date_range("1985-06-01", "2026-06-30")
    df_hist = pd.DataFrame({"date": dates, "tmax": np.random.uniform(10, 32, len(dates))})

def find_heatwaves(df, threshold, min_duration=3):
    temp = df.copy()
    temp["is_hot"] = temp["tmax"] >= threshold
    temp["group"] = (temp["is_hot"] != temp["is_hot"].shift()).cumsum()
    temp.loc[~temp["is_hot"], "group"] = 0

    hw = (
        temp[temp["is_hot"]]
        .groupby("group")
        .agg(
            start=("date", "min"),
            end=("date", "max"),
            duration=("date", "count"),
            max_tmax=("tmax", "max"),
            mean_tmax=("tmax", "mean")
        )
        .reset_index(drop=True)
    )
    return hw[hw["duration"] >= min_duration].reset_index(drop=True)

def compute_heatwave_risk(days: int, temp: float, df_hist=df_hist):
    """确定性计算乌普萨拉高温热浪风险的核心函数"""
    if not isinstance(days, (int, np.integer)) or days < 1:
        return {"level": "invalid", "comment": "Number of days must be an integer ≥ 1."}
    if not isinstance(temp, (int, float, np.floating)) or temp < -40 or temp > 45:
        return {"level": "invalid", "comment": "Temperature must be realistic (°C)."}

    record_max = float(df_hist["tmax"].max())
    record_date = str(df_hist.loc[df_hist["tmax"].idxmax(), "date"].date())

    # Level 2
    if temp >= 30 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=30.0, min_duration=3)
        count = len(hist)
        score = min(0.95, 0.60 + 0.025 * max(0, 15 - count))
        return {
            "level": "level2",
            "description": "Extreme heat (Level 2): ≥30°C for ≥3 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Matches SMHI yellow-warning level. Rare but high health impact in Uppsala.",
            "record_max": record_max,
            "record_date": record_date
        }
    # Extreme single day
    if temp >= 33 and days >= 1:
        extreme_days = int((df_hist["tmax"] >= 33).sum())
        score = min(0.75, 0.45 + 0.02 * extreme_days)
        return {
            "level": "extreme_single",
            "description": "Extreme single day: ≥33°C (even for 1 day)",
            "historical_count": extreme_days,
            "risk_score": float(round(score, 2)),
            "comment": "Very high temperature for Uppsala. Notable even as a single day.",
            "record_max": record_max,
            "record_date": record_date
        }
    # Level 1
    if temp >= 27 and days >= 3:
        hist = find_heatwaves(df_hist, threshold=27.0, min_duration=3)
        count = len(hist)
        score = min(0.80, 0.30 + 0.006 * count)
        return {
            "level": "level1",
            "description": "Health-relevant heat (Level 1): ≥27°C for ≥3 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Temperature range where Swedish studies show clear increases in mortality risk.",
            "record_max": record_max,
            "record_date": record_date
        }
    # Level 3
    if temp >= 25 and days >= 5:
        hist = find_heatwaves(df_hist, threshold=25.0, min_duration=5)
        count = len(hist)
        score = min(0.65, 0.18 + 0.004 * count)
        return {
            "level": "level3",
            "description": "Classic SMHI heatwave (Level 3): ≥25°C for ≥5 days",
            "historical_count": count,
            "risk_score": float(round(score, 2)),
            "comment": "Official SMHI climatological definition of a heatwave.",
            "record_max": record_max,
            "record_date": record_date
        }

    return {
        "level": "none",
        "description": "Below defined heatwave thresholds",
        "historical_count": 0,
        "risk_score": 0.05,
        "comment": "Does not meet any defined heat criteria.",
        "record_max": record_max,
        "record_date": record_date
    }


# =====================================================================
# 2. 模型与 Tokenizer 加载 (来自 app.py)
# =====================================================================
MODEL_NAME = "Qwen/Qwen3.5-0.8B"  # 可替换为 Qwen/Qwen3.5-4B 等

print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

temp_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=512
)

SYSTEM_PROMPT = """You are CLIMES-Monitor, an intelligent weather, climate, and public-health assistant for Uppsala, Sweden (including neighborhoods like Flogsta, Luthagen, and Gottsunda).

Your core guiding rules:
1. Flexible & Friendly Intent Handling:
   - For general everyday weather queries or local inquiries, provide warm, professional, and natural responses. Do NOT forcefully fabricate heatwave warnings or extreme weather risks when none are implied.
   - For questions about historical comparison, extreme temperatures, or multi-day heat scenarios, use precise numbers and tool calculations.

2. Strict Grounding (Zero Hallucination):
   - Never invent temperatures, historical occurrences, or risk numbers.
   - When background tool calculation results or historical database results are provided in the prompt, you MUST explicitly cite those exact numbers (e.g., historical count, warning level, record high).

3. Actionable & Localized Health Guidance:
   - Provide clear, practical public-health advice relevant to Uppsala residents (e.g., hydration, cooling, vulnerable groups) grounded in the provided reference materials or official guidelines.
   - If a user's query lacks specific parameter details (e.g., mentions 31°C but not duration), provide a reasonable response based on a standard 1-day or 3-day scenario, while gently inviting them to specify the duration for more precise calculations.

4. Formatting:
   - Use clean Markdown (including Markdown tables when summarizing metrics).
   - Avoid JSON format in your final response."""


# =====================================================================
# 3. 参数提取与 Agent 决策层 (替换原 Gemini API 调用的核心)
# =====================================================================
EXTRACTION_PROMPT = """Extract duration (days) and temperature (°C) for Uppsala heatwave assessment from user query.
Return ONLY a valid JSON with keys: "days" (int or null), "temperature" (float or null).
Example user: "4 days at 31 degrees" -> {"days": 4, "temperature": 31.0}
Example user: "31 degrees" -> {"days": null, "temperature": 31.0}"""

def extract_params(user_text):
    """通过正则与小模型联合提取天数和温度参数"""
    # 优先试用正则表达式快速精准提取（最稳定）
    days_match = re.search(r'(\d+)\s*(?:天|days|day)', user_text, re.IGNORECASE)
    temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:度|℃|°C|degrees)', user_text, re.IGNORECASE)

    days = int(days_match.group(1)) if days_match else None
    temp = float(temp_match.group(1)) if temp_match else None

    # 如果正则没拿到，让 Qwen 来辅助解析一次
    if days is None or temp is None:
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        try:
            # 提取 JSON 结构
            json_str = re.search(r'\{.*\}', output_text, re.DOTALL).group(0)
            data = json.loads(json_str)
            if days is None: days = data.get("days")
            if temp is None: temp = data.get("temperature")
        except Exception:
            pass

    return days, temp


# =====================================================================
# 4. Gradio 对话响应逻辑 (结合了 app.py 的 Streamer 与 agent.py 的 Tool 逻辑)
# =====================================================================
def bot_response(message, history):
    user_text = message.strip() if isinstance(message, str) else message.get("text", "").strip()

    # 1. 尝试提取参数
    days, temp = extract_params(user_text)

    tool_result = None
    # 2. 如果成功提取出天数和温度，则执行确定性工具
    if days is not None and temp is not None:
        try:
            tool_result = compute_heatwave_risk(days=int(days), temp=float(temp))
        except Exception as e:
            print("Tool execution error:", e)

    # 3. 构造给 Qwen 的 Prompt (Grounding)
    if tool_result and tool_result.get("level") != "invalid":
        # 如果调用了工具，把准确的数据结果喂给模型，让其解释
        prompt_instruction = f"""User asked: '{user_text}'
We executed the Python Uppsala Heatwave Risk Tool and got these EXACT results:
- Level: {tool_result.get('level')}
- Description: {tool_result.get('description')}
- Historical occurrences (1985-2026): {tool_result.get('historical_count')} times
- Risk Score: {tool_result.get('risk_score')}
- Uppsala Record High: {tool_result.get('record_max')} °C (on {tool_result.get('record_date')})
- Tool Note: {tool_result.get('comment')}

Please answer the user's question clearly in natural text. Explain what this risk level means, cite the historical count and record high from the tool result above, and provide health advice for Uppsala residents.
Format key metrics nicely using a Markdown table."""
    else:
        # 如果参数不全，直接进行通用对话或请求澄清
        prompt_instruction = user_text

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_instruction}
    ]

    text_prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    # 4. 使用 Streamer 实现打字机流式输出
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, generation_config=temp_config)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text


# =====================================================================
# 5. 启动纯文本界面
# =====================================================================
demo = gr.ChatInterface(
    bot_response, 
    multimodal=False,
    title="CLIMES-Monitor: Uppsala Weather & Heatwave Assistant"
)

if __name__ == "__main__":
    demo.launch()