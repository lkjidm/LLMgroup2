from transformers import AutoModelForMultimodalLM, AutoProcessor, logging, AutoTokenizer,GenerationConfig,TextIteratorStreamer
import json
import torch
from PIL import Image
import gradio as gr
from threading import Thread


# avoid having too much output from the transformers library
logging.set_verbosity_warning()

# fairly small but good model
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

quantization_config = None

# We're using the `AutoModelForMultimodalLM` class to enable multimodal generation
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"  # automatically uses right device e.g., GPU if available
)

# We're using the `AutoProcessor` class to enable multimodal generation
# Normally, you can use AutoTokenizer
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
temp_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=512
   
)

SYSTEM_PROMPT = (
    "You are a weather expert for Uppsala, Sweden. "
    "Your default behavior is to talk about weather and stir any conversation back to weather. "
    "Your main mission is to give advice and predictions on weather, climate, and heatwaves. "
    "You should only talk about weather and heatwaves in Uppsala, Sweden. "
    "You can ask user's identity and location inside Uppsala for more precise advice"
    "Be helpful, informative, and stay in character."
    "Use natural language and avoid using JSON format in your responses. "
)


# def bot_response(message, history):
#     user_text = message.get("text", "").strip()
#     uploaded_files = message.get("files", [])
    
#     current_image = None
#     json_info = None
    
#     if uploaded_files:
#         img_path = uploaded_files[0] if isinstance(uploaded_files[0], str) else uploaded_files[0]["path"]
#         current_image = Image.open(img_path)

    
#     conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    
#     current_content = []
#     if current_image:
#         current_content.append({"type": "image"})
        
#     if json_info:
       
#         if user_text:
#             instruction = f"The user asked: '{user_text}'. Answer the user's question using this background metadata about the mushroom image: {json_info}"
#         else:
#             instruction = f"The user uploaded an image without a question. Provide a concise summary of the mushroom in the image using this metadata: {json_info}"
#         current_content.append({"type": "text", "text": instruction})
#     else:
#         current_content.append({"type": "text", "text": user_text})

#     conversation.append({"role": "user", "content": current_content})

    
#     text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(
#         text=[text_prompt], 
#         images=[current_image] if current_image else None, 
#         return_tensors="pt"
#     ).to(model.device)

#     streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
#     generation_kwargs = dict(**inputs, streamer=streamer, generation_config=temp_config)

#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()

    
#     partial_text = ""
#     for new_text in streamer:
#         partial_text += new_text
#         yield partial_text

def bot_response(message, history):
    # In text-only mode, message is the user input string
    user_text = message.strip() if isinstance(message, str) else message.get("text", "").strip()

    # Build the conversation context
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]

    # Apply the chat template to create text_prompt
    text_prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, generation_config=temp_config)

    # Start generation in a separate thread for streaming output
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

demo = gr.ChatInterface(bot_response, multimodal=False,
    title="CLIMES-Monitor: Uppsala Weather & Heatwave Assistant")
if __name__ == "__main__":
    demo.launch()


