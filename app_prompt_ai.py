import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import openai

# 🔐 API Key da OpenAI (usando Streamlit Secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Cache do modelo para performance
@st.cache_resource
def carregar_modelo_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Descrição visual com CLIP
def descrever_imagem(image):
    model, processor = carregar_modelo_clip()
    textos = [
        "a confident man in black shirt",
        "a professional leader",
        "a motivational speaker",
        "a determined man posing",
        "a religious or spiritual figure",
        "a charismatic businessman"
    ]
    inputs = processor(text=textos, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    melhor_texto = textos[probs.argmax()]
    return melhor_texto

# Prompt com GPT
def gerar_prompt(descricao):
    resposta = o
