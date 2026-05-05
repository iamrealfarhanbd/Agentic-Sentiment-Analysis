import pandas as pd
import torch
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# --- 1. SETUP PATHS & MODELS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model_distilbert_best")

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# Llama 3 with low temperature to match your methodology
llm = OllamaLLM(model="llama3", temperature=0.1)

# --- 2. PROMPT TEMPLATES ---
# The Unanchored Baseline (Zero-Shot)
zero_shot_template = """
You are a Customer Service AI. Read this review: "{review_text}"
1. Determine the sentiment (Negative, Neutral, Positive).
2. What is the root cause?
3. Draft a response.
"""
zero_shot_prompt = PromptTemplate.from_template(zero_shot_template)

# Your Proposed Architecture (Anchored)
anchored_template = """
You are an elite Customer Experience Escalation Manager.
Our deterministic model flagged this review as: {sentiment}.

CUSTOMER REVIEW: "{review_text}"

1. Diagnose the exact operational root cause (e.g., Delivery Delay, Quality).
2. Draft a professional customer-facing response in the exact same language.
"""
anchored_prompt = PromptTemplate.from_template(anchored_template)

def get_bert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad(): outputs = model(**inputs)
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[torch.argmax(outputs.logits, dim=1).item()]

def run_ablation_study():
    # Load data and select 4 diverse examples using a fixed seed for reproducibility
    df = pd.read_csv(DATA_PATH)
    sample_reviews = df.sample(4, random_state=42)['clean_text'].tolist()

    print("\n" + "="*80)
    print(" 🧪 ABLATION STUDY & CASE STUDY GENERATION ".center(80))
    print("="*80)

    for i, review in enumerate(sample_reviews):
        print(f"\n--- 📝 REVIEW {i+1} ---")
        print(f"TEXT: {review}")

        # 1. Zero-Shot Run
        print("\n[baseline] Running Zero-Shot Llama 3...")
        zero_shot_res = llm.invoke(zero_shot_prompt.format(review_text=review))

        # 2. Anchored Run
        print("[proposed] Running BERT-Anchored Llama 3...")
        bert_sent = get_bert_sentiment(review)
        anchored_res = llm.invoke(anchored_prompt.format(sentiment=bert_sent, review_text=review))

        print(f"\n> BERT SENTIMENT ANCHOR: {bert_sent}")
        print("\n> ZERO-SHOT LLM OUTPUT:")
        print(zero_shot_res.strip()[:250] + "... [truncated]")

        print("\n> ANCHORED LLM OUTPUT:")
        print(anchored_res.strip()[:250] + "... [truncated]")
        print("-" * 80)

if __name__ == "__main__":
    run_ablation_study()