import torch
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import pandas as pd

# 1. Setup Professional Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")

# 2. Load the Perception Model (DistilBERT)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Perception Model on: {device}")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)


def get_sentiment(text):
    # UPDATED: Set max_length to 256 to strictly match your training methodology
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping[prediction]


# 3. Initialize the Reasoning Layer (Llama 3 via Ollama)
# UPDATED: Set temperature to 0.1 to prevent hallucinations and enforce professional tone
llm = OllamaLLM(model="llama3", temperature=0.1)

# Professional Agentic Template (Enterprise-Grade)
template = """
You are an elite Customer Experience (CX) Escalation Manager and Brand Protection Officer.
Our deterministic perception model has flagged the following customer review as: {sentiment}.

CUSTOMER REVIEW: "{review_text}"

YOUR DIRECTIVES (RULES OF ENGAGEMENT):
1. **Brand Protection & Compliance:** NEVER admit legal liability, gross negligence, or make explicit financial promises (e.g., do not say "we will give you a full refund"). Instead, use policy-safe escalation language (e.g., "we would like to investigate this to make things right" or "please contact support to process an exchange").
2. **Technical Root Cause Diagnosis:** Do not just say "bad product." Pinpoint the exact operational failure (e.g., Last-Mile Delivery Delay, Manufacturing Defect, UI/UX Confusion, Packaging Damage, or Customer Service Friction).
3. **De-Escalation & Tone:** - If Negative: Be empathetic, de-escalating, and action-oriented.
   - If Neutral: Be inquisitive, seeking specific feedback to turn them into a promoter.
   - If Positive: Build brand loyalty and encourage future engagement.
4. **Multilingual Constraint:** The customer-facing DRAFT must be written in the EXACT SAME LANGUAGE as the original customer review.

OUTPUT FORMAT REQUIREMENTS:
Provide your response exactly in the following structure:

SUMMARY: [1-sentence executive summary of the issue or praise]
WHY: [Precise operational root cause diagnosis based on the text]
SOLUTION: [Internal strategic action steps for the company to fix the root cause]
DRAFT: [A highly professional, compliance-safe, customer-facing response starting with 'Dear Customer', written in the review's original language]
"""

prompt = PromptTemplate.from_template(template)


def run_agentic_pipeline(review):
    # Step 1: Perception (BERT)
    sentiment = get_sentiment(review)
    print(f"\n[PERCEPTION] Sentiment Detected: {sentiment}")

    # Step 2: Reasoning (Llama 3)
    print("[AGENT] Processing reasoning and response...")
    formatted_prompt = prompt.format(sentiment=sentiment, review_text=review)
    response = llm.invoke(formatted_prompt)

    print("\n--- AGENTIC OUTPUT ---")
    print(response)


if __name__ == "__main__":
    print("\n=== LOADING DYNAMIC TEST DATA ===")
    try:
        # Pull 2 random reviews directly from your cleaned dataset
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "master_cleaned_data.csv"))
        sample_reviews = df.sample(2, random_state=None)['clean_text'].tolist()

        print("\n=== TEST 1: RANDOM REVIEW ===")
        run_agentic_pipeline(sample_reviews[0])

        print("\n=== TEST 2: RANDOM REVIEW ===")
        run_agentic_pipeline(sample_reviews[1])

    except Exception as e:
        print(f"Error loading data: {e}")