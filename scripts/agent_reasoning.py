import torch
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 1. Setup Professional Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")

# 2. Load the Perception Model (DistilBERT)
# This uses your fine-tuned model from Phase 2
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Perception Model on: {device}")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping[prediction]


# 3. Initialize the Reasoning Layer (Llama 3 via Ollama)
llm = OllamaLLM(model="llama3")

# Professional Agentic Template (Chain-of-Thought)
template = """
You are an AI Brand Manager Agent. 
Our sentiment model has flagged a review as: {sentiment}

CUSTOMER REVIEW: "{review_text}"

INSTRUCTIONS:
1. Identify the ROOT CAUSE of the sentiment (e.g., Delivery, Product Quality, Customer Service).
2. Provide a brief REASONING for your choice.
3. Draft an APPROPRIATE RESPONSE to the customer.

FORMAT:
ROOT CAUSE: [Category]
REASONING: [1 sentence explaining why]
DRAFT RESPONSE: [Professional reply]
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
    # Test with a difficult logistics review
    test_review = "I love the sneakers, but they arrived with a huge scratch on the side and the delivery driver just left them in the rain."
    run_agentic_pipeline(test_review)