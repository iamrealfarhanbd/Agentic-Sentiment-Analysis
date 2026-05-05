import streamlit as st
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Agentic Brand Manager", layout="wide")

# Custom CSS for the "Draft Div" and Button Rounding
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

    /* Rounded 5px buttons for all Streamlit elements */
    div.stButton > button {
        border-radius: 5px !important;
    }

    /* The Professional Green Box for the Draft Message */
    .draft-container {
        background-color: #e8f5e9;
        border: 1px solid #2e7d32;
        padding: 20px;
        border-radius: 10px;
        color: #1b5e20;
        font-family: 'sans-serif';
        margin-top: 10px;
        margin-bottom: 10px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# JavaScript Copy Function (Targeted to Draft Text Only)
def render_copy_button(text_to_copy, key_id):
    """Creates a custom HTML/JS button to copy specific text to clipboard"""
    safe_text = text_to_copy.replace("`", "\\`").replace("\n", "\\n").replace("'", "\\'")
    html_code = f"""
    <div id="copy-container-{key_id}">
        <button onclick="copyToClipboard_{key_id}()" style="
            background-color: #2e7d32; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            width: 100%;
            font-weight: bold;
            font-size: 14px;
        ">
            📋 Copy Draft Message Only
        </button>
    </div>
    <script>
    function copyToClipboard_{key_id}() {{
        const text = `{safe_text}`;
        navigator.clipboard.writeText(text).then(() => {{
            alert("Draft message copied to clipboard!");
        }});
    }}
    </script>
    """
    components.html(html_code, height=55)


# --- 2. MODEL ASSETS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")


@st.cache_resource
def load_assets():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    llm = OllamaLLM(model="llama3")
    return tokenizer, model, llm, device


try:
    tokenizer, bert_model, llm, device = load_assets()
except Exception as e:
    st.error(f"Model Load Error: {e}. Check if Phase 2 training finished.")


# --- 3. LOGIC FUNCTIONS ---
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    idx = torch.argmax(outputs.logits, dim=1).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[idx]


def get_agentic_output(text, sentiment):
    template = """
    You are a Brand Manager. Sentiment is {sentiment}. Review: "{text}"
    Provide:
    SUMMARY: (Short summary)
    WHY: (Root cause analysis)
    SOLUTION: (Action steps)
    DRAFT: (Professional response starting with 'Dear Customer')
    """
    prompt = PromptTemplate.from_template(template)
    full_text = llm.invoke(prompt.format(sentiment=sentiment, text=text))
    # Clean special characters automatically
    full_text = full_text.replace("**", "").replace("__", "").strip()
    # Extract Draft part specifically for the copy button
    if "DRAFT:" in full_text:
        analysis_part = full_text.split("DRAFT:")[0].strip()
        draft_part = full_text.split("DRAFT:")[1].strip()
    else:
        analysis_part = full_text
        draft_part = "Draft not found. Please review full AI output."

    return analysis_part, draft_part


# --- 4. DASHBOARD UI ---
st.title("🤖 Agentic Brand Management Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["⚡ Real-Time Analysis", "📊 Sentiment Analytics", "💡 Agentic Solutions"])

# --- TAB 1: REAL-TIME ANALYSIS ---
with tab1:
    col_input, col_output = st.columns([1, 1])
    with col_input:
        st.subheader("Customer Input")
        user_review = st.text_area("Type customer review here:", height=200, key="t1_text")
        run_btn = st.button("Run Analysis", key="t1_run_btn")

    with col_output:
        st.subheader("AI Agent Insights")
        if run_btn and user_review:
            with st.spinner("Processing..."):
                sentiment = get_sentiment(user_review)
                #add color to the metric based on sentiment
                s_color = {"Positive": "green", "Neutral": "orange", "Negative":
                            "red"}[sentiment]
                st.markdown(f"#### AI Perception Result: :{s_color}[{sentiment}]")

                # Logic: Get reasoning and draft separately
                analysis, draft = get_agentic_output(user_review, sentiment)

                st.markdown("### 🧠 Analysis & Solution")
                st.info(analysis)

                st.markdown("### ✉️ Draft Response")
                # Put the draft inside the custom Green Div
                st.markdown(f'<div class="draft-container">{draft}</div>', unsafe_allow_html=True)

                # The Targeted Copy Button
                render_copy_button(draft, "main_tab")
        else:
            st.info("Results will appear here after running analysis.")

# --- TAB 2: SENTIMENT ANALYTICS ---
with tab2:
    st.header("Bulk Sentiment Classification")
    up_file_2 = st.file_uploader("Upload CSV", type="csv", key="up2")
    if up_file_2:
        df2 = pd.read_csv(up_file_2)
        if st.button("Generate Dashboard", key="tab2_btn"):
            with st.spinner("Analyzing Data..."):
                sample_df = df2.head(100).copy()
                sample_df['Sentiment'] = sample_df['review_body'].apply(get_sentiment)
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig, ax = plt.subplots()
                    sample_df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax,
                                                                   colors=['#ff9999', '#66b3ff', '#99ff99'])
                    st.pyplot(fig)
                with c2:
                    st.dataframe(sample_df[['review_body', 'Sentiment']])

# --- TAB 3: AGENTIC SOLUTIONS ---
with tab3:
    st.header("Strategic Response Management")
    up_file_3 = st.file_uploader("Upload Review Data", type="csv", key="up3")

    if up_file_3:
        # Load data and sample it
        df3 = pd.read_csv(up_file_3).head(10)
        for index, row in df3.iterrows():
            with st.expander(f"Review #{index + 1}: {row['review_body'][:60]}..."):
                st.write(f"**Full Review Content:** {row['review_body']}")

                if st.button(f"Generate Agentic Solution for #{index + 1}", key=f"btn_agent_{index}"):
                    with st.spinner("Agent Reasoning..."):
                        # Get Sentiment Label first
                        sent = get_sentiment(row['review_body'])

                        # Show the result label with color
                        s_color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}[sent]
                        st.markdown(f"#### AI Perception Result: :{s_color}[{sent}]")

                        # Get Agent Reasoning and Draft
                        analysis, draft = get_agentic_output(row['review_body'], sent)

                        st.markdown("#### 🧠 Agent Reasoning & Solution")
                        st.info(analysis)

                        st.markdown("#### ✉️ Draft Response")
                        st.markdown(f'<div class="draft-container">{draft}</div>', unsafe_allow_html=True)

                        # Copy Button
                        render_copy_button(draft, f"btn_{index}")