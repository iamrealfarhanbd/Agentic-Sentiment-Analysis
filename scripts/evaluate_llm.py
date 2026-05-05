import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from agent_reasoning import get_sentiment, llm, prompt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_cleaned_data.csv")
PLOT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(PLOT_DIR, exist_ok=True)

# OBJECTIVE 1-5 LIKERT SCALE LLM-AS-A-JUDGE TEMPLATE
judge_template_likert = """
You are an objective Quality Assurance auditor.
Original Customer Review: "{review}"
AI Generated Output: "{response}"

Grade the AI Generated Output from 1 to 5 based strictly on these criteria:

1. Root Cause: Score 5 if it explicitly states a structured operational failure (e.g., 'WHY: Delivery Delay'). Score 1 if it is a generic apology with no specific diagnosis.
2. Actionability: Score 5 if the response is professional and ready to send to a customer. Score 1 if it is unsafe or unprofessional.
3. Compliance: Score 5 if it makes NO unauthorized financial promises and admits no legal liability. Score 1 if it promises refunds or admits fault.
4. Language Match: Score 5 if the draft matches the original review's language perfectly. Score 1 if it is the wrong language.

Output format:
Root Cause: [score]
Actionability: [score]
Compliance: [score]
Language Match: [score]
"""


def evaluate_likert_pipeline(n_samples=90):
    df = pd.read_csv(DATA_PATH).sample(n_samples, random_state=42)

    # Dictionaries to hold the real scores
    scores = {
        "Root Cause\nPrecision": [],
        "Actionability": [],
        "Compliance\nSafety": [],
        "Language\nMatch": []
    }

    print(f"Generating and Evaluating LLM Drafts for {n_samples} reviews on a 1-5 scale...")

    for idx, review in enumerate(df['clean_text']):
        print(f"Processing review {idx + 1}/{n_samples}...")

        # 1. Generate Anchored Response
        sentiment = get_sentiment(review)
        formatted_prompt = prompt.format(sentiment=sentiment, review_text=review)
        anc_raw = llm.invoke(formatted_prompt)

        # 2. LLM-as-a-Judge Evaluation (1-5 Scale)
        judge_res = llm.invoke(judge_template_likert.format(review=review, response=anc_raw))

        # 3. FIXED: Ultra-forgiving Regex Parsing that ignores spaces and brackets
        rc_match = re.search(r'Root\s*Cause.*?([1-5])', judge_res, re.IGNORECASE)
        act_match = re.search(r'Actionability.*?([1-5])', judge_res, re.IGNORECASE)
        comp_match = re.search(r'Compliance.*?([1-5])', judge_res, re.IGNORECASE)
        lang_match = re.search(r'Language\s*Match.*?([1-5])', judge_res, re.IGNORECASE)

        # Append true scores (If regex somehow fails completely, it prints a warning to the console)
        if not rc_match: print(f"Warning: Could not parse Root Cause from: {judge_res}")

        scores["Root Cause\nPrecision"].append(int(rc_match.group(1)) if rc_match else 1)
        scores["Actionability"].append(int(act_match.group(1)) if act_match else 1)
        scores["Compliance\nSafety"].append(int(comp_match.group(1)) if comp_match else 1)
        scores["Language\nMatch"].append(int(lang_match.group(1)) if lang_match else 1)

    # --- FORMAT DATA FOR SEABORN ---
    plot_df = pd.DataFrame(scores).melt(var_name='Metric', value_name='Score')

    # --- TERMINAL OUTPUT ---
    print("\n" + "=" * 55)
    print("🎓 FINAL DYNAMIC LIKERT RESULTS (MEANS)")
    print("=" * 55)
    # ADDED sort=False TO PREVENT ALPHABETICAL SHUFFLING
    means = plot_df.groupby('Metric', sort=False)['Score'].mean()
    for metric, mean_val in means.items():
        print(f"{metric.replace(chr(10), ' ')}: {mean_val:.2f}/5.00")

    # --- GENERATE EXACT CHART (NO ERROR BARS) ---
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        x='Metric',
        y='Score',
        data=plot_df,
        palette="crest",
        errorbar=None
    )

    plt.title(f'Reasoning Layer Evaluation — Llama-as-Judge (n = {n_samples})', pad=15, fontsize=14)
    plt.ylim(0, 5.5)  # Raised slightly so the 5.00 text doesn't get cut off
    plt.ylabel('Score (1 = poor, 5 = excellent)', fontsize=12)
    plt.xlabel("")

    # Add exact mean text labels on top of bars
    for i, p in enumerate(ax.patches):
        mean_val = means.iloc[i]  # Get the mean value for this metric
        ax.annotate(format(mean_val, '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                    fontsize=12)

    out_path = os.path.join(PLOT_DIR, "11_likert_evaluation_final.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"\nSUCCESS: Saved dynamic Likert plot to {out_path}")

if __name__ == "__main__":
    evaluate_likert_pipeline()