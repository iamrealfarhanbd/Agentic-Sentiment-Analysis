import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Professional File Paths
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
OUTPUTS_DIR = "../outputs"

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def clean_text_advanced(text):
    """
    Advanced text cleaning: Removes HTML, URLs, non-alphanumeric noise, and normalizes spacing.
    """
    if not isinstance(text, str):
        return ""
    # 1. Remove HTML tags using regex
    text = re.sub(r'<[^>]*>', '', text)
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 3. Remove non-alphanumeric noise but keep basic punctuation and accents
    text = re.sub(r'[^a-zA-Z\u00C0-\u017F0-9\s.,!?\']', '', text)
    # 4. Normalize whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def process_single_file(file_name):
    raw_path = os.path.join(RAW_DIR, file_name)
    if not os.path.exists(raw_path):
        print(f"Error: {file_name} not found in {RAW_DIR}")
        return

    # Load Data
    df = pd.read_csv(raw_path)
    original_count = len(df)

    # 1. Text Cleaning
    print(f"Cleaning {original_count} reviews...")
    df['clean_text'] = df['review_body'].apply(clean_text_advanced)

    # 2. Filter for target languages (English and German)
    df_filtered = df[df['language'].isin(['en', 'de'])].copy()
    processed_count = len(df_filtered)

    # 3. Save Master Cleaned File
    output_path = os.path.join(PROCESSED_DIR, "master_cleaned_data.csv")
    df_filtered.to_csv(output_path, index=False)

    # --- Visualization Part (Now with 'hue' fixed) ---
    sns.set_theme(style="whitegrid")

    # Visualization 1: Language Distribution
    plt.figure(figsize=(10, 6))
    lang_counts = df_filtered['language'].value_counts().reset_index()
    lang_counts.columns = ['Language', 'Count']
    sns.barplot(data=lang_counts, x='Language', y='Count', hue='Language', palette='viridis', legend=False)
    plt.title('Language Distribution: English vs German', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUTS_DIR, 'language_distribution.png'))
    plt.close()

    # Visualization 2: Data Retention Summary
    plt.figure(figsize=(8, 6))
    retention_data = pd.DataFrame({
        'Stage': ['Original (All Langs)', 'Cleaned (EN/DE Only)'],
        'Count': [original_count, processed_count]
    })
    sns.barplot(data=retention_data, x='Stage', y='Count', hue='Stage', palette='Blues_d', legend=False)
    plt.title('Data Cleaning Retention Summary', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUTS_DIR, 'data_retention.png'))
    plt.close()

    print(f"\n--- SUCCESS ---")
    print(f"Master cleaned data saved to: {output_path}")
    print(f"Language and Retention plots saved to: {OUTPUTS_DIR}")
    print(f"Total reviews preserved: {processed_count}")


if __name__ == "__main__":
    process_single_file("test.csv")