import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Professional File Paths
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
OUTPUTS_DIR = "../outputs"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def clean_text_advanced(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\u00C0-\u017F0-9\s.,!?\']', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def process_single_file(file_name):
    raw_path = os.path.join(RAW_DIR, file_name)
    if not os.path.exists(raw_path):
        print(f"Error: {file_name} not found in {RAW_DIR}")
        return

    df = pd.read_csv(raw_path)
    original_count = len(df)

    print(f"Cleaning {original_count} reviews...")
    df['clean_text'] = df['review_body'].apply(clean_text_advanced)

    # Calculate word count for tokenization justification
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))

    df_filtered = df[df['language'].isin(['en', 'de'])].copy()
    processed_count = len(df_filtered)

    output_path = os.path.join(PROCESSED_DIR, "master_cleaned_data.csv")
    df_filtered.to_csv(output_path, index=False)

    # --- GENERATE METHODOLOGY IMAGES (1 to 3) ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    # IMAGE 1: Language Distribution
    plt.figure(figsize=(8, 6))
    lang_counts = df_filtered['language'].value_counts().reset_index()
    lang_counts.columns = ['Language', 'Count']
    ax = sns.barplot(data=lang_counts, x='Language', y='Count', hue='Language', palette='viridis', legend=False)
    plt.title('Language Distribution (English vs German)', fontsize=14, pad=15)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    plt.savefig(os.path.join(OUTPUTS_DIR, '1_language_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 2: Data Retention Summary
    plt.figure(figsize=(8, 6))
    retention_data = pd.DataFrame(
        {'Stage': ['Original (All)', 'Cleaned (EN/DE)'], 'Count': [original_count, processed_count]})
    ax = sns.barplot(data=retention_data, x='Stage', y='Count', hue='Stage', palette='Blues_d', legend=False)
    plt.title('Data Cleaning Retention', fontsize=14, pad=15)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    plt.savefig(os.path.join(OUTPUTS_DIR, '2_data_retention.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 3: Review Length Distribution (Justifies max_length=256)
    plt.figure(figsize=(9, 6))
    sns.histplot(data=df_filtered, x='word_count', hue='language', bins=50, kde=True, palette='Set2')
    plt.axvline(x=256, color='red', linestyle='--', label='Tokenization Cutoff (256)')
    plt.title('Review Word Count Distribution', fontsize=14, pad=15)
    plt.xlim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, '3_review_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"--- SUCCESS: Images 1-3 saved to {OUTPUTS_DIR} ---")


if __name__ == "__main__":
    process_single_file("test.csv")  # Change to your actual raw file name if needed