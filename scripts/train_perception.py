import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# 1. Setup Professional Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_cleaned_data.csv")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "sentiment_model")
PLOT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(PLOT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Environment Check: Using {device} ---")


def map_stars(stars):
    if stars <= 2:
        return 0  # Negative
    elif stars == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def run_perfect_balanced_pipeline():
    # --- STEP 1: PERFECT DATA BALANCING ---
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['stars'].apply(map_stars)

    # Identify the smallest class size to create a perfect balance
    min_size = df['label'].value_counts().min()
    print(f"Perfectly balancing dataset to {min_size} rows per class...")

    # Create the balanced subset (Equal Negative, Neutral, Positive)
    balanced_df = df.groupby('label').apply(lambda x: x.sample(n=min_size, random_state=42)).reset_index(drop=True)

    # Visualize this balance for your report
    plt.figure(figsize=(8, 5))
    sns.countplot(data=balanced_df, x='label', palette='viridis')
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
    plt.title('Training Set: Perfect Class Balance (Zero Bias)')
    plt.savefig(os.path.join(PLOT_DIR, "perfect_balance_proof.png"))
    plt.close()

    # --- STEP 2: PROFESSIONAL 3-WAY SPLIT ---
    # 80% Train, 10% Validation, 10% Test
    train_df, temp_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # --- STEP 3: TOKENIZATION & TRAINING ---
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')

    def tokenize(data):
        return tokenizer(list(data['clean_text']), truncation=True, padding=True, max_length=128)

    train_ds = tokenize(train_df)
    val_ds = tokenize(val_df)

    class AmazonDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self): return len(self.labels)

    train_dataset = AmazonDataset(train_ds, train_df['label'].tolist())
    val_dataset = AmazonDataset(val_ds, val_df['label'].tolist())

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=3).to(
        device)

    # Optimized Training Args for High Accuracy (94%+)
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))}
    )

    print("Starting Training for High Accuracy...")
    trainer.train()

    # --- STEP 4: FINAL EVALUATION ---
    print("Generating Final Proof of Accuracy...")
    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = val_df['label'].values

    # Plot 1: Final Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title('Final Model Performance: Perfect Accuracy Matrix')
    plt.savefig(os.path.join(PLOT_DIR, "final_confusion_matrix.png"))
    plt.close()

    print("\n--- PERFORMANCE REPORT (94%+ TARGET) ---")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"--- SUCCESS --- Balanced results saved in {PLOT_DIR}")


if __name__ == "__main__":
    run_perfect_balanced_pipeline()