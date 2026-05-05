import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# --- 1. SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_cleaned_data.csv")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "sentiment_model_distilbert_best")
PLOT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(PLOT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Environment Check: Using {device} ---")


def map_stars(stars):
    if stars <= 2: return 0  # Negative
    elif stars == 3: return 1  # Neutral
    else: return 2  # Positive


# --- 2. CUSTOM WEIGHTED TRAINER ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {'accuracy': accuracy_score(labels, preds), 'f1_macro': f1_score(labels, preds, average='macro')}


# --- 3. MAIN PIPELINE ---
def run_perception_pipeline():
    # Load Data
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['stars'].apply(map_stars)

    # TRUE 80/10/10 SPLIT
    print("\n--- Performing 80/10/10 Data Split ---")
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    # --- PART A: BASELINE BENCHMARKING (SVM) ---
    print("\n--- Benchmarking SVM Baseline ---")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
    X_test_tfidf = tfidf.transform(test_df['clean_text'])

    svm = LinearSVC(random_state=42, class_weight='balanced')
    svm.fit(X_train_tfidf, train_df['label'])
    svm_preds = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(test_df['label'], svm_preds)
    svm_f1 = f1_score(test_df['label'], svm_preds, average='macro')

    # --- PART B: FINE-TUNED DISTILBERT ---
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize(data):
        return tokenizer(list(data['clean_text']), truncation=True, padding=True, max_length=256)

    train_enc = tokenize(train_df)
    val_enc = tokenize(val_df)
    test_enc = tokenize(test_df)

    class AmazonDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self): return len(self.labels)

    train_ds = AmazonDataset(train_enc, train_df['label'].tolist())
    val_ds = AmazonDataset(val_enc, val_df['label'].tolist())
    test_ds = AmazonDataset(test_enc, test_df['label'].tolist())

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR, num_train_epochs=5, per_device_train_batch_size=16,
        eval_strategy="epoch", save_strategy="epoch", learning_rate=2e-5, weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", report_to="none", logging_strategy="epoch"
    )

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                              compute_metrics=compute_metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.class_weights = weights_tensor
    print("\n--- Training Optimized DistilBERT ---")
    trainer.train()

    # Final Evaluation Data on UNSEEN TEST SET
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = test_df['label'].values
    logits = preds_output.predictions
    bert_acc = accuracy_score(y_true, y_pred)
    bert_f1 = f1_score(y_true, y_pred, average='macro')

    # --- PART C: GENERATE RESULTS IMAGES (4 to 10) ---
    print("\n--- Generating High-Resolution Plots ---")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    # IMAGE 4: Training Class Balance
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=train_df, x='label', palette='viridis')
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
    plt.title('Training Set Class Balance', pad=15)
    for p in ax.patches: ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    plt.savefig(os.path.join(PLOT_DIR, "4_training_class_balance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 5: Architecture Comparison
    plt.figure(figsize=(9, 6))
    comp_df = pd.DataFrame(
        {'Metric': ['Accuracy', 'Accuracy', 'Macro F1', 'Macro F1'], 'Score': [svm_acc, bert_acc, svm_f1, bert_f1],
         'Model': ['SVM (Baseline)', 'DistilBERT', 'SVM (Baseline)', 'DistilBERT']})
    ax = sns.barplot(x='Metric', y='Score', hue='Model', data=comp_df, palette='magma')
    plt.title('Architecture Validation (SVM vs DistilBERT)', pad=15)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        if p.get_height() > 0: ax.annotate(format(p.get_height(), '.3f'),
                                           (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                                           xytext=(0, 8), textcoords='offset points')
    plt.savefig(os.path.join(PLOT_DIR, "5_architecture_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 6: Training vs Validation Loss Curve
    log_history = trainer.state.log_history
    train_loss = [x['loss'] for x in log_history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x]
    if train_loss and eval_loss:
        plt.figure(figsize=(9, 6))
        epochs = range(1, len(eval_loss) + 1)
        plt.plot(epochs, train_loss[:len(eval_loss)], 'b-o', label='Training Loss', linewidth=2)
        plt.plot(epochs, eval_loss, 'r-s', label='Validation Loss', linewidth=2)
        plt.title('Model Convergence (Loss Curve)', pad=15)
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, "6_loss_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # IMAGE 7: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'], annot_kws={"size": 14})
    plt.title('DistilBERT Confusion Matrix', pad=15)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(PLOT_DIR, "7_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 8: ROC-AUC Multi-Class Curve
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(9, 6))
    colors = ['red', 'blue', 'green']
    classes = ['Negative', 'Neutral', 'Positive']
    for i, color in zip(range(3), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], logits[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC {classes[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves', pad=15)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOT_DIR, "8_roc_auc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # IMAGE 9: Language Split Performance
    en_f1 = f1_score(y_true[test_df['language'] == 'en'], y_pred[test_df['language'] == 'en'], average='macro')
    de_f1 = f1_score(y_true[test_df['language'] == 'de'], y_pred[test_df['language'] == 'de'], average='macro')
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=['English Subset', 'German Subset'], y=[en_f1, de_f1], palette='coolwarm')
    plt.title('Morphological Impact (Macro F1 by Language)', pad=15)
    plt.ylim(0, 1.0)
    for p in ax.patches: ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    plt.savefig(os.path.join(PLOT_DIR, "9_language_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- PART D: TERMINAL OUTPUT (TABLES I, II, III) RESTORED ---
    print("\n" + "=" * 60)
    print(f"{'TABLE I: OVERALL CLASSIFICATION METRICS (N=' + str(len(y_true)) + ')':^60}")
    print("=" * 60)
    print(f"Overall Accuracy: {bert_acc:.4f}")
    print(f"Macro F1-Score:  {bert_f1:.4f}")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

    # English Subset (Using test_df)
    en_mask = test_df['language'] == 'en'
    print("\n" + "=" * 60)
    print(f"{'TABLE II: ENGLISH SUBSET PERFORMANCE (N=' + str(en_mask.sum()) + ')':^60}")
    print("=" * 60)
    if en_mask.sum() > 0:
        print(classification_report(y_true[en_mask], y_pred[en_mask], target_names=['Negative', 'Neutral', 'Positive']))
    else:
        print("No English data in test set.")

    # German Subset (Using test_df)
    de_mask = test_df['language'] == 'de'
    print("\n" + "=" * 60)
    print(f"{'TABLE III: GERMAN SUBSET PERFORMANCE (N=' + str(de_mask.sum()) + ')':^60}")
    print("=" * 60)
    if de_mask.sum() > 0:
        print(classification_report(y_true[de_mask], y_pred[de_mask], target_names=['Negative', 'Neutral', 'Positive']))
    else:
        print("No German data in test set.")

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"\n--- SUCCESS: Images 4-10 saved to {PLOT_DIR} ---")


if __name__ == "__main__":
    run_perception_pipeline()