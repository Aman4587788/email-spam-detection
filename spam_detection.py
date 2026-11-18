"""
Email Spam Detection Using Machine Learning
=============================================

Mini Project - Artificial Intelligence (Semester 5)

Author : Aman Raj
Enroll. : 92400103185
Class  : 5EC5, Department of Computer Engineering, Marwadi University

This script classifies SMS messages as "spam" or "ham" (legitimate)
using two classical machine learning algorithms:

    1. Multinomial Naive Bayes
    2. Logistic Regression

Both models are trained on TF-IDF features extracted from the
SMS Spam Collection dataset (UCI Machine Learning Repository,
dataset 228) and compared on accuracy, precision, recall and F1-score.

Dataset source:
    https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Usage:
    python spam_detection.py [path_to_SMSSpamCollection]
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "SMSSpamCollection"
TEST_SIZE = 0.20          # 80% train / 20% test
RANDOM_STATE = 42         # fixed seed so the results are reproducible


# ----------------------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------------------
def load_dataset(path):
    """Load the SMS Spam Collection file (tab separated: label, message)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    # Convert the text labels to numbers: spam = 1, ham = 0
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ----------------------------------------------------------------------
# 2. Basic text preprocessing
# ----------------------------------------------------------------------
def clean_text(text):
    """
    Very light preprocessing:
      * convert to lowercase
      * keep only letters and spaces (remove digits / punctuation)
      * collapse extra whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# 3. Train one model and evaluate it
# ----------------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model, predict on the test set and return its metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        # pos_label=1 -> metrics are reported for the "spam" class
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def plot_confusion_matrix(cm, title, filename):
    """Save a simple confusion-matrix heatmap as an image."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Ham", "Spam"])
    ax.set_yticks([0, 1], labels=["Ham", "Spam"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    # ---- Load data --------------------------------------------------
    df = load_dataset(DATASET_PATH)
    total = len(df)
    spam_count = int((df["label"] == "spam").sum())
    ham_count = int((df["label"] == "ham").sum())

    print("=" * 64)
    print("EMAIL / SMS SPAM DETECTION USING MACHINE LEARNING")
    print("=" * 64)
    print(f"Dataset            : SMS Spam Collection (UCI, dataset 228)")
    print(f"Total messages     : {total}")
    print(f"Ham  (legitimate)  : {ham_count} ({ham_count / total * 100:.2f}%)")
    print(f"Spam               : {spam_count} ({spam_count / total * 100:.2f}%)")

    # ---- Preprocess -------------------------------------------------
    df["clean_message"] = df["message"].apply(clean_text)

    # ---- Train / test split (stratified keeps the class ratio) -----
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_num"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )
    print(f"\nTrain / test split : {len(X_train_raw)} train / "
          f"{len(X_test_raw)} test (stratified, random_state={RANDOM_STATE})")

    # ---- TF-IDF vectorization --------------------------------------
    # English stop words are removed; unigrams + bigrams are used so the
    # models can also learn short phrases such as "free entry" or "call now".
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"TF-IDF vocabulary  : {X_train.shape[1]} features")

    # ---- Train and evaluate both models -----------------------------
    models = [
        ("Multinomial Naive Bayes", MultinomialNB()),
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, solver="liblinear")),
    ]

    results = []
    for name, model in models:
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(f"Accuracy           : {metrics['accuracy']:.4f}")
        print(f"Precision (spam)   : {metrics['precision']:.4f}")
        print(f"Recall (spam)      : {metrics['recall']:.4f}")
        print(f"F1-score (spam)    : {metrics['f1']:.4f}")
        cm = metrics["confusion_matrix"]
        print("Confusion matrix   : [[TN, FP], [FN, TP]] = "
              f"[[{cm[0, 0]}, {cm[0, 1]}], [{cm[1, 0]}, {cm[1, 1]}]]")

        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(cm, f"Confusion Matrix - {name}",
                              f"confusion_matrix_{safe_name}.png")

    # ---- Summary ----------------------------------------------------
    best = max(results, key=lambda m: m["f1"])
    print("\n" + "=" * 64)
    print(f"Best model by F1-score: {best['model']} "
          f"(F1 = {best['f1']:.4f}, Accuracy = {best['accuracy']:.4f})")
    print("=" * 64)

    # ---- Save a machine-readable summary for the report -------------
    with open("results_summary.csv", "w") as fh:
        fh.write("model,accuracy,precision,recall,f1\n")
        for m in results:
            fh.write(f"{m['model']},{m['accuracy']:.4f},{m['precision']:.4f},"
                     f"{m['recall']:.4f},{m['f1']:.4f}\n")


if __name__ == "__main__":
    main()
