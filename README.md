# Email Spam Detection Using Machine Learning

A machine learning project that classifies messages as **spam** or **ham** (legitimate), comparing two classic text-classification algorithms — **Multinomial Naive Bayes** and **Logistic Regression**.

> AI Mini Project — Artificial Intelligence (01CE0525), Semester 5
> **Author:** Aman Raj (Enrollment No. 92400103185, Class 5EC5)
> Department of Computer Engineering, Marwadi University

## Dataset

**SMS Spam Collection** (UCI Machine Learning Repository, Dataset 228)
- 5,572 messages — 4,825 ham (86.59%) and 747 spam (13.41%)
- Link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Citation: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.*

## Methodology

1. Text preprocessing (cleaning the message text)
2. **TF-IDF** vectorization (unigrams + bigrams, English stop words removed — 29,241 features)
3. Stratified train/test split — 4,457 train / 1,115 test (`random_state=42`)
4. Train both models and evaluate on the same test set

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 96.23% | 1.000 | 0.718 | 0.836 |
| Logistic Regression | **96.41%** | 1.000 | **0.732** | **0.845** |

Both models achieved **perfect precision** — not a single legitimate message was flagged as spam. All errors were missed spam messages (false negatives), mainly due to the class imbalance in the dataset. **Logistic Regression performed slightly better overall.**

See `results.txt` for the full metrics and confusion matrices.

## Files in this repository

- `spam_detection.py` — complete, commented Python code (data loading → preprocessing → TF-IDF → both models → metrics + confusion-matrix plots)
- `results.txt` — dataset summary and the full evaluation results
- `Email_Spam_Detection_Research_Paper.pdf` — the full research paper for this project
- `confusion_matrix_multinomial_naive_bayes.png`, `confusion_matrix_logistic_regression.png` — confusion matrix charts

## How to run

```bash
pip install pandas scikit-learn matplotlib
python spam_detection.py
```

The script downloads/loads the SMS Spam Collection dataset, trains both models, and prints accuracy, precision, recall and F1-score for each.

## Tech stack

Python · pandas · scikit-learn · matplotlib
