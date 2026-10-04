"""Run: python tfidf_model.py (after python prepare_data.py)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA = Path(__file__).resolve().parent / "data"


def main():
    if not (DATA / "processed.json.gz").exists():
        raise SystemExit("Run python prepare_data.py first.")
    with gzip.open(DATA / "processed.json.gz", "rt", encoding="utf-8") as f:
        data = json.load(f)
    train, validation, test = [data["splits"][s] for s in ("train", "validation", "test")]
    labels = sorted(map(int, data["labels"]))
    names = [data["labels"][str(label)] for label in labels]
    y_train, y_val, y_test = [[row["label"] for row in rows] for rows in (train, validation, test)]
    started = time.perf_counter()
    print("Fitting TF-IDF on training text...", flush=True)
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3,
                               max_df=0.95, stop_words="english", sublinear_tf=True, dtype=np.float32)
    vectorizer.fit([row["text"] for row in train])
    x_train = vectorizer.transform([row["text"] for row in train])
    x_val = vectorizer.transform([row["text"] for row in validation])
    best_rank = (-1, -1)
    candidates = []
    for c in (0.1, 1, 10):
        classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=1500, random_state=42)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_val)
        score = f1_score(y_val, pred, average="macro", labels=labels, zero_division=0)
        accuracy = accuracy_score(y_val, pred)
        candidates.append({"C": c, "macro_f1": score, "accuracy": accuracy})
        print(f"C={c}: validation macro-F1={score:.4f}", flush=True)
        if (score, accuracy) > best_rank:
            best_rank, best_c, best_model = (score, accuracy), c, classifier
    # Test text enters only after the validation choice has been made.
    before = time.perf_counter()
    pred = best_model.predict(vectorizer.transform([row["text"] for row in test]))
    test_ms = (time.perf_counter() - before) * 1000 / len(test)
    report = classification_report(y_test, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    majority = max(set(y_train), key=y_train.count)
    results = {"model": "tfidf", "C": best_c, "validation": candidates,
               "accuracy": accuracy_score(y_test, pred),
               "macro_f1": f1_score(y_test, pred, labels=labels, average="macro", zero_division=0),
               "weighted_f1": f1_score(y_test, pred, labels=labels, average="weighted", zero_division=0),
               "majority_accuracy": accuracy_score(y_test, [majority] * len(test)),
               "report": report, "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
               "test_predictions": pred.tolist(), "test_ms_per_document": test_ms,
               "run_seconds": time.perf_counter() - started}
    (DATA / "tfidf_results.json").write_text(json.dumps(results, indent=2))
    print(f"Selected C={best_c}. Test accuracy={results['accuracy']:.2%}, macro-F1={results['macro_f1']:.4f}")
    print(classification_report(y_test, pred, labels=labels, target_names=names, zero_division=0))
    print("Saved data/tfidf_results.json")
    while sys.stdin.isatty():
        text = input("Text to classify (Enter to finish): ").strip()
        if not text:
            break
        vector = vectorizer.transform([text])
        if vector.nnz == 0:
            print("No words match the training vocabulary.")
            continue
        label = best_model.predict(vector)[0]
        print("Predicted category:", data["labels"][str(label)])


if __name__ == "__main__":
    main()
