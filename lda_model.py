"""Run: python lda_model.py (after python prepare_data.py).

Counts feed LDA; topic proportions feed Logistic Regression.
This count-based LDA input is an adaptation of the reference study.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import json
from pathlib import Path
import sys
import time

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
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
    best_rank = (-1, -1)
    candidates = []
    for topics in (20, 40):
        print(f"Fitting LDA with {topics} topics...", flush=True)
        vectorizer = CountVectorizer(max_features=15000, min_df=5, max_df=0.95, stop_words="english")
        counts = vectorizer.fit_transform([row["text"] for row in train])
        lda = LatentDirichletAllocation(n_components=topics, learning_method="online", max_iter=5,
                                       batch_size=128, max_doc_update_iter=30, random_state=42,
                                       n_jobs=1, evaluate_every=-1)
        lda.fit(counts)
        x_train = lda.transform(counts)
        x_val = lda.transform(vectorizer.transform([row["text"] for row in validation]))
        for c in (0.1, 1, 10):
            classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=1500, random_state=42)
            classifier.fit(x_train, y_train)
            pred = classifier.predict(x_val)
            score = f1_score(y_val, pred, average="macro", labels=labels, zero_division=0)
            accuracy = accuracy_score(y_val, pred)
            candidates.append({"topics": topics, "C": c, "macro_f1": score, "accuracy": accuracy})
            print(f"Topics={topics}, C={c}: validation macro-F1={score:.4f}", flush=True)
            if (score, accuracy) > best_rank:
                best_rank, best_c, best_topics = (score, accuracy), c, topics
                best_vectorizer, best_lda, best_model = vectorizer, lda, classifier
    before = time.perf_counter()
    x_test = best_lda.transform(best_vectorizer.transform([row["text"] for row in test]))
    pred = best_model.predict(x_test)
    test_ms = (time.perf_counter() - before) * 1000 / len(test)
    majority = max(set(y_train), key=y_train.count)
    results = {"model": "lda", "topics": best_topics, "C": best_c, "validation": candidates,
               "accuracy": accuracy_score(y_test, pred),
               "macro_f1": f1_score(y_test, pred, labels=labels, average="macro", zero_division=0),
               "weighted_f1": f1_score(y_test, pred, labels=labels, average="weighted", zero_division=0),
               "majority_accuracy": accuracy_score(y_test, [majority] * len(test)),
               "report": classification_report(y_test, pred, labels=labels, target_names=names, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
               "test_predictions": pred.tolist(), "test_ms_per_document": test_ms,
               "run_seconds": time.perf_counter() - started}
    (DATA / "lda_results.json").write_text(json.dumps(results, indent=2))
    print(f"Selected {best_topics} topics, C={best_c}. Test accuracy={results['accuracy']:.2%}, macro-F1={results['macro_f1']:.4f}")
    print(classification_report(y_test, pred, labels=labels, target_names=names, zero_division=0))
    print("Saved data/lda_results.json")
    while sys.stdin.isatty():
        text = input("Text to classify (Enter to finish): ").strip()
        if not text:
            break
        counts = best_vectorizer.transform([text])
        if counts.nnz == 0:
            print("No words match the training vocabulary.")
            continue
        label = best_model.predict(best_lda.transform(counts))[0]
        print("Predicted category:", data["labels"][str(label)])


if __name__ == "__main__":
    main()
