"""Run: python doc2vec_model.py (after python prepare_data.py).

PV-DBOW learns 100-dimensional vectors; Logistic Regression classifies them.
Doc2Vec training and inference can take tens of minutes on a CPU.
Gensim caps each pass at 10,000 retained tokens after filtering/downsampling.
"""
import os
import sys
# Set hashing before Python starts, for repeatable Doc2Vec initialization.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "42":
    os.environ["PYTHONHASHSEED"] = "42"
    os.execv(sys.executable, [sys.executable, *sys.argv])
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA = Path(__file__).resolve().parent / "data"


def tokens(text):
    return re.findall(r"[a-z]{2,}", text.lower())


def stable_hash(value):
    if isinstance(value, str):
        value = value.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def infer(model, texts):
    vectors = []
    for text in texts:
        words = tokens(text)
        model.random.seed(stable_hash(" ".join(words)) ^ 42)
        vectors.append(model.infer_vector(words, epochs=30))
    return np.asarray(vectors, dtype=np.float32)


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
    print("Training Doc2Vec for 30 epochs...", flush=True)
    documents = [TaggedDocument(tokens(row["text"]), [i]) for i, row in enumerate(train)]
    model = Doc2Vec(vector_size=100, dm=0, dbow_words=0, epochs=30, min_count=5,
                    negative=5, sample=1e-4, workers=1, seed=42,
                    hashfxn=stable_hash, max_vocab_size=100000)
    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=30)
    # Infer every partition in the same way, including training documents.
    print("Inferring training and validation vectors...", flush=True)
    x_train = infer(model, [row["text"] for row in train])
    x_val = infer(model, [row["text"] for row in validation])
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
    print("Evaluating the selected classifier on test vectors...", flush=True)
    before = time.perf_counter()
    pred = best_model.predict(infer(model, [row["text"] for row in test]))
    test_ms = (time.perf_counter() - before) * 1000 / len(test)
    majority = max(set(y_train), key=y_train.count)
    results = {"model": "doc2vec", "C": best_c, "validation": candidates,
               "accuracy": accuracy_score(y_test, pred),
               "macro_f1": f1_score(y_test, pred, labels=labels, average="macro", zero_division=0),
               "weighted_f1": f1_score(y_test, pred, labels=labels, average="weighted", zero_division=0),
               "majority_accuracy": accuracy_score(y_test, [majority] * len(test)),
               "report": classification_report(y_test, pred, labels=labels, target_names=names, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
               "test_predictions": pred.tolist(), "test_ms_per_document": test_ms,
               "run_seconds": time.perf_counter() - started}
    (DATA / "doc2vec_results.json").write_text(json.dumps(results, indent=2))
    print(f"Selected C={best_c}. Test accuracy={results['accuracy']:.2%}, macro-F1={results['macro_f1']:.4f}")
    print(classification_report(y_test, pred, labels=labels, target_names=names, zero_division=0))
    print("Saved data/doc2vec_results.json")
    while sys.stdin.isatty():
        text = input("Text to classify (Enter to finish): ").strip()
        if not text:
            break
        if not any(word in model.wv.key_to_index for word in tokens(text)):
            print("No words match the training vocabulary.")
            continue
        label = best_model.predict(infer(model, [text]))[0]
        print("Predicted category:", data["labels"][str(label)])


if __name__ == "__main__":
    main()
