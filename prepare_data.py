"""Run this file first: python prepare_data.py

Python 3.11/3.12. Install the packages once:
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 gensim==4.3.3

Data: https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0
Category names: https://github.com/chartbeat-labs/textacy/blob/main/src/textacy/datasets/supreme_court.py
Metadata license: CC BY-NC 3.0 US; opinion text distributed from FindLaw by textacy.
Implementation prepared with Codex assistance. Record actual team work separately.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATA = Path(__file__).resolve().parent / "data"
URL = "https://github.com/bdewilde/textacy-data/releases/download/supreme_court_py3_v1.0/supreme-court-py3.json.gz"
SHA256 = "2909fc637b44326b7f588371828f6446883dc9c66cf31f6a5805a50248fbb0f4"
LABELS = {1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
          4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
          8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
          11: "Interstate Relations", 12: "Federal Taxation",
          13: "Miscellaneous", 14: "Private Action"}


def main():
    DATA.mkdir(exist_ok=True)
    raw_file = DATA / "raw.json.gz"
    if not raw_file.exists():
        print("Downloading the original corpus (about 111 MB)...", flush=True)
        partial = DATA / "download.tmp"
        try:
            urllib.request.urlretrieve(URL, partial)
            partial.replace(raw_file)
        except Exception as error:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"Download failed. Check your internet connection and retry. {error}")
    digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit("The downloaded file does not match the original dataset checksum. Remove data/raw.json.gz and retry.")
    with gzip.open(raw_file, "rt", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    rows = []
    removed = Counter()
    for index, record in enumerate(raw):
        text = record.get("text") or ""
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if not normalized:
            removed["empty_text"] += 1
            continue
        try:
            label = int(record.get("issue_area"))
        except (TypeError, ValueError):
            label = -1
        if label not in LABELS:
            removed["unknown_label"] += 1
            continue
        rows.append({"id": f"scotus-{index:05d}", "text": text, "label": label,
                     "citation": str(record.get("us_cite_id") or "").strip(),
                     "text_hash": hashlib.sha256(normalized.encode()).hexdigest()})

    # Join records linked by a citation or identical text, including indirect links.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for field in ("citation", "text_hash"):
        seen = {}
        for i, row in enumerate(rows):
            value = row[field]
            if value:
                if value in seen:
                    a, b = find(i), find(seen[value])
                    parent[max(a, b)] = min(a, b)
                else:
                    seen[value] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    cleaned = []
    for group in groups.values():
        if len({row["label"] for row in group}) > 1:
            removed["conflicting_labels"] += len(group)
            continue
        seen = set()
        group_id = min(row["id"] for row in group)
        for row in group:
            if row["text_hash"] in seen:
                removed["duplicates"] += 1
                continue
            seen.add(row["text_hash"])
            row["group"] = group_id
            cleaned.append(row)
    counts = defaultdict(set)
    for row in cleaned:
        counts[row["label"]].add(row["group"])
    excluded = {label: len(group_ids) for label, group_ids in counts.items() if len(group_ids) < 5}
    removed["rare_categories"] = sum(row["label"] in excluded for row in cleaned)
    cleaned = [row for row in cleaned if row["label"] not in excluded]

    y = np.array([row["label"] for row in cleaned])
    group_ids = np.array([row["group"] for row in cleaned])
    x = np.zeros(len(cleaned))
    remaining, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, group_ids))
    train, validation = next(StratifiedGroupKFold(4, shuffle=True, random_state=42).split(
        x[remaining], y[remaining], group_ids[remaining]))
    indices = {"train": remaining[train], "validation": remaining[validation], "test": test}
    splits = {name: [cleaned[int(i)] for i in values] for name, values in indices.items()}
    labels = sorted(set(y.tolist()))
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not ({row["group"] for row in splits[a]} & {row["group"] for row in splits[b]})
    for name, rows in splits.items():
        assert sorted({row["label"] for row in rows}) == labels, f"Missing category in {name}"
    output = {"labels": {str(label): LABELS[label] for label in labels}, "splits": splits,
              "source": URL, "sha256": digest, "seed": 42, "raw_count": len(raw),
              "removed": dict(removed), "excluded_categories": excluded}
    with gzip.open(DATA / "processed.json.gz", "wt", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"Raw opinions: {len(raw)}; retained: {len(cleaned)}; categories: {len(labels)}")
    print("Removed:", dict(removed))
    print("Excluded categories:", {LABELS[k]: v for k, v in excluded.items()})
    print("Split sizes:", {name: len(rows) for name, rows in splits.items()})
    print("Saved data/processed.json.gz. Run any of the three model files next.")


if __name__ == "__main__":
    main()
