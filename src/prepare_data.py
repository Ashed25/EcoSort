"""Scan the dataset, remove duplicates, and write stratified train/val/test splits.

Labels are the 3 top-level folders. Subfolders (item types) are kept as a
`subclass` column so the split is stratified per item type and the evaluation
can report per-item accuracy, without training on them as separate classes.
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["nonrecyclable", "organic", "recyclable"]
EXCLUDE = {"nonrecyclable/logo"}  # signs, not waste items
EXTS = {".jpg", ".jpeg", ".png"}


def scan(root):
    rows = []
    for label in CLASSES:
        for dirpath, _, files in os.walk(os.path.join(root, label)):
            rel = os.path.relpath(dirpath, root).replace("\\", "/")
            if rel in EXCLUDE:
                continue
            sub = rel.split("/")[1] if "/" in rel else label
            for f in files:
                if os.path.splitext(f)[1].lower() in EXTS:
                    rows.append({"path": f"{rel}/{f}", "label": label, "subclass": sub})
    return rows


def phash_all(root, rows):
    hashes, keep = [], []
    for r in rows:
        try:
            with Image.open(os.path.join(root, r["path"])) as im:
                h = imagehash.phash(im.convert("RGB"))
            hashes.append(int(str(h), 16))
            keep.append(r)
        except Exception as e:
            print("unreadable, skipped:", r["path"], e)
    return keep, np.array(hashes, dtype=np.uint64)


def dedupe(rows, hashes, max_dist):
    """Union-find over pHash Hamming distance; keep one image per group.
    Groups that mix labels are dropped entirely (ambiguous)."""
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 1024):
        block = hashes[start:start + 1024, None] ^ hashes[None, :]
        dist = np.bitwise_count(block)
        for a, b in zip(*np.nonzero(dist <= max_dist)):
            a += start
            if a < b:
                parent[find(a)] = find(b)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    kept, dup, conflict = [], 0, 0
    for idx in groups.values():
        if len({rows[i]["label"] for i in idx}) > 1:
            conflict += len(idx)
            continue
        kept.append(rows[idx[0]])
        dup += len(idx) - 1
    print(f"duplicates removed: {dup}, cross-label conflicts dropped: {conflict}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--max-dist", type=int, default=4, help="pHash Hamming distance for near-duplicates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = scan(args.root)
    print("scanned:", len(rows))
    rows, hashes = phash_all(args.root, rows)
    rows = dedupe(rows, hashes, args.max_dist)

    strat = [r["subclass"] for r in rows]
    train, rest = train_test_split(rows, test_size=0.30, stratify=strat, random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=[r["subclass"] for r in rest], random_state=args.seed)

    os.makedirs(args.out, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(args.out, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "subclass"])
            w.writeheader()
            w.writerows(split)
        print(f"{name}: {len(split)}", dict(Counter(r["label"] for r in split)))


if __name__ == "__main__":
    main()
