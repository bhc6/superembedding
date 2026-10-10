"""Prepare a small product-disjoint attribute dataset from a pinned HF shard."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/fashion600")
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.count < 30:
        raise ValueError("use at least 30 images")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = "ashraq/fashion-product-images-small"
    revision = "3859c76db2f6f3d3b9a3863345e3ccdbff75879d"
    filename = "data/train-00000-of-00002-6cff4c59f91661c3.parquet"
    path = hf_hub_download(
        repo, filename, repo_type="dataset", revision=revision,
        local_dir=str(root / "source"),
    )
    columns = ["id", "baseColour", "articleType", "gender", "image"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    colors = {"Black", "White", "Blue", "Red", "Green", "Grey", "Brown", "Pink"}
    articles = {"Tshirts", "Shirts", "Casual Shoes", "Sports Shoes", "Tops", "Handbags"}
    groups = defaultdict(list)
    hashes, ids = set(), set()
    for row in rows:
        if row["baseColour"] not in colors or row["articleType"] not in articles:
            continue
        if row["gender"] not in {"Men", "Women"}:
            continue
        raw = row["image"]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if digest in hashes or row["id"] in ids:
            continue
        hashes.add(digest)
        ids.add(row["id"])
        row["image_sha256"] = digest
        groups[(row["baseColour"], row["articleType"], row["gender"])].append(row)
    rng = random.Random(args.seed)
    # Joint strata with sufficient support; evaluation shares values with training.
    groups = {k: v for k, v in groups.items() if len(v) >= 10}
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < args.count and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < args.count:
                selected.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    if len(selected) != args.count:
        raise ValueError(f"only {len(selected)} eligible unique images")
    used = Counter()
    records = []
    (root / "images").mkdir(exist_ok=True)
    for row in selected:
        labels = {"color": row["baseColour"], "article_type": row["articleType"], "audience": row["gender"]}
        key = tuple(labels.values())
        offset = used[key] % 5
        split = "train" if offset < 3 else ("val" if offset == 3 else "test")
        used[key] += 1
        relative = f"images/{row['id']}.jpg"
        # Keep source image bytes; PIL determines the actual format on load.
        (root / relative).write_bytes(row["image"]["bytes"])
        records.append({"id": str(row["id"]), "image": relative, "split": split,
                        "labels": labels, "image_sha256": row["image_sha256"]})
    (root / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8",
    )
    metadata = {
        "dataset": repo, "revision": revision, "shard": filename, "seed": args.seed,
        "count": len(records), "splits": dict(Counter(r["split"] for r in records)),
        "label_keys": ["color", "article_type", "audience"],
        "directions": ["the dominant color of the product", "the type of clothing or accessory",
                       "the product's intended catalog audience (men or women)"],
        "instruction": "Extract evidence about the pictured product for the specified attribute direction.",
        "protocol": "Joint-attribute stratified product split; exact decoded-image deduplication. "
                    "Seen attribute values/combinations; near-duplicate products not excluded. "
                    "Audience is a catalog label, not a person's gender inference. "
                    "This is a development subset, not the official benchmark split.",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
