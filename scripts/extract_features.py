"""Cache genuine OmniJev features, with resume and per-image timing."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Native Windows path: avoid optional Triton/FLA kernels; use PyTorch fallback.
os.environ.setdefault("MSO_FLA", "0")
os.environ.setdefault("MSO_BRANCH_CHUNK", "4")

import torch
from filelock import FileLock
from superembedding import AttributeBasis
from superembedding.omnijev import OmniJevEmbeddingEncoder, load_omnijev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fashion600")
    ap.add_argument("--output", default="runs/fashion600/features")
    ap.add_argument("--adapter", default="models/OmniJev-0.8B-v1.1")
    ap.add_argument("--base", default="models/Qwen3.5-0.8B")
    ap.add_argument("--max-pixels", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=0, help="0 means all; use 3 for an initial GPU smoke run")
    ap.add_argument('--disable-lora', action='store_true', help='Base-network ablation; retains OmniJev special-token embeddings')
    args = ap.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the CUDA virtual environment for feature extraction")
    Path(args.output).resolve().parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(args.output).resolve())+'.lock'):
        extract(args)


def extract(args):
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    basis = AttributeBasis(tuple(metadata["directions"]), metadata["instruction"])
    records = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = {
        "adapter": str(Path(args.adapter).resolve()), "base": str(Path(args.base).resolve()),
        "basis": {"directions": list(basis.directions), "instruction": basis.instruction},
        "max_pixels": args.max_pixels, "dataset_revision": metadata["revision"],
    }
    if args.disable_lora:
        provenance['disable_lora'] = True
    manifest = output / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != provenance:
        raise ValueError("feature cache settings changed; use a new output directory")
    manifest.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if args.limit:
        records = records[:args.limit]
    todo = [r for r in records if not (output / f"{r['id']}.pt").exists()]
    if not todo:
        print("All requested features already cached.")
        return
    start = time.perf_counter()
    print("Loading OmniJev on", torch.cuda.get_device_name(), flush=True)
    mso = load_omnijev(args.adapter, args.base, max_pixels=args.max_pixels)
    encoder = OmniJevEmbeddingEncoder(mso, basis).eval()
    torch.cuda.reset_peak_memory_stats()
    stats = []
    for i, record in enumerate(todo, 1):
        tick = time.perf_counter()
        with encoder.backbone.disable_adapter() if args.disable_lora else nullcontext():
            u, zq = encoder.extract_features({"images": [str(root / record["image"])]})
        torch.cuda.synchronize()
        target = output / f"{record['id']}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save({"u": u.cpu(), "zq": zq.cpu(), "id": record["id"],
                    "image_sha256": record["image_sha256"]}, temporary)
        temporary.replace(target)
        stats.append(time.perf_counter() - tick)
        if i == 1 or i % 10 == 0 or i == len(todo):
            print(json.dumps({"done": i, "total": len(todo), "seconds_per_image": sum(stats)/len(stats),
                              "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
                              "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20,
                              "shape": list(u.shape)}), flush=True)
    report = {"processed": len(stats), "wall_seconds_including_load": time.perf_counter()-start,
              "mean_seconds_per_image": sum(stats)/len(stats),
              "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    (output / ("smoke.json" if args.limit else "extraction.json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
