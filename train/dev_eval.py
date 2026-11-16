"""Dev evaluation: val queries against a val-only corpus, embedder only.
Primary metric: ndcg@10. Writes data/synthetic_val/dev/{tag}.json."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from train.draw_gen_sample import sha256_file

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
DENSE_MANIFEST = Path("data/processed/index/dense_base.manifest.json")
DENSE_NPY = Path("data/processed/index/dense_base.npy")
VAL_Q = Path("data/synthetic_val/queries.jsonl")
OUT_DIR = Path("data/synthetic_val/dev")
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")


def load_corpus():
    val = set(json.loads(SPLITS.read_text(encoding="utf-8"))["val"])
    idx, pids, texts = [], [], []
    with PASSAGES.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            if p["paper_id"] in val:
                idx.append(i)
                pids.append(p["passage_id"])
                texts.append(p["embed_text"])
    return idx, pids, texts


def load_queries():
    rows = [json.loads(l) for l in VAL_Q.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["kept"]]


def embed(model, texts, batch_size):
    out = model.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                       normalize_embeddings=False, show_progress_bar=False)
    return torch.nn.functional.normalize(out.float(), dim=1)


def metrics_from_ranks(r):
    r = np.asarray(r)
    in10 = r <= 10
    return {"ndcg@10": float(np.where(in10, 1.0 / np.log2(r + 1.0), 0.0).mean()),
            "mrr@10": float(np.where(in10, 1.0 / r, 0.0).mean()),
            "recall@10": float(in10.mean()),
            "recall@50": float((r <= 50).mean())}


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path or HF name; default is the base model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--check-cache", action="store_true",
                    help="compare fresh passage vectors with the cached dense_base.npy rows")
    ap.add_argument("--passage-batch", type=int, default=64)
    ap.add_argument("--query-batch", type=int, default=128)
    args = ap.parse_args()

    man = json.loads(DENSE_MANIFEST.read_text(encoding="utf-8"))
    if man["passages_sha256"] != sha256_file(PASSAGES):
        sys.exit("passages.jsonl does not match the index manifest")
    out_path = OUT_DIR / f"{args.tag}.json"
    if out_path.exists():
        sys.exit(f"{out_path} exists; refusing to overwrite")
    model_name = args.model or man["model"]

    idx, pids, texts = load_corpus()
    qs = load_queries()
    pos = {p: i for i, p in enumerate(pids)}
    src = np.array([pos[q["source_passage_id"]] for q in qs])
    print(f"val passages {len(pids)}  kept val queries {len(qs)}  model {model_name}")

    model = SentenceTransformer(model_name, device="cuda")
    model.half()
    model.max_seq_length = man["max_seq_length"]
    P = embed(model, texts, args.passage_batch)
    Q = embed(model, [man["query_prefix"] + q["query"] for q in qs], args.query_batch)

    if args.check_cache:
        cached = torch.from_numpy(np.load(DENSE_NPY, mmap_mode="r")[idx].astype(np.float32)).to("cuda")
        cached = torch.nn.functional.normalize(cached, dim=1)
        cos = (P * cached).sum(1)
        print(f"cache check: min cosine {cos.min().item():.5f}, mean {cos.mean().item():.6f}")

    scores = Q @ P.T
    src_t = torch.from_numpy(src).to("cuda")
    s_src = scores[torch.arange(len(qs), device="cuda"), src_t]
    ranks = (1 + (scores > s_src[:, None]).sum(1)).cpu().numpy()  # strict >: ties count in favour

    voice = np.array([q["voice"] for q in qs])
    res = {"all": metrics_from_ranks(ranks)}
    for v in VOICES:
        res[v] = metrics_from_ranks(ranks[voice == v])
    for k, m in res.items():
        print(f"{k:22s} " + "  ".join(f"{a} {b:.4f}" for a, b in m.items()))
    print(f"median rank {int(np.median(ranks))}  max rank {int(ranks.max())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"tag": args.tag, "model": model_name, "n_passages": len(pids), "n_queries": len(qs),
           "passages_sha256": man["passages_sha256"], "metrics": res,
           "qids": [q["qid"] for q in qs], "ranks": ranks.tolist()}
    out_path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()