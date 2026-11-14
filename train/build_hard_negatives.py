"""Build data/synthetic/hard_negatives.jsonl from the scored pool.

Starts from hn_scored.jsonl (pool C: hybrid_base ranks 10-50, train papers only,
source paper excluded) and removes
  1. candidates whose reranker logit is above T, and
  2. candidates that are the source passage of a different query whose dense_base
     cosine similarity to this query is at least TAU (all 73,767 queries count).
Queries are never dropped for having few negatives.

  python -m train.build_hard_negatives --dry-run   statistics only, writes nothing
  python -m train.build_hard_negatives             writes the file, never overwrites
  python -m train.build_hard_negatives --check     validates the written file
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
QUERIES = Path("data/synthetic/queries.jsonl")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")
RANKS_MANIFEST = Path("data/synthetic/ranks/ranks.manifest.json")
DENSE_MANIFEST = Path("data/processed/index/dense_base.manifest.json")
HN = Path("data/synthetic/hn_scored.jsonl")
OUT = Path("data/synthetic/hard_negatives.jsonl")
OUT_TMP = Path("data/synthetic/hard_negatives.jsonl.tmp")
OUT_MANIFEST = Path("data/synthetic/hard_negatives.manifest.json")

LO, HI = 10, 50
VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
SHORT = ["lay", "conf", "half"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_pairs(queries, src_ids, tau):
    """All (a, b) with cosine(query a, query b) >= tau and different source passages.

    Encoding follows D019: bge-base-en-v1.5, fp16 forward pass, QUERY_PREFIX from the
    dense manifest, vectors L2-normalised in fp32."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = json.loads(DENSE_MANIFEST.read_text(encoding="utf-8"))
    model = SentenceTransformer(dm["model"], device=device)
    if device == "cuda":
        model.half()
    texts = [dm["query_prefix"] + q["query"] for q in queries]
    emb = model.encode(texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
    emb = emb.float()
    emb = emb / emb.norm(dim=1, keepdim=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    src_t = torch.tensor(src_ids, device=emb.device)
    n = emb.shape[0]
    a_parts, b_parts = [], []
    for s in range(0, n, 2048):
        sims = emb[s:s + 2048] @ emb.T
        idx = torch.nonzero(sims >= tau)
        a = idx[:, 0] + s
        b = idx[:, 1]
        ok = src_t[a] != src_t[b]  # also removes self pairs and sibling queries
        a_parts.append(a[ok].cpu().numpy().astype(np.int32))
        b_parts.append(b[ok].cpu().numpy().astype(np.int32))
        del sims, idx
    return np.concatenate(a_parts), np.concatenate(b_parts)


def load_inputs():
    ranks_manifest = json.loads(RANKS_MANIFEST.read_text(encoding="utf-8"))
    passage_ids = [r["passage_id"] for r in read_jsonl(PASSAGES)]
    if len(passage_ids) != ranks_manifest["passages_lines"]:
        raise SystemExit("passages.jsonl line count does not match the ranks manifest")
    queries = list(read_jsonl(QUERIES))
    ranks = list(read_jsonl(SRC_IDS))
    if len(queries) != len(ranks) or any(q["qid"] != r["qid"] for q, r in zip(queries, ranks)):
        raise SystemExit("queries.jsonl and ranks.jsonl are not aligned")
    src_ids = [r["src_point_id"] for r in ranks]
    return ranks_manifest, passage_ids, queries, src_ids


def build(args, pairs_fn=compute_pairs):
    if not args.dry_run and (OUT.exists() or OUT_TMP.exists()):
        raise SystemExit(f"{OUT} already exists. Delete it by hand to rebuild.")
    ranks_manifest, passage_ids, queries, src_ids = load_inputs()
    kept = [i for i, q in enumerate(queries) if q["kept"]]

    a, b = pairs_fn(queries, src_ids, args.tau)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    src_b = np.asarray(src_ids)[b]
    starts = np.searchsorted(a, np.arange(len(queries) + 1))
    n_nb = np.diff(starts)

    kn = n_nb[kept]
    print(f"\nnear-duplicate queries at cosine >= {args.tau}: {len(a)} directed pairs over "
          f"{len(queries)} queries")
    print(f"kept queries with at least one: {(kn > 0).mean():.3f}; mean neighbours {kn.mean():.2f}, "
          f"p99 {np.percentile(kn, 99):.0f}, max {kn.max()}")

    v_of = {v: j for j, v in enumerate(VOICES)}
    n_q = np.zeros(3)
    n_c = np.zeros(3)
    d_thr = np.zeros(3)
    d_nd = np.zeros(3)
    d_any = np.zeros(3)
    final_counts = [[] for _ in range(3)]

    out_f = None if args.dry_run else open(OUT_TMP, "w", encoding="utf-8", newline="\n")
    try:
        for k, row in enumerate(read_jsonl(HN)):
            i = kept[k]
            if row["qid"] != queries[i]["qid"]:
                raise SystemExit(f"hn_scored row {k} is not in kept query order")
            bad = set(src_b[starts[i]:starts[i + 1]].tolist())
            j = v_of[queries[i]["voice"]]
            negs = []
            for pid, rank, score in row["cands"]:
                thr = score > args.t
                nd = pid in bad
                n_c[j] += 1
                d_thr[j] += thr
                d_nd[j] += nd
                d_any[j] += thr or nd
                if not (thr or nd):
                    negs.append({"passage_id": passage_ids[pid], "baseline_rank": rank,
                                 "ce_score": score})
            n_q[j] += 1
            final_counts[j].append(len(negs))
            if out_f:
                out_f.write(json.dumps({"qid": row["qid"], "pos_ce_score": row["pos_ce_score"],
                                        "negatives": negs}) + "\n")
        else:
            if k + 1 != len(kept):
                raise SystemExit(f"hn_scored has {k + 1} rows, expected {len(kept)}")
    finally:
        if out_f:
            out_f.close()

    print(f"\nPool C candidates per voice and what removes them (t = {args.t}, tau = {args.tau}); %")
    print(f"{'':<6}" + "".join(f"{s:>8}" for s in SHORT))
    for label, arr in (("thr", d_thr), ("neardup", d_nd), ("either", d_any)):
        print(f"{label:<8}" + "".join(f"{100 * arr[j] / n_c[j]:>8.2f}" for j in range(3)))
    print("\nNegatives left per query")
    print(f"{'':<12}" + "".join(f"{s:>8}" for s in SHORT))
    print(f"{'mean':<12}" + "".join(f"{np.mean(final_counts[j]):>8.1f}" for j in range(3)))
    for lim, name in ((10, "% < 10"), (5, "% < 5"), (1, "% zero")):
        print(f"{name:<12}" + "".join(
            f"{100 * np.mean(np.array(final_counts[j]) < lim):>8.2f}" for j in range(3)))

    if args.dry_run:
        print("\ndry run: nothing written")
        return
    os.replace(OUT_TMP, OUT)
    OUT_MANIFEST.write_text(json.dumps({
        "n_queries": len(kept),
        "n_negatives": int(n_c.sum() - d_any.sum()),
        "pool": f"hybrid_base ranks {LO}-{HI}, train papers only, source paper excluded",
        "ce_threshold": args.t,
        "ce_rule": "drop a candidate if its bge-reranker-base logit (text input, D024/D026) is above the threshold",
        "near_dup_tau": args.tau,
        "near_dup_rule": ("drop a candidate that is the source passage of another query "
                          "(kept or dropped, different source passage) with dense_base query-query "
                          "cosine >= tau, QUERY_PREFIX applied"),
        "queries_never_dropped_for_few_negatives": True,
        "scored_file_sha256": sha256_file(HN),
        "passages_lines": ranks_manifest["passages_lines"],
        "passages_sha256": ranks_manifest["passages_sha256"],
        "created": datetime.now().replace(microsecond=0).isoformat(),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT} and {OUT_MANIFEST}")


def check():
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    t = manifest["ce_threshold"]
    train = set(json.loads(SPLITS.read_text(encoding="utf-8"))["train"])
    queries = list(read_jsonl(QUERIES))
    kept = [q for q in queries if q["kept"]]
    bad = {"order": 0, "not_train": 0, "same_paper": 0, "is_source": 0, "rank": 0, "score": 0,
           "duplicate": 0}
    n_rows = n_neg = 0
    for k, row in enumerate(read_jsonl(OUT)):
        q = kept[k]
        n_rows += 1
        if row["qid"] != q["qid"]:
            bad["order"] += 1
        src = q["source_passage_id"]
        seen = set()
        for neg in row["negatives"]:
            n_neg += 1
            pid = neg["passage_id"]
            paper = pid.split("::")[0]
            bad["not_train"] += paper not in train
            bad["same_paper"] += paper == src.split("::")[0]
            bad["is_source"] += pid == src
            bad["rank"] += not (LO <= neg["baseline_rank"] <= HI)
            bad["score"] += neg["ce_score"] > t
            bad["duplicate"] += pid in seen
            seen.add(pid)
    print(f"{OUT}: {n_rows} rows (expected {len(kept)}), {n_neg} negatives "
          f"(manifest {manifest['n_negatives']})")
    print("violations: " + ", ".join(f"{k} {v}" for k, v in bad.items()))
    if n_rows != len(kept) or n_neg != manifest["n_negatives"] or any(bad.values()):
        raise SystemExit("CHECK FAILED")
    print("check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.80)
    a = ap.parse_args()
    check() if a.check else build(a)