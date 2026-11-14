"""Score hard-negative candidates and positives with bge-reranker-base.

Output: data/synthetic/hn_scored.jsonl, one row per kept query in queries.jsonl order:
  {"qid", "pos_ce_score", "cands": [[point_id, baseline_rank, ce_score], ...]}
Candidates are pool C (hybrid_base ranks 10-50, train papers only, source paper
excluded) in rank order. No score threshold and no near-duplicate filtering here.

Resumable. Never overwrites a complete file: delete it by hand to restart.
  python -m train.score_negatives --limit 200    dry run, writes under scratch_synth\\
  python -m train.score_negatives                full run, about 1.1 hours
  python -m train.score_negatives --check [--limit N]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
QUERIES = Path("data/synthetic/queries.jsonl")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")
RANKS_MANIFEST = Path("data/synthetic/ranks/ranks.manifest.json")
TOP100 = Path("data/synthetic/ranks/hybrid_base_top100.npy")
OUT = Path("data/synthetic/hn_scored.jsonl")
OUT_MANIFEST = Path("data/synthetic/hn_scored.manifest.json")
SCRATCH = Path("scratch_synth")

LO, HI = 10, 50  # 1-based inclusive rank window
CHUNK = 300      # queries per scoring call (about 8k pairs)
BATCH_SIZE = 32  # 128 was 10x slower in the timing run, so it is not used
VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_done(path, kept_qids, repair):
    """Number of complete, correctly ordered rows in path. Optionally cut a torn last line."""
    if not path.exists():
        return 0
    good, offset = 0, 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
            except ValueError:
                break
            if good >= len(kept_qids) or row["qid"] != kept_qids[good]:
                raise SystemExit(f"{path} does not follow the kept query order at row {good}. "
                                 "Delete it by hand to restart.")
            good += 1
            offset += len(line)
    if offset != path.stat().st_size:
        if repair:
            with open(path, "r+b") as f:
                f.truncate(offset)
            print(f"cut a partial last line; {good} complete rows kept")
        else:
            print(f"note: {path} has a partial last line after row {good}")
    return good


def setup(limit):
    manifest = json.loads(RANKS_MANIFEST.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    paper_ids = [r["paper_id"] for r in read_jsonl(PASSAGES)]
    if len(paper_ids) != manifest["passages_lines"]:
        raise SystemExit("passages.jsonl line count does not match the ranks manifest")
    uniq = {p: i for i, p in enumerate(sorted(set(paper_ids)))}
    paper = np.array([uniq[p] for p in paper_ids], dtype=np.int32)
    paper_is_train = np.zeros(len(uniq), dtype=bool)
    for p in splits["train"]:
        paper_is_train[uniq[p]] = True

    queries = list(read_jsonl(QUERIES))
    src_ids = [r["src_point_id"] for r in read_jsonl(SRC_IDS)]
    top = np.load(TOP100)
    if not (len(queries) == len(src_ids) == top.shape[0] == manifest["n"]):
        raise SystemExit("queries, ranks and npy row counts disagree")

    kept = [i for i, q in enumerate(queries) if q["kept"]]
    if limit:
        kept = kept[:limit]

    def pool_c(i):
        raw = top[i, LO - 1:HI]
        valid = raw >= 0
        win = raw[valid]
        rk = np.arange(LO, HI + 1)[valid]
        p = paper[win]
        keep = paper_is_train[p] & (p != paper[src_ids[i]]) & (win != src_ids[i])
        return list(zip(win[keep].tolist(), rk[keep].tolist()))

    return manifest, queries, src_ids, kept, pool_c


def run(limit):
    out = (SCRATCH / f"hn_scored_limit{limit}.jsonl") if limit else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest, queries, src_ids, kept, pool_c = setup(limit)
    kept_qids = [queries[i]["qid"] for i in kept]
    done = load_done(out, kept_qids, repair=True)
    if done == len(kept):
        print(f"{out} is already complete ({done} rows). Nothing to do.")
        return
    print(f"{len(kept)} queries to score, {done} already done, writing {out}")

    from retrieval.rerank import Reranker
    rr = Reranker(batch_size=BATCH_SIZE)
    rr.stats = {"pairs": 0, "passage_truncated": 0, "query_truncated": 0}

    t0 = time.perf_counter()
    pairs_done = 0
    with open(out, "ab") as f:
        for start in range(done, len(kept), CHUNK):
            batch = kept[start:start + CHUNK]
            qs, ps, spans = [], [], []
            for i in batch:
                q = rr._cap_query(queries[i]["query"])
                cands = pool_c(i)
                ids = [src_ids[i]] + [pid for pid, _ in cands]
                spans.append((i, cands))
                qs.extend([q] * len(ids))
                ps.extend(rr.texts[pid] for pid in ids)
            scores = rr.score_pairs(qs, ps, batch_size=BATCH_SIZE, count_truncation=False)
            pos = 0
            for i, cands in spans:
                row = {
                    "qid": queries[i]["qid"],
                    "pos_ce_score": round(scores[pos], 4),
                    "cands": [[pid, rank, round(s, 4)]
                              for (pid, rank), s in zip(cands, scores[pos + 1:pos + 1 + len(cands)])],
                }
                pos += 1 + len(cands)
                f.write((json.dumps(row) + "\n").encode("utf-8"))
            f.flush()
            pairs_done += len(ps)
            n_done = start + len(batch)
            rate = pairs_done / (time.perf_counter() - t0)
            left = (len(kept) - n_done) / max(1, len(batch)) * (len(ps) / rate)
            print(f"{n_done}/{len(kept)} queries, {rate:.0f} pairs/s, about {left / 60:.1f} min left",
                  flush=True)

    print(f"query_truncated this session: {rr.stats['query_truncated']}")
    if not limit:
        OUT_MANIFEST.write_text(json.dumps({
            "n_queries": len(kept),
            "model": "BAAI/bge-reranker-base",
            "input": "text, pair truncation only_second, max_length 512, query cap 128 tokens",
            "pool": f"hybrid_base ranks {LO}-{HI}, train papers only, source paper excluded",
            "order": "queries.jsonl order, kept queries only",
            "batch_size": BATCH_SIZE,
            "chunk_queries": CHUNK,
            "passages_lines": manifest["passages_lines"],
            "passages_sha256": manifest["passages_sha256"],
            "qids_sha256": manifest["qids_sha256"],
            "created": datetime.now().replace(microsecond=0).isoformat(),
        }, indent=1), encoding="utf-8")
        print(f"wrote {OUT_MANIFEST}")


def check(limit):
    out = (SCRATCH / f"hn_scored_limit{limit}.jsonl") if limit else OUT
    _, queries, _, kept, _ = setup(limit)
    kept_qids = [queries[i]["qid"] for i in kept]
    voice_of = {queries[i]["qid"]: queries[i]["voice"] for i in kept}
    done = load_done(out, kept_qids, repair=False)
    print(f"{out}: {done} complete rows of {len(kept)} expected")
    if done != len(kept):
        sys.exit(1)

    stats = {v: {"pos": [], "cand": [], "pos_beats_all": [], "cand_above_pos": []}
             for v in VOICES}
    empty = nan = 0
    for row in read_jsonl(out):
        s = stats[voice_of[row["qid"]]]
        pos = row["pos_ce_score"]
        c = np.array([x[2] for x in row["cands"]], dtype=float)
        nan += int(np.isnan(pos)) + int(np.isnan(c).sum())
        s["pos"].append(pos)
        if len(c) == 0:
            empty += 1
            continue
        s["cand"].extend(c.tolist())
        s["pos_beats_all"].append(pos > c.max())
        s["cand_above_pos"].extend((c > pos).tolist())
    print(f"queries with no candidates: {empty}; NaN scores: {nan}")
    print(f"{'voice':<22}{'pos mean':>9}{'cand mean':>10}{'pos>all':>9}{'cand>pos':>9}")
    for v in VOICES:
        s = stats[v]
        if not s["pos"]:
            continue
        print(f"{v:<22}{np.mean(s['pos']):>9.2f}{np.mean(s['cand']):>10.2f}"
              f"{np.mean(s['pos_beats_all']):>9.3f}{np.mean(s['cand_above_pos']):>9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check(a.limit) if a.check else run(a.limit)