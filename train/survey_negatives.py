"""Hard-negative candidate pool: how many candidates survive each exclusion rule.

CPU only. Reads the frozen files and prints tables. Writes nothing.
Run from the repo root: python -m train.survey_negatives
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
QUERIES = Path("data/synthetic/queries.jsonl")
RANKS = Path("data/synthetic/ranks/ranks.jsonl")
RANKS_MANIFEST = Path("data/synthetic/ranks/ranks.manifest.json")
TOP100 = Path("data/synthetic/ranks/hybrid_base_top100.npy")

LO, HI = 10, 50  # 1-based inclusive rank window
VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
COLS = ["window", "nontrn", "source", "samesec", "adjunit",
        "samepap", "left_A", "left_B", "left_C", "above"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    manifest = json.loads(RANKS_MANIFEST.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))

    passage_ids, paper_ids, sec = [], [], []
    for r in read_jsonl(PASSAGES):
        passage_ids.append(r["passage_id"])
        paper_ids.append(r["paper_id"])
        sec.append(r["section_idx"])
    if len(passage_ids) != manifest["passages_lines"]:
        raise SystemExit(
            f"passages.jsonl has {len(passage_ids)} lines, "
            f"ranks manifest says {manifest['passages_lines']}")

    uniq = {p: i for i, p in enumerate(sorted(set(paper_ids)))}
    paper = np.array([uniq[p] for p in paper_ids], dtype=np.int32)
    sec = np.array(sec, dtype=np.int32)
    paper_is_train = np.zeros(len(uniq), dtype=bool)
    for p in splits["train"]:
        paper_is_train[uniq[p]] = True

    queries = list(read_jsonl(QUERIES))
    ranks = list(read_jsonl(RANKS))
    top = np.load(TOP100)
    n = len(queries)
    if not (len(ranks) == n == top.shape[0] == manifest["n"]) or top.shape[1] != 100:
        raise SystemExit(
            f"size mismatch: queries {n}, ranks {len(ranks)}, "
            f"npy {top.shape}, manifest n {manifest['n']}")
    for i in range(n):
        if queries[i]["qid"] != ranks[i]["qid"]:
            raise SystemExit(f"row {i}: qid mismatch between queries and ranks")
        if passage_ids[ranks[i]["src_point_id"]] != queries[i]["source_passage_id"]:
            raise SystemExit(f"row {i}: src_point_id does not match source_passage_id")

    rows = defaultdict(list)
    for i, q in enumerate(queries):
        if not q["kept"]:
            continue
        src = ranks[i]["src_point_id"]
        raw = top[i, LO - 1:HI]
        valid = raw >= 0
        win = raw[valid]
        rk = np.arange(LO, HI + 1)[valid]
        p = paper[win]
        d_sec = np.abs(sec[win] - sec[src])
        same_paper = p == paper[src]
        is_src = win == src
        same_sec = same_paper & (d_sec == 0)
        adj = same_paper & (d_sec == 1)
        non_train = ~paper_is_train[p]
        sr = ranks[i]["hybrid_base_src_rank"]
        stratum = "in" if (sr is not None and sr <= HI) else "out"
        above = int((rk < sr).sum()) if stratum == "in" else len(win)
        base = non_train | is_src | same_sec
        rows[(q["voice"], stratum)].append((
            len(win), non_train.sum(), is_src.sum(), same_sec.sum(),
            adj.sum(), same_paper.sum(),
            (~base).sum(), (~(base | adj)).sum(), (~(base | same_paper)).sum(),
            above))

    def pick(voice, stratum):
        out = []
        for (v, s), d in rows.items():
            if voice in ("ALL", v) and stratum in ("all", s):
                out.extend(d)
        return np.array(out, dtype=float)

    total = sum(len(d) for d in rows.values())
    print(f"kept queries: {total}. Window: hybrid_base ranks {LO}-{HI}. "
          f"Train papers: {len(splits['train'])}.")
    print("Stratum: in = source at hybrid_base rank <= 50, out = rank > 50 or absent.")
    print("Pool A = window minus non-train papers, the source, and same paper+section_idx.")
    print("Pool B = A minus section_idx +/- 1 in the source paper. Pool C = A minus the whole source paper.")
    print("Columns: nontrn = papers outside train, samesec = same paper and section_idx, adjunit = section_idx +/- 1 in the source paper,")
    print("samepap = same paper, above = window candidates ranked above the source (all of them when out).")

    print("\nMean candidates per query. Rules are counted independently, so they overlap.")
    print(f"{'voice':<22}{'strat':<6}{'n':>7}" + "".join(f"{c:>8}" for c in COLS))
    for voice in VOICES + ["ALL"]:
        for stratum in ("in", "out", "all"):
            a = pick(voice, stratum)
            if len(a) == 0:
                continue
            print(f"{voice:<22}{stratum:<6}{len(a):>7}"
                  + "".join(f"{a[:, j].mean():>8.2f}" for j in range(len(COLS))))

    print("\nShare of queries with 0 candidates left / fewer than 5 left")
    print(f"{'voice':<22}{'strat':<6}{'A':>16}{'B':>16}{'C':>16}")
    for voice in VOICES + ["ALL"]:
        for stratum in ("in", "out", "all"):
            a = pick(voice, stratum)
            if len(a) == 0:
                continue
            cells = "".join(
                f"{(a[:, j] == 0).mean():>8.3f}/{(a[:, j] < 5).mean():<7.3f}"
                for j in (6, 7, 8))
            print(f"{voice:<22}{stratum:<6}{cells}")


if __name__ == "__main__":
    main()