import json
from eval.loaders import load_passage_ids, load_run

valid = load_passage_ids("data/processed/passages.jsonl")
qids = [json.loads(l)["qid"] for l in open("data/eval/questions.jsonl", encoding="utf-8") if l.strip()]
SYSTEMS = {"bm25": 100, "dense_base": 100, "hybrid_base": 100, "hybrid_base_rerank_base": 50}

for s, n in SYSTEMS.items():
    run = load_run(f"runs/{s}.jsonl", valid, system=s)
    assert list(run) == qids, f"{s}: qids differ from questions.jsonl"
    assert all(len(v) == n for v in run.values()), f"{s}: wrong result count"
    sc = []
    for line in open(f"runs/{s}.jsonl", encoding="utf-8"):
        r = [x["score"] for x in json.loads(line)["results"]]
        sc.append(all(a >= b for a, b in zip(r, r[1:])))
    assert all(sc), f"{s}: scores not sorted"
    print("ok", s, len(run), "questions")