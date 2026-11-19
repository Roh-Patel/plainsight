import json, os, sys, argparse

from retrieval.common import load_passages
from retrieval.rerank import Reranker

DEPTH = 50


def read_run(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["qid"]] = [(x["passage_id"], x["score"]) for x in r["results"]]
    return out


def rerank_run(rr, run, qids, pid2idx, idx2pid):
    queries = {}
    with open("data/eval/questions.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["qid"]] = q["question"]
    cands = [[(pid2idx[p], s) for p, s in run[q][:DEPTH]] for q in qids]
    res = rr.rerank([queries[q] for q in qids], cands, depth=DEPTH, batch_size=32)
    return {q: [(idx2pid[i], s) for i, s in r] for q, r in zip(qids, res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default="models/reranker/rr-v1/epoch-1")
    a = ap.parse_args()
    out = "runs/hybrid_ft_embed_rerank_ft.jsonl"
    if not a.check and os.path.exists(out):
        sys.exit(out + " exists, refusing to overwrite")

    pids = [r["passage_id"] for r in load_passages(("passage_id",))]
    pid2idx = {p: i for i, p in enumerate(pids)}

    if a.check:
        rr = Reranker(texts=[r["text"] for r in load_passages(("text",))])
        src, ref = read_run("runs/hybrid_base.jsonl"), read_run("runs/hybrid_base_rerank_base.jsonl")
        qids = sorted(src)
        got = rerank_run(rr, src, qids, pid2idx, pids)
        exact = sum([p for p, _ in got[q]] == [p for p, _ in ref[q]] for q in qids)
        top10 = sum([p for p, _ in got[q]][:10] == [p for p, _ in ref[q]][:10] for q in qids)
        diff = max(abs(s1 - s2) for q in qids for (_, s1), (_, s2) in zip(got[q], ref[q]))
        print("check vs hybrid_base_rerank_base: exact order %d/%d, top-10 order %d/%d, max |logit diff| %.5f"
              % (exact, len(qids), top10, len(qids), diff), rr.stats)
        return

    rr = Reranker(model_name=a.model, texts=[r["text"] for r in load_passages(("text",))])
    src = read_run("runs/hybrid_ft_embed.jsonl")
    qids = sorted(src)
    got = rerank_run(rr, src, qids, pid2idx, pids)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for q in qids:
            f.write(json.dumps({"qid": q, "system": "hybrid_ft_embed_rerank_ft",
                                "results": [{"passage_id": p, "score": s} for p, s in got[q]]}) + "\n")
    print("wrote", out, len(qids), "questions", rr.stats, "model", a.model)


if __name__ == "__main__":
    main()