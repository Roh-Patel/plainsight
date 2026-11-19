import hashlib, json, os, sys
from datetime import datetime

import anthropic

from train.judge_prompt import PROMPT_SHA256, build_prompt

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4000
D = "scratch_synth"
ITEMS = os.path.join(D, "judge_val_items.jsonl")
REQS = os.path.join(D, "judge_val_requests.jsonl")
REC = os.path.join(D, "judge_val_batch.json")


def sha_passages():
    h = hashlib.sha256()
    with open("data/processed/passages.jsonl", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def submit():
    if os.path.exists(REC):
        sys.exit(REC + " exists, refusing to submit again")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    sha = sha_passages()
    with open("data/processed/index/dense_base.manifest.json", encoding="utf-8") as f:
        if json.load(f)["passages_sha256"] != sha:
            sys.exit("passages.jsonl does not match the dense_base manifest")
    with open(ITEMS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    need = {i for r in rows for i in r["point_ids"]}
    text = {}
    with open("data/processed/passages.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                text[i] = json.loads(line)["text"]
    qs = {}
    qneed = {r["qid"] for r in rows}
    with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] in qneed:
                qs[r["qid"]] = r["query"]
    reqs = []
    for n, r in enumerate(rows):
        prompt = build_prompt(qs[r["qid"]], [text[i] for i in r["point_ids"]])
        reqs.append({"custom_id": "v%04d" % n,
                     "params": {"model": MODEL, "max_tokens": MAX_TOKENS,
                                "messages": [{"role": "user", "content": prompt}]}})
    with open(REQS, "w", encoding="utf-8", newline="\n") as f:
        for q in reqs:
            f.write(json.dumps(q) + "\n")
    size = sum(len(json.dumps(q).encode("utf-8")) for q in reqs)
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    rec = {"batch_id": batch.id, "n_requests": len(reqs), "model": MODEL,
           "max_tokens": MAX_TOKENS, "prompt_sha256": PROMPT_SHA256,
           "passages_sha256": sha, "created": datetime.now().isoformat(timespec="seconds")}
    with open(REC, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("submitted", batch.id, len(reqs), "requests,", round(size / 1024 ** 2, 1), "MB")
    print("prompt_sha256", PROMPT_SHA256)


def status():
    with open(REC, encoding="utf-8") as f:
        rec = json.load(f)
    b = anthropic.Anthropic().messages.batches.retrieve(rec["batch_id"])
    print(b.id, b.processing_status, b.request_counts)


if __name__ == "__main__":
    {"submit": submit, "status": status}[sys.argv[1]]()