import hashlib, json, re, sys

RUBRIC = ("Give a score of 1 if the passage alone FULLY answers the query, or 0 if it does not "
          "fully answer the query. Assume the person asking the query has the same knowledge "
          "an average, ordinary person would have.")

HEAD = ("You are grading search results. For each numbered passage below, apply this rule "
        "independently of the other passages:\n\n" + RUBRIC + "\n\nQUERY: ")
MID = "\n\nPASSAGES:\n"
TAIL = ('\n\nReturn only a JSON object mapping each passage number to 0 or 1, for example '
        '{"1": 0, "2": 1}. Include every number once and write nothing else.')

PROMPT_SHA256 = hashlib.sha256((HEAD + "|" + MID + "|" + TAIL).encode("utf-8")).hexdigest()


def build_prompt(query, passages):
    body = "\n\n".join("[%d]\n%s" % (i, p) for i, p in enumerate(passages, 1))
    return HEAD + query + MID + body + TAIL


def parse(raw, n):
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        return None
    if not isinstance(d, dict) or set(d) != {str(i) for i in range(1, n + 1)}:
        return None
    if any(v not in (0, 1) or isinstance(v, bool) for v in d.values()):
        return None
    return [d[str(i)] for i in range(1, n + 1)]


if __name__ == "__main__":
    with open("scratch_synth/judge_val_items.jsonl", encoding="utf-8") as f:
        row = json.loads(f.readline())
    need = set(row["point_ids"])
    text = {}
    with open("data/processed/passages.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                text[i] = json.loads(line)["text"]
    q = None
    with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] == row["qid"]:
                q = r["query"]
                break
    p = build_prompt(q, [text[i] for i in row["point_ids"]])
    print("prompt_sha256", PROMPT_SHA256)
    print("chars", len(p), "n_passages", len(row["point_ids"]))
    print("parse ok", parse('{"1": 0, "2": 1, "3": 0, "4": 0}', 4), "| bad", parse('{"1": 2}', 4))
    print(p[:700].encode("ascii", "replace").decode())