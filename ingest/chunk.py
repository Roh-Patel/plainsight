"""T05 chunker

Run: python -m ingest.chunk [--limit N] [--verify]
"""
import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass

MODEL = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 400
MIN_UNIT_TOKENS = 50
OVERLAP_TOKENS = 50
OVERLAP_SENTENCES = 2

CANDIDATES = "data/processed/candidates.jsonl"
PARSED = "data/processed/parsed.jsonl"
PASSAGES = "data/processed/passages.jsonl"
DROPS = "data/processed/chunk_drops.jsonl"

STATS = Counter()

# ---------------------------------------------------------------- sentences

ABBREV = {
    "al", "fig", "figs", "eq", "eqs", "sect", "sec", "e.g", "i.e", "cf", "vs",
    "approx", "ref", "refs", "dr", "prof", "mr", "mrs", "ms", "tab", "ch",
    "ed", "eds", "vol", "pp", "resp", "ca",
}
_BOUNDARY = re.compile(r"([.!?])([\"'”’)\]]*)(\s+)")
_LAST_WORD = re.compile(r"[\w.]+$")


def _starts_sentence(ch):
    return ch.isupper() or ch.isdigit() or ch in "\"“‘'([∙•"


def split_sentences(text):
    """Regex sentence splitter with an abbreviation guard. No NLTK/spaCy."""
    text = text.strip()
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        if end >= len(text):
            break
        if not _starts_sentence(text[end]):
            continue
        if m.group(1) == ".":
            w = _LAST_WORD.search(text[:m.start()])
            if w:
                word = w.group(0)
                if word.lower() in ABBREV:
                    continue
                if len(word) == 1 and word.isalpha() and word.isupper():
                    continue  # initial, e.g. "J. Smith"
        out.append(text[start:m.end(2)].strip())
        start = end
    out.append(text[start:].strip())
    return [s for s in out if s]


# ---------------------------------------------------------------- tokenizer

class BgeTokenizer:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.model_max_length = 10**9

    def count(self, texts):
        if not texts:
            return []
        return [len(x) for x in self.tok(texts, add_special_tokens=False)["input_ids"]]

    def hard_split(self, text, max_tokens):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        pieces = []
        for i in range(0, len(offs), max_tokens):
            j = min(i + max_tokens, len(offs)) - 1
            pieces.append(text[offs[i][0]:offs[j][1]].strip())
        return [p for p in pieces if p]


def hard_split_checked(text, tk, max_tokens):
    """Token split of one oversized sentence. Re-counts pieces; shrinks if a cut
    inside a word makes a piece re-tokenize longer than max_tokens."""
    size = max_tokens
    while size > 0:
        pieces = tk.hard_split(text, size)
        counts = tk.count(pieces)
        if all(c <= max_tokens for c in counts):
            return list(zip(pieces, counts))
        size -= 5
    raise ValueError("hard split failed")


# ------------------------------------------------------------------ packing

@dataclass
class Atom:
    text: str
    n: int
    para: int
    sents: list  # [(sentence_text, n_tokens)], used for overlap


def build_atoms(paragraphs, tk, max_tokens):
    """A paragraph that fits is one atom. A longer one becomes one atom per
    sentence; a sentence over max_tokens is hard-split by tokens."""
    per_para = [split_sentences(p) for p in paragraphs]
    counts = tk.count([s for ss in per_para for s in ss])
    atoms, k = [], 0
    for pi, (p, ss) in enumerate(zip(paragraphs, per_para)):
        sents = list(zip(ss, counts[k:k + len(ss)]))
        k += len(ss)
        total = sum(n for _, n in sents)
        if total <= max_tokens:
            atoms.append(Atom(p, total, pi, sents))
            continue
        STATS["paragraphs_split_by_sentence"] += 1
        for s, n in sents:
            if n <= max_tokens:
                atoms.append(Atom(s, n, pi, [(s, n)]))
            else:
                STATS["sentences_hard_split"] += 1
                for piece, c in hard_split_checked(s, tk, max_tokens):
                    atoms.append(Atom(piece, c, pi, [(piece, c)]))
    return atoms


def join_atoms(atoms):
    parts = []
    for i, a in enumerate(atoms):
        if i:
            parts.append("\n\n" if a.para != atoms[i - 1].para else " ")
        parts.append(a.text)
    return "".join(parts)


def overlap_atoms(prev_atoms, overlap_tokens, max_sents):
    flat = [(s, n, a.para) for a in prev_atoms for (s, n) in a.sents]
    picked, tot = [], 0
    for s, n, p in reversed(flat[-max_sents:]):
        if tot + n > overlap_tokens:
            break
        picked.append(Atom(s, n, p, [(s, n)]))
        tot += n
    picked.reverse()
    return picked


def pack(atoms, max_tokens, overlap_tokens, overlap_sents):
    """Greedy packing. Returns [(text, n_tokens)]. n_tokens is the sum of atom
    counts, exact because wordpiece counts are additive across whitespace."""
    chunks, cur, cur_n = [], [], 0
    for a in atoms:
        if cur and cur_n + a.n > max_tokens:
            chunks.append((join_atoms(cur), cur_n))
            ov = overlap_atoms(cur, overlap_tokens, overlap_sents)
            cur, cur_n = ov, sum(x.n for x in ov)
            if cur_n + a.n > max_tokens:
                cur, cur_n = [], 0  # overlap does not fit; start clean
        cur.append(a)
        cur_n += a.n
    if cur:
        chunks.append((join_atoms(cur), cur_n))
    return chunks


# -------------------------------------------------------------- junk filters

STOP = frozenset("""the of and a an in to is are was were be been this that these those
for with on at by from as it its we our which can has have had not or but also than
then their there such may more most into between using used use""".split())
_WORD = re.compile(r"[A-Za-z']+")
_LONG_RUN = re.compile(r"[A-Za-z]{25,}")


def function_word_frac(text):
    """Share of whitespace-delimited words that are English function words.
    Prose is about 0.37; tables, name lists and non-English text are near 0."""
    n = max(1, len(text.split()))
    return sum(w.lower() in STOP for w in _WORD.findall(text)) / n


def junk_reason(text):
    """D017 rules, in order."""
    if sum(c.isspace() for c in text) * 12 < len(text):
        return "missing_spaces"
    if len(_LONG_RUN.findall(text)) >= 3:
        return "long_letter_runs"
    stop = function_word_frac(text)
    if stop < 0.10:
        return "low_function_words"
    if stop < 0.25:
        words = text.split()
        nonalpha = sum(1 for w in words if not any(c.isalpha() for c in w))
        if nonalpha / len(words) > 0.30:
            return "table_residue"
        if text.count(",") / len(words) > 0.25:
            return "comma_density"
    return None


# ------------------------------------------------------------------- driver

def process_paper(paper_id, title, units, tk, max_tokens=MAX_TOKENS,
                  min_unit_tokens=MIN_UNIT_TOKENS, overlap_tokens=OVERLAP_TOKENS,
                  overlap_sents=OVERLAP_SENTENCES):
    passages, drops = [], []
    for u in units:
        paras = [p.strip() for p in u["text"].split("\n\n") if p.strip()]
        atoms = build_atoms(paras, tk, max_tokens) if paras else []
        total = sum(a.n for a in atoms)
        if total < min_unit_tokens:
            drops.append({"paper_id": paper_id, "section_idx": u["idx"],
                          "chunk_idx": None, "reason": "unit_lt_50_tokens",
                          "n_tokens": total, "preview": u["text"][:100]})
            continue
        for ci, (text, n) in enumerate(pack(atoms, max_tokens, overlap_tokens, overlap_sents)):
            reason = junk_reason(text)
            if reason:
                drops.append({"paper_id": paper_id, "section_idx": u["idx"],
                              "chunk_idx": ci, "reason": reason,
                              "n_tokens": n, "preview": text[:100]})
                continue
            parts = [title] + ([u["title"]] if u["title"] else []) + [text]
            passages.append({
                "passage_id": f"{paper_id}::{u['idx']:03d}::{ci:03d}",
                "paper_id": paper_id,
                "section_title": u["title"],
                "section_idx": u["idx"],
                "chunk_idx": ci,
                "text": text,
                "embed_text": " | ".join(parts),
                "n_tokens": n,
            })
    return passages, drops


def pct(v):
    v = sorted(v)
    n = len(v)
    out = {q: v[min(n - 1, int(n * q / 100))] for q in (5, 25, 50, 75, 95, 99)}
    out["max"] = v[-1]
    return out


def verify(path, tk, n=2000):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.seed(0)
    sample = random.sample(rows, min(n, len(rows)))
    body = tk.count([r["text"] for r in sample])
    emb = tk.count([r["embed_text"] for r in sample])
    mism = sum(1 for r, b in zip(sample, body) if r["n_tokens"] != b)
    print(f"verify (sample {len(sample)}): n_tokens mismatches={mism}, "
          f"max body={max(body)}, max embed_text+2 specials={max(emb) + 2}, "
          f"over 512={sum(e + 2 > 512 for e in emb)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N papers only")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    tk = BgeTokenizer()
    titles = {}
    with open(CANDIDATES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            titles[r["paper_id"]] = " ".join(r["title"].split())

    n_papers = n_units = 0
    n_tok, reasons = [], Counter()
    with open(PARSED, encoding="utf-8") as f, \
         open(PASSAGES, "w", encoding="utf-8", newline="\n") as fp, \
         open(DROPS, "w", encoding="utf-8", newline="\n") as fd:
        for line in f:
            if args.limit and n_papers >= args.limit:
                break
            row = json.loads(line)
            n_papers += 1
            n_units += len(row["units"])
            passages, drops = process_paper(row["paper_id"], titles[row["paper_id"]],
                                            row["units"], tk)
            for p in passages:
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_tok.append(p["n_tokens"])
            for d in drops:
                fd.write(json.dumps(d, ensure_ascii=False) + "\n")
                reasons[d["reason"]] += 1
            if n_papers % 500 == 0:
                print(f"  {n_papers} papers, {len(n_tok)} passages", flush=True)

    print(f"papers={n_papers} units={n_units} passages kept={len(n_tok)}")
    print("drops by reason:", dict(reasons))
    print("stats:", dict(STATS))
    print("passage n_tokens:", pct(n_tok))
    print(f"passages < 50 tokens: {sum(t < 50 for t in n_tok)}, "
          f"< 150 tokens: {sum(t < 150 for t in n_tok)}")
    if args.verify:
        verify(PASSAGES, tk)


if __name__ == "__main__":
    main()