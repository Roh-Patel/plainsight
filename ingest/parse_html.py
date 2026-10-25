import argparse
import collections
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from ingest.html_text import prep, text_of, heading_title

HTML = Path("data/raw/html")
FETCH_LOG = Path("data/raw/fetch_log.jsonl")
P = Path("data/processed")

DROP_IDS = {"2401.02195": "font-map garble across about 44% of text (D014)"}
CHECK = ("2405.01605 2405.13283 2407.15263 2409.14321 2410.03034 2501.10594 2501.11761 "
         "2501.14016 2504.01263 2508.18111 2509.08044 2511.10275 2512.12453 2512.17976 "
         "2601.23152 2602.13604 2608.06086 2401.00715 2401.02195 2401.03437 2403.01283 "
         "2401.01400").split()
SECT = {"ltx_" + n for n in ("part", "chapter", "section", "subsection", "subsubsection",
                             "paragraph", "subparagraph", "appendix")}
TEXT_CLS = {"ltx_para", "ltx_itemize", "ltx_enumerate", "ltx_description",
            "ltx_theorem", "ltx_proof", "ltx_p"}
DROP_SEL = ["figure.ltx_figure", "figure.ltx_table", ".ltx_tabular", ".ltx_listing",
            ".ltx_TOC", ".ltx_page_navbar", ".ltx_authors", ".ltx_dates", ".ltx_pubnotes",
            ".ltx_abstract", ".ltx_keywords", ".ltx_classification", ".ltx_acknowledgements",
            ".ltx_bibliography", ".ltx_pagination", ".ltx_rdf", ".ltx_subtitle",
            ".ltx_title_document", ".ltx_logical-block"]
HEAD = re.compile(r"^h[1-6]$")
DROP_TITLE = re.compile(
    r"^(references|bibliography|abstract|acknowledg(e)?ments?|"
    r"(statements and )?declarations?|disclosures?( statement)?|"
    r"(authors? )?contributions?( statement)?|credit authorship contribution statement|"
    r"(declaration of )?competing interests?( statement)?|"
    r"conflicts? of interests?( statement| declaration| disclosure)?|"
    r"funding( sources?| statement)?|"
    r"((data|code|software|materials|and) )+availability( statements?)?|"
    r"availability of( (data|code|software|materials|and))+|"
    r"orcid i?ds?|data and resources|software attribution|"
    r"software and third party data repository citations|"
    r"institutional review board statement|informed consent statement|"
    r"declaration of generative ai.*)$")
RESID = re.compile(r"\((?:\s|,|;|e\.g\.|i\.e\.|see|cf\.|and|or)*\)")
TEX = re.compile(r"\\[A-Za-z]{3,}")
AUTH = re.compile(r"^(\[(orcid|style)=|show\]|\]\S+@)")
GARBLE = re.compile("[\u2309\u230a\u2294\u2293\u22a3\u2240\u2195\u21d5\u2291\u2292\u230b\u2308]")
KEY = re.compile("acknowledg|fund|contribut|availab|conflict|compet|declar|ethic|disclos|"
                 "statement|supplement|software|code")


def norm(t):
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def pct(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, int(q * len(a)))] if a else 0


class Walker:
    def __init__(self, lg):
        self.units, self.buf, self.crumbs, self.lg = [], [], [], lg

    def flush(self):
        text = "\n\n".join(self.buf)
        if text.strip():
            self.units.append({"title": " > ".join(c for c in self.crumbs if c), "text": text})
        self.buf = []

    def visit(self, el):
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            cls = set(c.get("class", []))
            if HEAD.match(c.name) or "ltx_title" in cls:
                continue
            if c.name == "section" and cls & SECT:
                self.section(c, cls)
            elif cls & TEXT_CLS or c.name in ("p", "ul", "ol"):
                t = text_of(c)
                hits = RESID.findall(t)
                if hits:
                    self.lg["resid"] += len(hits)
                    if len(self.lg["resid_samples"]) < 3:
                        self.lg["resid_samples"].append(hits[0])
                    t = RESID.sub("", t)
                    t = re.sub(r"\s+([,.;:])", r"\1", t)
                    t = re.sub(r"\s{2,}", " ", t).strip()
                if AUTH.match(t):
                    self.lg["junk"] += 1
                    t = ""
                if "\\lxSVG" in t:
                    self.lg["junk"] += 1
                    t = ""
                if t:
                    self.buf.append(t)
            else:
                self.lg["unknown"][c.name + "." + ".".join(sorted(cls))] += 1
                self.visit(c)

    def section(self, c, cls):
        h = c.find(HEAD, recursive=False)
        title = ""
        if h is not None:
            title = heading_title(h).strip(" .:")
        else:
            self.lg["no_heading"] += 1
        if DROP_TITLE.match(norm(title)):
            self.lg["dropped_titles"].append(title)
            self.flush()
            n0 = len(self.units)
            for ch in c.children:
                if not isinstance(ch, Tag):
                    continue
                chcls = set(ch.get("class", []))
                if ch.name == "section" and chcls & SECT:
                    self.section(ch, chcls)
                elif not (HEAD.match(ch.name) or "ltx_title" in chcls):
                    self.lg["dropped_chars"] += len(text_of(ch))
            self.lg["rescued"] += sum(len(u["text"]) for u in self.units[n0:])
            return
        self.flush()
        self.crumbs.append("Appendix: " + title if "ltx_appendix" in cls and title else title)
        self.visit(c)
        self.flush()
        self.crumbs.pop()
        self.lg["kept_titles"].append(title)


def parse(pid):
    lg = {"dropped_sel": collections.Counter(), "unknown": collections.Counter(),
          "dropped_titles": [], "kept_titles": [], "no_heading": 0, "logical": [],
          "resid": 0, "resid_samples": [], "dropped_chars": 0, "rescued": 0, "junk": 0}
    soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
    root = soup.select_one("article.ltx_document")
    if root is None:
        return None, dict(lg, reason="no_article")
    for b in root.select(".ltx_logical-block"):
        s = " ".join(b.get_text(" ").split())
        lg["logical"].append([len(s), s[:150]])
    for sel in DROP_SEL:
        for t in root.select(sel):
            t.extract()
            lg["dropped_sel"][sel] += 1
    lg["prep"] = prep(root).get("count", {})
    lg["root_chars"] = len(text_of(root))
    w = Walker(lg)
    w.visit(root)
    w.flush()
    units = [{"idx": i, **u} for i, u in enumerate(w.units)]
    total = sum(len(u["text"]) for u in units)
    garble = len(GARBLE.findall(" ".join(u["text"] for u in units))) / max(total, 1)
    return ({"paper_id": pid, "units": units, "n_chars": total},
            dict(lg, garble=garble, cov=total / max(lg["root_chars"] - lg["dropped_chars"], 1)))


def ok_ids():
    last = {}
    for line in FETCH_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["paper_id"]] = r["status"]
    return sorted(p for p, s in last.items() if s == "ok")


def summarize(logs, rows):
    S = []
    pl = [l for l in logs if l["status"] == "parsed"]
    S.append(f"status: {dict(collections.Counter(l['status'] for l in logs))}")
    for l in logs:
        if l["status"] != "parsed":
            S.append(f"  {l['status']}: {l['paper_id']} {l.get('reason')}")
    nu = [l["n_units"] for l in pl]
    S.append(f"units per paper: min={min(nu, default=0)} median={pct(nu, .5)} "
             f"p90={pct(nu, .9)} max={max(nu, default=0)}")
    single = [l["paper_id"] for l in pl if l["n_units"] == 1]
    S.append(f"single-unit papers: {len(single)} {single[:40]}")
    lens = [len(u["text"]) for r in rows for u in r["units"]]
    S.append(f"unit chars: p10={pct(lens, .1)} median={pct(lens, .5)} p90={pct(lens, .9)} "
             f"max={max(lens, default=0)}; units under 300 chars: {sum(1 for x in lens if x < 300)} of {len(lens)}")
    S.append("smallest papers (id, chars, units): " + str(
        sorted((l["paper_id"], l["n_chars"], l["n_units"]) for l in pl)[:0]
        or sorted(((l["n_chars"], l["paper_id"], l["n_units"]) for l in pl))[:10]))
    S.append(f"garble ratio >0.001 in {sum(1 for l in pl if l['garble'] > 0.001)} papers; top 10: "
             f"{sorted(((round(l['garble'], 5), l['paper_id']) for l in pl), reverse=True)[:10]}")
    S.append(f"residue removed: {sum(l['resid'] for l in pl)} in {sum(1 for l in pl if l['resid'])} papers; "
             f"samples: {[s for l in pl for s in l['resid_samples']][:15]}")
    d = collections.Counter(norm(t) for l in pl for t in l["dropped_titles"])
    k = collections.Counter(norm(t) for l in pl for t in l["kept_titles"])
    S.append(f"\ndropped titles (top 25): {d.most_common(25)}")
    S.append(f"kept titles (top 40): {k.most_common(40)}")
    S.append(f"kept titles matching boilerplate keywords: {[(t, n) for t, n in k.most_common() if KEY.search(t)][:40]}")
    u = collections.Counter()
    ds = collections.Counter()
    for l in pl:
        u.update(l["unknown"])
        ds.update(l["dropped_sel"])
    S.append(f"\nunknown container signatures: {u.most_common(15)}")
    S.append(f"dropped selectors: {dict(ds)}")
    S.append(f"sections without heading: {sum(l['no_heading'] for l in pl)}")
    lb = [(l["paper_id"], *x) for l in pl for x in l["logical"]]
    S.append(f"logical blocks removed: {len(lb)}; first 10: {lb[:10]}")
    res = [(r["paper_id"], m.group(0)) for r in rows for u in r["units"] for m in RESID.finditer(u["text"])]
    S.append(f"empty-paren residue: {len(res)} in {len({p for p, _ in res})} papers; "
             f"top: {collections.Counter(x for _, x in res).most_common(10)}")
    S.append(f"logical blocks over 600 chars: {[x for x in lb if x[1] > 600]}")
    cv = [l["cov"] for l in pl]
    S.append(f"coverage (unit chars / article chars after prep): p5={pct(cv, .05):.3f} "
             f"median={pct(cv, .5):.3f} min={min(cv):.3f}; below 0.8: {sum(1 for x in cv if x < 0.8)}")
    low = sorted(pl, key=lambda l: l["cov"])[:15]
    S.append("lowest coverage: " + str([(round(l["cov"], 2), l["paper_id"], l["n_units"], l["n_chars"],
                                        sorted(l["unknown"].items(), key=lambda kv: -kv[1])[:2]) for l in low]))
    tex = [(r["paper_id"], m.group(0), u["text"][max(0, m.start() - 40):m.end() + 40])
           for r in rows for u in r["units"] for m in TEX.finditer(u["text"])]
    S.append(f"TeX leak: {len(tex)} in {len({x[0] for x in tex})} papers; "
             f"top: {collections.Counter(x[1] for x in tex).most_common(15)}")
    S.append(f"TeX samples: {tex[::max(1, len(tex) // 8)][:8]}")
    S.append(f"TeX leak by paper (top 10): {collections.Counter(x[0] for x in tex).most_common(10)}")
    for mac in ("\\lxSVG", "\\par", "\\timeform", "\\hbox"):
        hit = [x for x in tex if x[1] == mac]
        S.append(f"{mac}: {len(hit)} in {len({x[0] for x in hit})} papers; samples: {[(x[0], x[2]) for x in hit[:3]]}")

    def dshare(t):
        s = t.replace(" ", "")
        return sum(ch.isdigit() for ch in s) / max(len(s), 1)
    num = [(r["paper_id"], u["title"][-30:], round(dshare(u["text"]), 2), u["text"][:90])
           for r in rows for u in r["units"] if len(u["text"]) > 200 and dshare(u["text"]) > 0.3]
    S.append(f"digit-heavy units (>30% digits, >200 chars): {len(num)} in {len({x[0] for x in num})} papers; "
             f"samples: {num[::max(1, len(num) // 8)][:8]}")
    short = [(r["paper_id"], u["title"][-40:], u["text"][:90])
             for r in rows for u in r["units"] if len(u["text"]) < 300]
    S.append(f"short-unit samples ({len(short)} total): {short[::max(1, len(short) // 12)][:12]}")
    S.append(f"rescued from dropped-title sections: {sum(l['rescued'] for l in pl)} chars in "
             f"{sum(1 for l in pl if l['rescued'])} papers; junk paragraphs dropped: {sum(l['junk'] for l in pl)}")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--nodrop", action="store_true")
    a = ap.parse_args()
    ids = CHECK if a.check else (a.ids or ok_ids())
    if a.limit:
        ids = ids[:a.limit]
    sfx = "_test" if (a.check or a.ids or a.limit) else ""
    rows, logs = [], []
    for i, pid in enumerate(ids, 1):
        if i % 200 == 0:
            print(f"{i}/{len(ids)}")
        if pid in DROP_IDS and not a.nodrop:
            logs.append({"paper_id": pid, "status": "dropped", "reason": DROP_IDS[pid]})
            continue
        try:
            row, lg = parse(pid)
        except Exception as e:
            logs.append({"paper_id": pid, "status": "error", "reason": repr(e)})
            continue
        if row is None or not row["units"]:
            logs.append({"paper_id": pid, "status": "dropped", "reason": lg.get("reason", "no_text")})
            continue
        if lg.get("garble", 0) > 0.01 and not a.nodrop:
            logs.append({"paper_id": pid, "status": "dropped", "reason": f"garble ratio {lg['garble']:.3f}"})
            continue
        rows.append(row)
        logs.append({"paper_id": pid, "status": "parsed", "n_units": len(row["units"]),
                     "n_chars": row["n_chars"], **{k: (dict(v) if isinstance(v, collections.Counter) else v)
                                                   for k, v in lg.items()}})
    P.mkdir(parents=True, exist_ok=True)
    with open(P / f"parsed{sfx}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(P / f"parse_log{sfx}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in logs)
    (P / f"parse_summary{sfx}.txt").write_text("\n".join(summarize(logs, rows)), encoding="utf-8")
    if sfx:
        S = []
        for r in rows:
            S.append(f"\n===== {r['paper_id']} ({len(r['units'])} units, {r['n_chars']} chars)")
            for u in r["units"]:
                S.append(f"{u['idx']:02d} | {u['title']!r} | {len(u['text'])} | {u['text'][:100]!r}")
        (P / f"parse_sample{sfx}.txt").write_text("\n".join(S), encoding="utf-8")
    print(f"parsed {len(rows)} of {len(ids)}; wrote parsed{sfx}.jsonl, parse_log{sfx}.jsonl, parse_summary{sfx}.txt")


if __name__ == "__main__":
    main()
