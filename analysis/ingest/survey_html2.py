import re
import collections
from pathlib import Path
from bs4 import BeautifulSoup

HTML = Path("data/raw/html")
OUT = Path("data/processed/survey_html2.txt")
SAMPLE = "2401.00715"
SECTIONLESS = ("2405.01605 2405.13283 2407.15263 2409.14321 2410.03034 2501.10594 "
               "2501.11761 2501.14016 2504.01263 2508.18111 2509.08044 2511.10275 "
               "2512.12453 2512.17976 2601.23152 2602.13604 2608.06086").split()
KEYS = ("ERROR", "note", "theorem", "proof", "itemize", "enumerate", "listing",
        "paragraph", "appendix", "chapter", "part", "acknowledg", "footnote",
        "missing", "citemacro", "keywords", "subsubsection")
WANT = ["ltx_note", "ltx_ERROR", "ltx_itemize", "ltx_theorem", "ltx_acknowledgements",
        "ltx_appendix", "ltx_paragraph", "ltx_subsubsection", "ltx_listing"]

out = []
p = out.append


def load(pid):
    return BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")


def sig(el):
    c = el.get("class", [])
    return el.name + ("." + ".".join(c) if c else "")


def seq(el, limit=40):
    runs = []
    for c in el.find_all(recursive=False):
        s = sig(c)
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return ", ".join(f"{s} x{n}" if n > 1 else s for s, n in runs[:limit])


# 1. class document frequency beyond the first survey
df = collections.Counter()
first = {}
for f in sorted(HTML.glob("*.html")):
    t = f.read_text(encoding="utf-8", errors="replace")
    cl = set()
    for m in re.finditer(r'class="([^"]*)"', t):
        cl.update(m.group(1).split())
    cl = {c for c in cl if c.startswith("ltx_")}
    df.update(cl)
    for w in WANT:
        if w in cl and w not in first:
            first[w] = f.stem
p("== classes ranked 71-220, plus keyword matches beyond that ==")
for i, (c, n) in enumerate(df.most_common()):
    if 70 <= i < 220 or (i >= 220 and any(k in c for k in KEYS)):
        p(f"{n:5d}  {c}")

# 2. the 17 sectionless files
p("\n== sectionless files: top-level children and headings ==")
for pid in SECTIONLESS:
    soup = load(pid)
    art = soup.find(class_="ltx_document")
    p(f"\n-- {pid}")
    p("   document children: " + (seq(art) if art else "NO ltx_document"))
    n = 0
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        if h.find_parent(class_="ltx_TOC"):
            continue
        n += 1
        if n > 14:
            break
        p(f"   {h.name}.{'.'.join(h.get('class', []))} | parent={sig(h.parent)} | "
          f"{' '.join(h.get_text().split())[:70]!r}")

# 3. nesting, equation placement, math text (sample file)
soup = load(SAMPLE)
p(f"\n== nesting in {SAMPLE} ==")
for s in soup.find_all("section", class_="ltx_section"):
    if s.find("section", class_="ltx_subsection"):
        p("section children: " + seq(s))
        sub = s.find("section", class_="ltx_subsection")
        p("first subsection children: " + seq(sub))
        break
else:
    p("NO section with a subsection in sample")

eq = soup.find("table", class_="ltx_equation")
p("\n== equation placement ==")
if eq:
    p(f"parent={sig(eq.parent)}  grandparent={sig(eq.parent.parent)}")
    p("parent children: " + seq(eq.parent))

p("\n== math text handling ==")
para = soup.find("div", class_="ltx_para")
m = para.find("math") if para else None
if m:
    m2 = BeautifulSoup(str(m), "lxml")
    for a in m2.find_all(["annotation", "annotation-xml"]):
        a.decompose()
    p(f"get_text raw={m.get_text()!r}  annotation removed={m2.get_text()!r}  alttext={m.get('alttext')!r}")

# 4. snippets of rarer structures
p("\n== first example of rarer structures ==")
for w in WANT:
    if w not in first:
        p(f"\n-- {w}: df={df.get(w, 0)} (none found)")
        continue
    s2 = load(first[w])
    el = s2.find(class_=w)
    p(f"\n-- {w}: df={df[w]}, file {first[w]}\n{str(el)[:600] if el else 'NOT FOUND AFTER PARSE'}")

OUT.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {OUT} ({len(out)} lines)")