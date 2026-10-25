from pathlib import Path
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from ingest.html_text import prep, text_of
from ingest.parse_html import DROP_SEL, HTML

OUT = Path("data/processed/diag_cov.txt")
out = []


def sig(el):
    return el.name + "".join("." + c for c in el.get("class", []))


def kids(el, depth, maxd):
    rows = []
    bare = sum(len(" ".join(str(c).split())) for c in el.children if isinstance(c, NavigableString))
    out.append("  " * depth + f"[bare text directly in {sig(el)}: {bare} chars]")
    for c in el.children:
        if isinstance(c, Tag):
            rows.append((len(text_of(c)), c))
    for n, c in sorted(rows, key=lambda r: -r[0])[:6]:
        h = c.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
        out.append("  " * depth + f"{n:7d} {sig(c)} {text_of(h)[:50] if h else ''}")
    if rows and depth < maxd:
        kids(max(rows, key=lambda r: r[0])[1], depth + 1, maxd)


for pid in ["2607.14326", "2510.03916", "2408.10341", "2407.17541", "2407.07515"]:
    soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
    root = soup.select_one("article.ltx_document")
    for sel in DROP_SEL:
        for t in root.select(sel):
            t.extract()
    prep(root)
    out.append(f"\n== {pid} root chars={len(text_of(root))}")
    kids(root, 0, 2)

raw = (HTML / "2504.17585.html").read_text(encoding="utf-8", errors="replace")
i = raw.find("7\\times 10^{-7}")
out.append("\n== raw HTML around the TeX leak, 2504.17585 ==")
out.append(raw[max(0, i - 900):i + 500] if i >= 0 else "NOT FOUND")

Path("data/processed/diag_cov.txt").write_text("\n".join(out), encoding="utf-8")
print(f"wrote {OUT} ({len(out)} lines)")