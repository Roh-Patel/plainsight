from pathlib import Path
from bs4 import BeautifulSoup
from ingest.html_text import prep, text_of, mml
from ingest.parse_html import HTML, DROP_SEL

out = []
for pid in ["2411.04078", "2504.17585"]:
    soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
    root = soup.select_one("article.ltx_document")
    for sel in DROP_SEL:
        for t in root.select(sel):
            t.extract()
    before = len(root.find_all("math"))
    log = prep(root)
    left = root.find_all("math")
    out.append(f"\n== {pid}: math before={before} counted_by_prep={log['count'].get('math')} left={len(left)}")
    for m in left[:3]:
        chain = [q.name + "." + ".".join(q.get("class", [])) for q in m.parents][:6]
        out.append(f"  tag={m.name} attrs={dict(list(m.attrs.items())[:3])}")
        out.append(f"  ancestors: {' < '.join(chain)}")
        out.append(f"  mml(m)={mml(m)!r}  alttext={m.get('alttext')!r}")
        out.append(f"  has semantics child: {m.find('semantics') is not None}; depth in tree: {len(list(m.parents))}")
    # second pass: does a repeat find them?
    log2 = prep(root)
    out.append(f"  second prep counted math={log2['count'].get('math')}; left now={len(root.find_all('math'))}")
Path("data/processed/diag_math.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote data/processed/diag_math.txt")