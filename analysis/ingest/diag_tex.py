from pathlib import Path
from bs4 import BeautifulSoup
from ingest.html_text import prep, text_of
from ingest.parse_html import HTML

out = []
for pid, needle in [("2411.04078", "well above 10"), ("2504.17585", "softer case"),
                    ("2401.17764", "coverage. PIONIER")]:
    soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
    root = soup.select_one("article.ltx_document")
    p = next((x for x in root.select("div.ltx_para p, div.ltx_para") if needle in x.get_text()), None)
    out.append(f"\n== {pid} ({needle!r}) ==")
    if p is None:
        out.append("paragraph not found")
        continue
    parents = [q.name + "." + ".".join(q.get("class", [])) for q in p.parents][:4]
    out.append("ancestors: " + " < ".join(parents))
    i = str(p).find(needle)
    out.append("RAW: " + str(p)[max(0, i - 100):i + 900])
    prep(root)
    out.append("AFTER PREP: " + repr(text_of(p))[max(0, i - 100):i + 400])
    out.append(f"math tags left: {len(p.find_all('math'))}")
Path("data/processed/diag_tex.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote data/processed/diag_tex.txt")