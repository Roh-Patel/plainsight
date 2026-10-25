import re
import collections
from pathlib import Path
from bs4 import BeautifulSoup

HTML = Path("data/raw/html")
OUT = Path("data/processed/diag_text.txt")
GARBLE = re.compile("[\u2100-\u214f\u2190-\u23ff\u25a0-\u25ff\U0001D400-\U0001D7FF]")
out = []
p = out.append

CASES = [("2401.00715", "\\penalty\\ 10\\,\\textnormal{M}_{\\textnormal{Jup}}"),
         ("2401.00715", "\\beta\\>"),
         ("2403.01283", "i_{\\mathrm{M}}=0)")]
p("== raw MathML for spacing / OPEN cases ==")
for pid, alt in CASES:
    soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
    m = next((x for x in soup.find_all("math") if x.get("alttext") == alt), None)
    p(f"\n-- {pid} {alt!r}\n{str(m)[:1200] if m else 'NOT FOUND'}")

p("\n== garble survey (chars in symbol blocks outside <math>, before the bibliography) ==")
rows = []
for f in sorted(HTML.glob("*.html")):
    t = f.read_text(encoding="utf-8", errors="replace")
    i = t.find('<section class="ltx_bibliography"')
    t = t[:i] if i > 0 else t
    t = re.sub(r'<span class="ltx_ERROR[^"]*">.*?</span>', "", t, flags=re.S)
    t = re.sub(r"<math.*?</math>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    hits = GARBLE.findall(t)
    if hits:
        j = GARBLE.search(t).start()
        rows.append((len(hits), f.stem, len(t), " ".join(t[max(0, j - 40):j + 80].split())))
rows.sort(reverse=True)
c = collections.Counter()
for n, *_ in rows:
    c["files with >=1"] += 1
    for k in (5, 20, 100, 500):
        if n >= k:
            c[f"files with >={k}"] += 1
p(str(dict(c)))
p("\ntop 25 (count, paper, text_len, sample around first hit):")
for n, pid, L, s in rows[:25]:
    p(f"{n:6d}  {pid}  len={L}  {s!r}")
p("\nsample of low-count files (5-19 hits):")
for n, pid, L, s in [r for r in rows if 5 <= r[0] < 20][:10]:
    p(f"{n:6d}  {pid}  {s!r}")

OUT.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {OUT} ({len(out)} lines)")
