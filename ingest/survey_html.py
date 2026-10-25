import re
import collections
from pathlib import Path
from bs4 import BeautifulSoup

HTML = Path("data/raw/html")
OUT = Path("data/processed/survey_html.txt")
STRUCT = ["ltx_section", "ltx_subsection", "ltx_appendix", "ltx_abstract",
          "ltx_bibliography", "ltx_para", "ltx_equation", "ltx_equationgroup"]

files = sorted(HTML.glob("*.html"))
df = collections.Counter()
nosec = []
first_sectioned = None

for f in files:
    t = f.read_text(encoding="utf-8", errors="replace")
    classes = set()
    for m in re.finditer(r'class="([^"]*)"', t):
        classes.update(m.group(1).split())
    df.update(c for c in classes if c.startswith("ltx_"))
    if "ltx_section" not in classes:
        nosec.append((f.stem, len(t), t.count("ltx_para"),
                      [c for c in STRUCT if c in classes]))
    elif first_sectioned is None:
        first_sectioned = f

out = []
p = out.append
p(f"files: {len(files)}")
p("\n== ltx_ classes by number of files containing them (top 70) ==")
for c, n in df.most_common(70):
    p(f"{n:5d}  {c}")

p(f"\n== files with no ltx_section token: {len(nosec)} ==")
for pid, size, npara, present in nosec:
    p(f"{pid}  bytes={size}  para_substr={npara}  has={present}")

soup = BeautifulSoup(first_sectioned.read_text(encoding="utf-8", errors="replace"), "lxml")
p(f"\n== sample file: {first_sectioned.stem} ==")
sec = soup.find("section", class_="ltx_section")
if sec is None:
    sec = soup.find(class_="ltx_section")
p("-- first section tag/class: " + (f"{sec.name} {sec.get('class')}" if sec else "NONE"))
if sec:
    h = sec.find(re.compile(r"^h[1-6]$"))
    p("-- heading: " + (str(h)[:500] if h else "NONE"))
    para = sec.find(class_="ltx_para")
    p("-- first para: " + (str(para)[:1000] if para else "NONE"))
    p("-- direct children: " + ", ".join(
        f"{c.name}.{'.'.join(c.get('class', []))}" for c in sec.find_all(recursive=False)))
for label, cls in [("abstract", "ltx_abstract"), ("bibliography", "ltx_bibliography"),
                   ("equation", "ltx_equation"), ("figure", "ltx_figure"),
                   ("table", "ltx_table")]:
    el = soup.find(class_=cls)
    p(f"\n-- {label}: " + (str(el)[:700] if el else "NONE in sample file"))
m = soup.find("math")
p("\n-- first <math> attrs: " + (str(m.attrs)[:400] if m else "NONE"))

OUT.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {OUT} ({len(out)} lines)")