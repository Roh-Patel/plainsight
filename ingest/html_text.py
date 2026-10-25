import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

HTML = Path("data/raw/html")
OUT = Path("data/processed/html_text_demo.txt")
INVISIBLE = {ord(c): None for c in "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064\ufeff"}
MACROS = {"\\uppartial": "\u2202", "\\Sigmaup": "\u03a3", "\\Lambdaup": "\u039b",
          "\\upmum": "\u03bcm", "\\degree": "\u00b0",
          "\\upomega": "\u03c9", "\\upsigma": "\u03c3",
          "\\farcs": "\u2033", "\\earth": "\u2295"}
PAREN_CITES = {"ltx_citemacro_citep", "ltx_citemacro_cite", "ltx_citemacro_citepalias",
               "ltx_citemacro_citeyearpar", "ltx_citemacro_citenum"}
BLOCKS = ["p", "div", "li", "ul", "ol", "table", "tr", "td", "th", "figcaption",
          "section", "h1", "h2", "h3", "h4", "h5", "h6"]
DEMO = ["2401.00715", "2401.02195", "2403.01283", "2401.03437", "2402.13702"]


def _wrap(s):
    return s if len(s) <= 1 else "{" + s + "}"


def _paren(s):
    return s if s.isalnum() else "(" + s + ")"


def _em(v):
    m = re.match(r"\s*(-?\d*\.?\d+)em", v or "")
    return float(m.group(1)) if m else 0.0


def mml(el):
    """Presentation MathML to plain text, keeping sub/superscripts and fractions."""
    if isinstance(el, NavigableString):
        return str(el)
    n = el.name
    if n in ("annotation", "annotation-xml", "mphantom"):
        return ""
    if n == "mo":
        t = el.get_text().translate(INVISIBLE).strip()
        if t in ("OPEN", "CLOSE", "MIDDLE"):
            return ""
        if not t:
            return " " if max(_em(el.get("lspace")), _em(el.get("rspace"))) >= 0.15 else ""
    kids = [c for c in el.children if isinstance(c, Tag) or str(c).strip()]
    parts = [mml(c) for c in kids]
    if n == "msub" and len(parts) == 2:
        return parts[0] + "_" + _wrap(parts[1])
    if n == "msup" and len(parts) == 2:
        return parts[0] + "^" + _wrap(parts[1])
    if n == "msubsup" and len(parts) == 3:
        return parts[0] + "_" + _wrap(parts[1]) + "^" + _wrap(parts[2])
    if n == "mfrac" and len(parts) == 2:
        return "(" + parts[0] + "/" + parts[1] + ")"
    if n == "msqrt":
        return "sqrt(" + "".join(parts) + ")"
    return "".join(parts)


def clean(s):
    s = s.translate(INVISIBLE).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\\varv(?![A-Za-z])", "v", s)
    for k, v in MACROS.items():
        s = s.replace(k, v)
    s = re.sub(r"\^\{?[\u2218\u00b0]\}?", "\u00b0", s)
    s = re.sub(r"\^\{?([\u2032\u2033]+)\}?", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def prep(soup, log=None):
    """Mutate the soup in place so get_text() yields clean prose."""
    log = {} if log is None else log
    cnt = log.setdefault("count", {})
    smp = log.setdefault("samples", {"math": [], "cite_drop": [], "cite_keep": []})

    def bump(k):
        cnt[k] = cnt.get(k, 0) + 1

    for t in soup(["script", "style", "noscript"]):
        t.extract()
        bump("script")
    
    for sel, key in [(".ltx_note", "note"), (".ltx_ERROR", "error"),
                     (".ltx_missing_citation, .ltx_missing_label, .ltx_missing_image", "missing"),
                     (".ltx_equation, .ltx_equationgroup, .ltx_eqn_table", "display_eq")]:
        for t in soup.select(sel):
            t.extract()
            bump(key)
    for m in soup.find_all("math", attrs={"display": "block"}):
        m.extract()
        bump("display_math")

    for c in soup.select("cite"):
        txt = " ".join(c.get_text().split())
        if (set(c.get("class", [])) & PAREN_CITES) or (txt and txt[0] in "(["):
            if len(smp["cite_drop"]) < 8:
                smp["cite_drop"].append(txt[:80])
            c.extract()
            bump("cite_drop")
        else:
            if len(smp["cite_keep"]) < 8:
                smp["cite_keep"].append(txt[:80])
            bump("cite_keep")

    for m in soup.find_all("math"):
        alt = m.get("alttext", "")
        raw = mml(m).translate(INVISIBLE)
        s = raw.strip()
        if not s:
            s = alt.strip()
        else:
            if raw[:1].isspace() or re.match(r"\s*\\(?:[,>;: ]|quad|qquad)", alt):
                s = " " + s
            if raw[-1:].isspace() or re.search(r"\\(?:[,>;: ]|quad|qquad)\s*$", alt):
                s = s + " "
        if len(smp["math"]) < 400:
            smp["math"].append((alt, s.strip()))
        m.replace_with(NavigableString(s))
        bump("math")

    for s in soup.find_all("sup"):
        if "ltx_note_mark" in s.get("class", []):
            s.extract()
            continue
        t = " ".join(s.get_text().split())
        s.replace_with(NavigableString("^" + _wrap(t) if t else ""))
        bump("sup_text")

    for t in list(soup.find_all(BLOCKS + ["br"])) + soup.select(".ltx_tag"):
        t.insert_after(NavigableString(" "))
    return log


def text_of(el):
    return clean(el.get_text(""))


def heading_title(h):
    for t in h.select(".ltx_tag"):
        t.extract()
    return text_of(h)


def selftest():
    def m(x):
        html = f"<math><semantics>{x}<annotation>t</annotation></semantics></math>"
        return mml(BeautifulSoup(html, "lxml").math)
    assert m("<msup><mn>10</mn><mn>5</mn></msup>") == "10^5", m("<msup><mn>10</mn><mn>5</mn></msup>")
    assert m("<msub><mi>M</mi><mo>\u2299</mo></msub>") == "M_\u2299"
    assert m("<msup><mi>x</mi><mrow><mo>\u2212</mo><mn>3</mn></mrow></msup>") == "x^{\u22123}"
    assert m("<mfrac><mn>7</mn><mn>6</mn></mfrac>") == "(7/6)"
    assert m('<mn> 10</mn><mo lspace="0.170em" rspace="0em">\u200b</mo>'
             '<msub><mtext>M</mtext><mtext>Jup</mtext></msub>').strip() == "10 M_{Jup}"
    assert m('<mo fence="true">OPEN</mo><msub><mi>i</mi><mi>M</mi></msub>'
             '<mo>=</mo><mn>0</mn><mo>)</mo>') == "i_M=0)"
    assert clean("44.31^\u2218 N, 5^{\u00b0}") == "44.31\u00b0 N, 5\u00b0"
    t = BeautifulSoup('<p>a <math alttext="x"><semantics><mi>x</mi><annotation>x</annotation></semantics></math> b</p>', "lxml")
    prep(t)
    assert not t.find_all("math") and clean(t.get_text("")) == "a x b", clean(t.get_text(""))
    assert clean("(u,\\varv) and \\uppartialV at 4 to 14 \\upmum") == "(u,v) and \u2202V at 4 to 14 \u03bcm"
    assert clean("\\upomega and \\upsigma at 2\\farcs and \\earth") == "\u03c9 and \u03c3 at 2\u2033 and \u2295"
    print("selftest ok")


def demo(ids):
    out = []
    for pid in ids:
        soup = BeautifulSoup((HTML / f"{pid}.html").read_text(encoding="utf-8", errors="replace"), "lxml")
        log = prep(soup)
        out.append(f"\n===== {pid} =====")
        out.append(f"counts: {log['count']}")
        s = log["samples"]
        out.append("math (tex -> text), only ones with ^ _ frac sqrt:")
        rich = [(a, b) for a, b in s["math"] if re.search(r"\^|_|frac|sqrt", a)][:10]
        out += [f"   {a[:50]!r} -> {b[:50]!r}" for a, b in rich] or ["   none"]
        out.append(f"cites dropped: {s['cite_drop'][:5]}")
        out.append(f"cites kept   : {s['cite_keep'][:5]}")
        para = soup.select_one("section div.ltx_para") or soup.select_one("div.ltx_para")
        out.append("first body para: " + (text_of(para)[:700] if para else "NONE"))
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} lines)")


if __name__ == "__main__":
    selftest()
    demo(sys.argv[1:] or DEMO)
