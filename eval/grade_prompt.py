"""T19 grading prompts P1 and P2 (D066).

Both share DEFINITIONS, GUIDANCE and the input block verbatim. They differ only in the
framing and output sentences and in the order of the three worked examples. The examples
use an invented question outside the corpus domain, so they carry no eval content.
Hashes are over the templates (placeholders unfilled) and must not change once a
grading call has been made.

  python -m eval.grade_prompt     print hashes and both templates
"""
import hashlib
import json
import re

DEFINITIONS = """Grades:
0 = the passage does not address the question.
1 = the passage partially answers the question, or gives background needed for the answer without giving the answer itself.
2 = the passage answers the question."""

GUIDANCE = """Rules:
- Grade the passage text on its own. Do not fill gaps with your own knowledge: ask whether a reader of this passage would learn the answer.
- The passage comes from a research paper and may be technical. Technical wording is fine if the content answers the question.
- The paper title and section heading tell you what the passage is about; the grade is for the passage text.
- If the question asks several things, grade 2 needs the main thing asked."""

EXAMPLE_Q = "Why does the sky look blue during the day?"
EXAMPLES = {
    2: ("Sunlight is scattered by molecules in the atmosphere, and short (blue) wavelengths are "
        "scattered far more strongly than long (red) ones. Light reaching the eye from every part "
        "of the sky is therefore dominated by blue."),
    1: ("Rayleigh scattering by atmospheric molecules has a cross-section proportional to the inverse "
        "fourth power of wavelength. We use this dependence to correct our photometry for "
        "atmospheric extinction."),
    0: ("The sky-brightness monitor was installed in 2019 and records the zenith brightness every "
        "five minutes in the V band."),
}


def _examples(order):
    lines = [f'Example question: "{EXAMPLE_Q}"']
    for g in order:
        lines.append(f'Passage: "{EXAMPLES[g]}"\nGrade: {g}')
    return "\n\n".join(lines)


INPUT = """Question: {question}
Paper title: {title}
Section: {section}
Passage:
{passage}"""

P1 = ("You are judging search results for a test collection. Read the question and the retrieved "
      "passage below and give the passage one relevance grade.\n\n"
      + DEFINITIONS + "\n\n" + GUIDANCE + "\n\n" + _examples([2, 1, 0]) + "\n\n---\n\n" + INPUT
      + '\n\n---\n\nReply with only a JSON object containing the grade, for example {"grade": 1}.')

P2 = ("A learner typed the question below into a search engine over astronomy research papers, "
      "and the engine returned the passage below. Decide how well that passage answers the "
      "learner's question.\n\n"
      + DEFINITIONS + "\n\n" + GUIDANCE + "\n\n" + _examples([0, 1, 2]) + "\n\n---\n\n" + INPUT
      + '\n\n---\n\nAnswer with a JSON object of the form {"grade": 0}, {"grade": 1} or {"grade": 2} '
        "and write nothing else.")

PROMPTS = {"p1": P1, "p2": P2}
PROMPT_SHA256 = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in PROMPTS.items()}


def build(version, question, title, section, passage):
    """Fill a template. str.replace, not str.format, because the templates contain JSON braces."""
    return (PROMPTS[version].replace("{question}", question).replace("{title}", title)
            .replace("{section}", section or "(none)").replace("{passage}", passage))


def parse(raw):
    """Return 0, 1 or 2, or None if the reply is not exactly one JSON grade."""
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        return None
    if not isinstance(d, dict) or set(d) != {"grade"}:
        return None
    g = d["grade"]
    return g if g in (0, 1, 2) and not isinstance(g, bool) else None


if __name__ == "__main__":
    assert P1.count("{question}") == P2.count("{question}") == 1
    assert DEFINITIONS in P1 and DEFINITIONS in P2 and GUIDANCE in P1 and GUIDANCE in P2
    assert [parse(x) for x in ('{"grade": 2}', '```json\n{"grade": 0}\n```', '{"grade": 3}',
                               '{"grade": true}', 'grade 2', '{"grade": 1, "why": "x"}')] == [2, 0, None, None, None, None]
    for k, v in PROMPTS.items():
        print(f"===== {k}  sha256 {PROMPT_SHA256[k]}\n{v}\n")
