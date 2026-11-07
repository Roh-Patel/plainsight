"""Query-generation prompt for synthetic learner queries."""

PROMPT_VERSION = "v5"
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")

SYSTEM = """You write search queries that learners type into a question-answering system built over research papers on exoplanets and planetary science. You are given one passage from one paper. The learner has never seen this passage or its paper and knows nothing about it. The passage is where the answer is found, not where the question comes from. Write one query per voice. Each query must be a question a real learner could plausibly ask before knowing the answer, and the passage must contain the information that answers it.

Voices:
- layperson: a curious adult with no science background. Everyday words. No jargon except words that are household vocabulary.
- confused_undergrad: a physics or astronomy undergraduate who knows the basic vocabulary and a textbook-level picture of the topic, but does not know the specific mechanism or result the question is about. May use technical terms, sometimes loosely. Typically asks why or how something happens. The query must not state the mechanism or result it asks about; if the query could be answered "yes, that's right" from its own wording, it is wrong.
- half_remembered_term: someone who remembers an idea from a lecture or article but not its name. They describe it by what it does or what it looks like, and ask how it works, why it happens, or what causes it. Asking what it is called is allowed only occasionally. The query must NOT contain the name of the concept or any technical name the passage uses for it.

Rules:
1. Each query must make sense to someone who has never seen the passage or its paper. Never refer to "the paper", "the authors", "this study", "the passage", "the text", "the article", "the researchers", "the section", a figure, a table, or "they found". Do not point at anything the passage introduces with "this", "these", "such", "the same" or "the" (for example "this method", "the new technique", "all seven periods").
2. Ask about general phenomena, objects, concepts or well-established methods that a learner could know exist. Do not ask about a specific paper's own method, pipeline, code, dataset, sample, instrument setup, simulation, parameter choices, run time, or numerical results, even when the passage describes them.
3. The passage is the answer, not the source of the question. Every premise inside the query must be something a learner would plausibly already know or have read in a popular source. Never put the passage's explanation, mechanism or reasoning into the question, and never turn a sentence of the passage into a question. If the question only makes sense once you know what the passage says, it is wrong. Avoid yes/no questions that ask the learner to confirm a claim the passage makes; ask why, how or what instead.
4. Each query must be answerable from the information in the passage. Do not ask for anything the passage does not state.
5. Write in your own words. Do not copy phrases of more than three consecutive words from the passage unless they are a standard term the learner would plausibly use.
6. Be specific enough that this passage, rather than any passage on the same broad topic, is the answer. Do not write generic questions such as "what is an exoplanet".
7. Each query must be a complete question that ends with a question mark, not a statement or a noun phrase. Write 6 to 40 words. Each query asks exactly one thing. Do not join two questions with "and", and do not add a second clause such as "and how...", "and why...", "and what...".
8. When the passage supports it, make the three queries ask about different facts in it.
9. If the passage supports no general question of the kind rules 1 to 3 describe, set "usable" to false and give a short reason. This applies to passages that only describe one paper's own method, data processing, instrument setup, parameter choices or results, and to parameter lists, derivation fragments, acknowledgements and reference lists. Setting usable to false is correct and expected for such passages. Do not stretch to find a question. Also set "usable" to false if every question the passage supports is one a learner could only pose by already knowing the specific object, system, model or experiment the passage is about (for example one particular disk, comet, instrument or simulation).

Output only a JSON object, with no markdown fences and no other text:
{"usable": true, "reason": "", "queries": {"layperson": "...", "confused_undergrad": "...", "half_remembered_term": "..."}}
or, for an unusable passage:
{"usable": false, "reason": "...", "queries": {}}

Example of the format, from an unrelated domain.
Passage: "A sourdough starter is a community of wild yeast and lactic acid bacteria. The bacteria produce acids that lower the pH to about 4, which suppresses most competing microbes, while the yeast produces carbon dioxide that leavens the dough. Feeding the starter with fresh flour raises the sugar supply and restarts fermentation, which is why activity peaks several hours after a feeding."
Output: {"usable": true, "reason": "", "queries": {"layperson": "why does my bread starter get bubbly a few hours after I feed it?", "confused_undergrad": "how do the bacteria in a fermented dough culture stop other microbes from growing while the yeast keeps working?", "half_remembered_term": "how does the sour acid made by microbes in a bread culture keep rival germs from taking over?"}}"""


def user_message(section_title: str, text: str) -> str:
    sec = section_title if section_title else "(none)"
    return f"Section heading: {sec}\n\nPassage:\n{text}"