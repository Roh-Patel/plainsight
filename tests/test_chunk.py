from ingest.chunk import (build_atoms, hard_split_checked, junk_reason, pack,
                          process_paper, split_sentences)


class FakeTok:
    """Whitespace 'tokenizer' so tests need no model download."""

    def count(self, texts):
        return [len(t.split()) for t in texts]

    def hard_split(self, text, n):
        w = text.split()
        return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


def sent(i, words=10):
    return f"Sentence{i} " + " ".join(["the"] * (words - 1)) + "."


def test_split_keeps_abbreviations():
    t = ("Smith et al. (2020) found a planet, see Fig. 3 and Eq. (4). "
         "The mass is 3.5 Earth masses, i.e. small. J. Smith agrees. Is it real? Yes.")
    assert split_sentences(t) == [
        "Smith et al. (2020) found a planet, see Fig. 3 and Eq. (4).",
        "The mass is 3.5 Earth masses, i.e. small.",
        "J. Smith agrees.",
        "Is it real?",
        "Yes.",
    ]


def test_split_sect_abbreviation():
    assert split_sentences("Sect. 4.2 describes this. We then fit it.") == [
        "Sect. 4.2 describes this.", "We then fit it."]


def test_paragraph_that_fits_is_not_split():
    para = " ".join(sent(i) for i in range(9))
    atoms = build_atoms([para], FakeTok(), 100)
    assert pack(atoms, 100, 20, 2) == [(para, 90)]


def test_cap_and_first_chunk_clean():
    paras = [" ".join(sent(i) for i in range(j * 3, j * 3 + 3)) for j in range(6)]
    chunks = pack(build_atoms(paras, FakeTok(), 100), 100, 20, 2)
    assert len(chunks) == 3
    assert all(n <= 100 for _, n in chunks)
    assert chunks[0][0].startswith("Sentence0 ")
    assert all(n == len(t.split()) for t, n in chunks)


def test_overlap_is_tail_of_previous_chunk():
    paras = [" ".join(sent(i) for i in range(j * 3, j * 3 + 3)) for j in range(6)]
    chunks = pack(build_atoms(paras, FakeTok(), 100), 100, 20, 2)
    tail = split_sentences(chunks[0][0])[-2:]
    assert sum(len(s.split()) for s in tail) <= 20
    assert chunks[1][0].startswith(" ".join(tail))


def test_overlap_skipped_when_last_sentence_too_long():
    paras = [sent(i, 30) for i in range(3)]
    chunks = pack(build_atoms(paras, FakeTok(), 70), 70, 20, 2)
    assert chunks[0][0] == paras[0] + "\n\n" + paras[1]
    assert chunks[1][0] == paras[2]


def test_long_paragraph_splits_at_sentences():
    para = " ".join(sent(i) for i in range(25))
    chunks = pack(build_atoms([para], FakeTok(), 100), 100, 20, 2)
    assert len(chunks) >= 3
    assert all(n <= 100 for _, n in chunks)
    assert all(t.startswith("Sentence") and t.endswith(".") for t, _ in chunks)


def test_single_long_sentence_is_hard_split():
    words = ["Start"] + ["w"] * 248 + ["end."]
    text = " ".join(words)
    chunks = pack(build_atoms([text], FakeTok(), 100), 100, 20, 2)
    assert len(chunks) == 3
    assert all(n <= 100 for _, n in chunks)
    assert " ".join(t for t, _ in chunks).split() == words
    assert [c for _, c in hard_split_checked(text, FakeTok(), 100)] == [100, 100, 50]


def test_junk_filters():
    assert junk_reason("Word" * 60) == "missing_spaces"
    assert junk_reason(" ".join(sent(i) for i in range(5))) is None
    assert junk_reason(" ".join(["the"] * 30 + ["a" * 30] * 3)) == "long_letter_runs"
    assert junk_reason("Les étoiles T Tauri sont entourées de disques protoplanétaires " * 4) == "low_function_words"
    assert junk_reason(" ".join(f"Name{i} Surname{i}" for i in range(60))) == "low_function_words"
    table = " ".join(["the"] * 15 + ["1.5"] * 40 + ["Kepler"] * 45)
    assert junk_reason(table) == "table_residue"
    commas = ", ".join(["the"] * 10 + [f"Name{i}" for i in range(70)]) + "."
    assert junk_reason(commas) == "comma_density"


def test_math_commas_and_long_names_are_kept():
    math = ("We define the functions f(e,x),g(e,x),h(e,x),k(e,x) and f(e,x),g(e,x),h(e,x),k(e,x) "
            "and then we show that the result is the same for all of the cases that we consider "
            "in this section of the paper, where the values are given by the above.")
    assert junk_reason(math) is None
    prose = ("We present StellarSpectraObservationFitting.jl, an open-source package for obtaining "
             "precise radial velocities in the presence of noise and of the telluric lines.")
    assert junk_reason(prose) is None


def test_short_unit_dropped_and_embed_text_format():
    body = " ".join(sent(i) for i in range(6))
    units = [{"idx": 0, "title": "", "text": "2025"},
             {"idx": 1, "title": "Intro", "text": body},
             {"idx": 2, "title": "", "text": body}]
    passages, drops = process_paper("2401.00001", "Title", units, FakeTok())
    assert drops == [{"paper_id": "2401.00001", "section_idx": 0, "chunk_idx": None,
                      "reason": "unit_lt_50_tokens", "n_tokens": 1, "preview": "2025"}]
    assert [p["passage_id"] for p in passages] == ["2401.00001::001::000",
                                                   "2401.00001::002::000"]
    assert passages[0]["embed_text"].startswith("Title | Intro | Sentence0")
    assert passages[1]["embed_text"].startswith("Title | Sentence0")


def test_chunk_idx_assigned_before_filtering():
    a = " ".join(sent(i) for i in range(8))
    b = ", ".join(f"Name{i}" for i in range(80)) + "."
    c = " ".join(sent(i) for i in range(8, 16))
    units = [{"idx": 5, "title": "X", "text": "\n\n".join([a, b, c])}]
    passages, drops = process_paper("2401.00002", "T", units, FakeTok(),
                                    max_tokens=100, overlap_tokens=0)
    assert [p["chunk_idx"] for p in passages] == [0, 2]
    assert [(d["chunk_idx"], d["reason"]) for d in drops] == [(1, "low_function_words")]