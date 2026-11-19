# Plainsight: project plan

## Purpose

RAG system over exoplanet and planetary-science papers. A fine-tuned retriever and reranker bridge the gap between how learners phrase questions and how papers are written. AI engineer portfolio project. The headline result is the measured gain from fine-tuning, shown on a hand-checked benchmark.

## Fixed constraints

- Windows, Python 3.11, one 12 GB GPU (RTX 5070). Commands in these docs use Command Prompt (cmd) syntax.
- Evaluation rigor comes first. Never cut the evaluation. Pre-declare metrics and selection rules. Never tune on the 133 eval questions.
- No deadlines (D059). No task is dropped, shortened or reordered for time. One step at a time, each with a command that verifies it.
- Raw full text stays local. The public demo shows short snippets with arXiv links. Never redistribute full text (D006).

## Concept

- Corpus: arXiv astro-ph.EP papers from 2024 onward, exoplanet and planetary topics, about 3,000 papers with arXiv HTML.
- Learner: curious layperson through undergrad.
- Core claim: a retriever fine-tuned on learner-style questions beats off-the-shelf retrieval, especially when the question shares little vocabulary with the gold passage.
- The learning layer (level-aware answers, citations, glossary, go-deeper, abstention) is a thin product on top. It must not swallow the retrieval and evaluation work.

## Design

### 1. Ingestion
- Metadata from the arXiv snapshot (2026-10-18). Full text from arXiv HTML. Skip papers with no HTML.
- Requests at least 3 s apart, cached locally.
- BeautifulSoup parse: keep section titles and body text, drop references and boilerplate, inline math to plain text.

### 2. Passages and splits
- Passages up to 400 tokens, overlap within a section only. Title and section heading prepended before embedding (`embed_text`).
- Split 70/10/20 by paper, seed 0, frozen (D018).

### 3. Training data
- 30,815 train-split passages. Claude Haiku 5.5 (Batch API) writes three learner-style questions each: layperson, confused undergrad, half-remembered term (D037, D042, D043).
- Filters: form failures, duplicate text, closeness to eval questions. No rank-based drop (D045 to D047).
- Hard negatives: hybrid_base ranks 10 to 50, train papers only, excluding the source passage, its paper, near-duplicate query sources and candidates the base reranker scores above -6 (D048 to D051).

### 4. Systems

| # | Run name | What it is |
|---|----------|-----------|
| 1 | bm25 | bm25s over `embed_text` |
| 2 | dense_base | bge-base-en-v1.5 in Qdrant |
| 3 | hybrid_base | RRF of 1 and 2 |
| 4 | hybrid_base_rerank_base | 3, then bge-reranker-base on the top 50 |
| 5 | hybrid_ft_embed | RRF of BM25 and the fine-tuned embedder |
| 6 | hybrid_ft_embed_rerank_ft | 5, then the fine-tuned reranker (T18) |
| side | dense_ft | fine-tuned embedder alone, to isolate its effect (D060) |

Main ladder is 1 to 6. Fine-tuning: sentence-transformers, cached MNRL, fp16 (embedder); plain PyTorch loop with a softmax group loss over the positive and up to 7 negatives (reranker, D063 to D065). Every run is tracked in Weights & Biases.

### 5. Evaluation
- Test set: 133 hand-written learner questions over 43 topics (D031 to D033). Frozen.
- Labels: top 10 of every system pooled per question (D036). Claude Sonnet 5.5 pre-grades each passage 0 to 2. Owner reviews every LLM-vs-pool disagreement and a 25% random sample. Report kappa.
- Metrics (D028): nDCG@10 is the single primary metric. Also Recall@10/20, MRR@10, paper-level hit. Report judged@10 and judged@20 per system.
- Statistics (D029): bootstrap CIs, paired tests with Holm correction over six pre-fixed comparisons.
- Generalisation: the index holds all papers. Report gold from held-out test papers separately from train papers (D034).
- Vocabulary-gap analysis: bucket queries by lexical overlap with the gold passage. Show where fine-tuning helps most.
- Failure analysis: acronyms, numeric questions, multi-paper answers.

### 6. Learning layer
- Level selector (layperson, undergrad, grad) changes the prompt, not the retrieval.
- Answer: short answer, "what the papers say" with inline citations to retrieved passages only, glossary from the retrieved text.
- Go deeper: three follow-up passages, preferring review-style papers for beginners.
- Abstention: if the top reranker score is below a threshold calibrated on val plus off-topic questions, answer "not covered".
- Stack: FastAPI, Gradio or HTMX, Qdrant, docker-compose.

### 7. Repo layout

```
plainsight/
  ingest/  retrieval/  data/  train/  eval/  app/  tests/  runs/  results/
```
`models/` and most of `data/` are local and gitignored.

### 8. Deliverables
- Fine-tuned models and the query dataset (queries and passage IDs only) on Hugging Face, with model cards.
- README leading with the ablation table and vocabulary-gap chart, then failure analysis, architecture, how to run. Inputs for it are collected in DECISIONS.md under "README inputs".
- Tests and CI (chunking, RRF, metrics, smoke eval).

## Current status

2026-11-19. T01 to T18 done: all seven systems have run files in `runs/`. Next is T19. Before any label exists, a new DECISIONS entry must replace D036's review rule (a fixed random sample of about 250 items for kappa, plus review of pairs where two Sonnet 5.5 gradings disagree; see Open questions). Then pool the top 10 of all seven systems per question.

## Tasks

### Phase 1: corpus, baselines, test questions (done)
- [x] T01 Project setup: repo, venv, Git/GitHub, W&B, folders
- [x] T02 Candidate astro-ph.EP papers from the metadata snapshot
- [x] T03 HTML fetcher, run to 3,009 papers
- [x] T04 HTML parser
- [x] T05 Chunker with unit tests
- [x] T06 Paper-level 70/10/20 split
- [x] T07 Indexes: BM25 and dense (Qdrant via Docker)
- [x] T08 Hybrid (RRF) and off-the-shelf reranker
- [x] T09 Eval harness with tests
- [x] T10 133 learner questions over 43 topics (overshoot to absorb T20 attrition, D032)
- [x] T11 Systems 1 to 4 run on the questions

### Phase 2: training data, fine-tuning, evaluation
Done when: the seven-system ablation table exists with confidence intervals and hand-checked labels.

- [x] T12 Query-generation prompt tested on 50 passages
- [x] T13 Synthetic queries for 30,815 train passages
- [x] T14 Query filters
- [x] T15 Hard negatives
- [x] T16 Embedder fine-tune (emb-v1, checkpoint-810)
- [x] T17 Fine-tuned index, runs for dense_ft and hybrid_ft_embed
- [x] T18 Fine-tune the reranker on the D050 negatives (D061; the LLM-judge filter failed validation; rr-v1 epoch-1 selected)
- [ ] T19 Pool the top 10 of all seven systems per question, Sonnet 5.5 pre-grades 0 to 2
- [ ] T20 Review disagreements plus a 25% sample by hand, record kappa
- [ ] T21 Full ablation with bootstrap CIs and paired tests
- [ ] T22 Held-out-vs-train split, vocabulary-gap analysis, charts

### Phase 3: learning app, polish, release
Done when: a clean clone runs with one docker-compose command and the models and write-up are public.

- [ ] T23 /ask API (retrieve, rerank, generate from retrieved passages only)
- [ ] T24 Level-aware answer prompt (citations, glossary)
- [ ] T25 Go-deeper suggestions (review-style papers for beginners)
- [ ] T26 Abstention threshold calibrated on val plus off-topic questions
- [ ] T27 Frontend
- [ ] T28 docker-compose, cold start from a fresh clone
- [ ] T29 Tests and GitHub Actions CI
- [ ] T30 Failure analysis
- [ ] T31 Publish models and query dataset to Hugging Face with model cards
- [ ] T32 README
- [ ] T33 CV bullet from own results only

## CV bullet template

"Fine-tuned an embedding model and cross-encoder reranker on [N] synthetic learner-style queries over [N] exoplanet papers, improving nDCG@10 from [X] to [Y] on a hand-labelled benchmark ([N] queries, [kappa] annotator agreement), with the largest gains on low-lexical-overlap questions. Released models and benchmark on Hugging Face."

Fill the brackets only with numbers your own results produced.
