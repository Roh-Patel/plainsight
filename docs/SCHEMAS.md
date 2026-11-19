# Plainsight: data schemas

Contract between components. Implemented through T18 unless marked (planned). When code diverges from a schema, update this file the same day.

## Conventions

- Files are UTF-8 JSONL (one JSON object per line) unless stated otherwise.
- `paper_id`: arXiv ID without version, e.g. `2401.01234`.
- `passage_id`: `{paper_id}::{section_idx:03d}::{chunk_idx:03d}`, e.g. `2401.01234::003::002`.
- `qid`: `eval-0001` (hand-written), `syn-000001` (synthetic), `val-000001` (val dev).
- `point_id`: line index in `passages.jsonl`. It is also the Qdrant point ID and the row index of every index file.
- Paths are relative to the repo root. Raw data stays local and is never committed.
- Index and rank files carry `passages_lines` and `passages_sha256` in their manifests. Code must compare them to `passages.jsonl` at load time and refuse to run on a mismatch.

## Layout

```
data/
  raw/html/{paper_id}.html
  raw/fetch_log.jsonl
  processed/
    candidates.jsonl  parsed.jsonl  passages.jsonl  splits.json
    papers.jsonl                      (planned, not built)
    parse_log.jsonl  chunk_drops.jsonl   audit logs, not contracts
    index/dense_base.npy|.manifest.json   dense_ft.npy|.manifest.json
    index/bm25/   index/qdrant_{collection}.manifest.json
  synthetic/
    gen_sample.jsonl|.manifest.json  batches.jsonl  raw/  gen_log.jsonl (audit)
    queries.jsonl  eval_overlap.jsonl
    ranks/ranks.jsonl|.manifest.json  ranks/{system}_top100.npy
    hn_scored.jsonl|.manifest.json  hard_negatives.jsonl|.manifest.json
    train_triples.jsonl|.manifest.json
    synthetic_val/  gen_sample.jsonl|.manifest.json  batches.jsonl  raw/  gen_log.jsonl
                  queries.jsonl  dev/{tag}.json
                  rr_dev_candidates.jsonl|.manifest.json  rr_dev/{tag}.json
  eval/questions.jsonl  questions_draft.txt  topics.txt  topics_draft.txt  qrels.tsv (planned)
eval/label_audit.jsonl (planned)
runs/{system}.jsonl
results/{experiment}.json  results/per_query/{system}.jsonl
models/embedder/{tag}/checkpoint-{step}/ and final/   (gitignored)
models/reranker/{tag}/epoch-{n}/                      (gitignored)
```

## Ingestion

### raw/fetch_log.jsonl
Append-only, one row per fetch attempt. The last row per paper wins. `http` is an integer for ok, no_html and bad_html, and a string for error rows (e.g. "http 503", "ReadTimeout", "reopen").

- `ok`: HTTP 200 and at least 3 `ltx_para` elements (`is_article` in ingest/fetch_html.py). File saved to raw/html.
- `no_html`: HTTP 404.
- `bad_html`: HTTP 200 with fewer than 3 `ltx_para` (build-failure stub). Says nothing about section structure.
- `error`: timeout, 5xx, other failure, or a manual reopen row. Retried on the next run.

```json
{"paper_id": "2401.01234", "status": "ok", "http": 200, "bytes": 412345, "ts": "2026-10-20T14:03:11"}
```

### processed/candidates.jsonl
One row per snapshot paper with astro-ph.EP in its categories and a v1 date of 2024-01-01 or later. Written by T02, read by T03. Source of title and abstract for everything downstream (D011).

```json
{"paper_id": "2401.01234", "title": "...", "authors": ["..."], "published": "2024-01-03", "arxiv_categories": ["astro-ph.EP", "astro-ph.SR"], "abstract": "..."}
```

### processed/papers.jsonl (planned, not built)
One row per paper kept after parsing. `is_review` has no data source yet (needed at T25).

```json
{"paper_id": "2401.01234", "title": "...", "authors": ["..."], "published": "2024-01-03", "arxiv_categories": ["astro-ph.EP"], "abstract": "...", "html_path": "data/raw/html/2401.01234.html", "is_review": false, "n_passages": 41}
```

### processed/parsed.jsonl
Written by T04 (`ingest/parse_html.py`), read by T05. One row per parseable paper (3,008). No title or abstract. A unit is one section's own text, with nested sections split off as separate units.

```json
{"paper_id": "2401.03437", "units": [{"idx": 3, "title": "Observations > MATISSE data > First look at the data", "text": "para one\n\npara two"}], "n_chars": 89524}
```

- `idx`: position in document order from 0.
- `title`: breadcrumb of stripped headings joined by " > ", "Appendix: " prefix for appendices, `""` outside any heading and for flat papers.
- `text`: paragraphs joined by `\n\n`. Units reach 63,000 characters.
- Inline math becomes plain text (`10^5`, `M_⊙`). Display equations, footnotes, parenthetical citations, figures, tables and captions are removed (D012).

### processed/passages.jsonl
```json
{"passage_id": "2401.01234::003::002", "paper_id": "2401.01234", "section_title": "Atmospheric retrieval", "section_idx": 3, "chunk_idx": 2, "text": "...", "embed_text": "<paper title> | <section title> | <text>", "n_tokens": 312}
```

- `embed_text` is exactly what gets embedded. `text` is what gets shown, cited and fed to the reranker.
- `n_tokens` counts `text` only (bge-base-en-v1.5 tokenizer, no special tokens). Maximum 400.
- `section_idx` is the unit `idx` in parsed.jsonl. When `section_title == ""` the middle segment of `embed_text` is omitted.
- `chunk_idx` is assigned before junk filtering, so dropped chunks leave gaps and IDs stay stable.

### processed/splits.json
```json
{"seed": 0, "ratios": {"train": 0.7, "val": 0.1, "test": 0.2}, "train": ["paper_id"], "val": ["..."], "test": ["..."]}
```
Split by paper, never by passage. Covers the 3,008 papers with at least one passage. Sorted ID lists. Written by `ingest/split.py`, frozen (D018). Tracked in git; everything else in data/processed is ignored.

## Indexes

### processed/index/{name}.npy and {name}.manifest.json
Written by `ingest/dense.py` (T07, T17). Row `i` is line `i` of `passages.jsonl`. `--limit` runs write `{name}_limit{N}.*` and must never be indexed.

- Shape `(n_passages, 768)`, float16, L2-normalised in fp32 then stored as fp16 (norms 1.0 ± 0.0002, so re-normalise in fp32 before scoring, D027).
- Passages are embedded from `embed_text` with no prefix. Queries use `query_prefix` from the manifest.

```json
{"model": "BAAI/bge-base-en-v1.5", "dtype": "float16", "normalized": true, "n": 116356, "dim": 768, "limit": 0, "text_field": "embed_text", "passages_lines": 116356, "passages_sha256": "...", "query_prefix": "Represent this sentence for searching relevant passages: ", "max_seq_length": 512, "sentence_transformers": "6.2.0", "torch": "2.11.0+cu128", "seconds": 229.0, "created": "2026-10-26T15:00:00"}
```
For dense_ft, `model` is `models/embedder/emb-v1/checkpoint-810`.

### processed/index/bm25/
bm25s index saved with `BM25.save`, plus `manifest.json`. Row `i` is line `i` of `passages.jsonl`. Indexed field is `embed_text`.

```json
{"text_field": "embed_text", "tokenizer": "bm25s default regex", "stopwords": "en", "stemmer": "PyStemmer english", "method": "lucene", "k1": 1.5, "b": 0.75, "bm25s": "0.3.15", "passages_lines": 116356, "passages_sha256": "...", "seconds": 12.6}
```

### processed/index/qdrant_{collection}.manifest.json
Collections `passages_dense_base` and `passages_dense_ft`. Payload is `passage_id` and `paper_id` only, no text. Cosine distance. Eval runs search with `exact=True`.

```json
{"collection": "passages_dense_base", "emb_name": "dense_base", "n_points": 116356, "distance": "cosine", "id_scheme": "line index in passages.jsonl", "payload": ["passage_id", "paper_id"], "passages_lines": 116356, "passages_sha256": "...", "seconds": 268.1}
```

## Synthetic training data

### synthetic/gen_sample.jsonl and manifest
Frozen sample of train-split passages sent for query generation, in global shuffled order. `train/draw_gen_sample.py` refuses to overwrite without `--force`; never use it once a batch is submitted.

```json
{"order": 0, "point_id": 82889, "passage_id": "2401.01234::003::002", "paper_id": "2401.01234", "shard": 0}
```
`shard`: 0 is the 500-passage pilot, 1 to 6 hold 5,000 each, 7 holds 315 (30,815 total).

Manifest keys: `seed` (14), `cap_per_paper` (15), `min_tokens` (100), `min_function_word_fraction` (0.15), `pilot_size`, `shard_size`, `excluded_files`, `n_excluded` (100), `n`, `n_papers` (2,106), `shard_sizes` (all shards 0 to 7), `passages_lines`, `passages_sha256`, `created`. `train/gen_batch.py` checks `passages_sha256` against `passages.jsonl` and the dense manifest before every submit.

### synthetic/batches.jsonl
Append-only, one row per submitted shard. A shard with a row is never resubmitted.

```json
{"shard": 0, "batch_id": "msgbatch_...", "n_requests": 500, "prompt_version": "v5", "prompt_sha256": "...", "model": "claude-haiku-5-5", "max_tokens": 4000, "passages_sha256": "...", "created": "2026-11-08T00:00:00"}
```

### synthetic/raw/
Gitignored. The API serves batch results for 29 days and sampling is not reproducible (D037), so these files are the only record. Never delete `shard_{N}.api.jsonl`.

- `shard_{N}.api.jsonl`: verbatim API result per request, saved before parsing. `custom_id` is `p` plus the `point_id` zero-padded to 6 digits (e.g. `p082889`).
- `shard_{N}.jsonl`: one parsed row per request in `gen_sample.jsonl` order: `passage_id`, `prompt_version`, `model`, `sampling`, `stop_reason`, `input_tokens`, `output_tokens`, `raw`, `parsed`, `parse_error`, `max_tokens`, `thinking_tokens`, `custom_id`, `batch_id`, `shard`, `result_type`.
- `shard_{N}.gate.json`: automatic gate result (`passed`, counts, `config` = prompt version, model, `max_tokens`, prompt hash). Shards 1 to 7 could only be submitted after shard 0 passed with an identical config.

### synthetic/queries.jsonl
Written by `train/assemble_queries.py` (T13), then `kept` and `drop_reason` set by `train/apply_filters.py` (T14). 73,767 rows: three per `ok` passage in the order layperson, confused_undergrad, half_remembered_term, passages in `gen_sample.jsonl` order. 69,317 kept, 4,450 dropped. Dropped rows stay in the file. `drop_reason` is the first rule that fired: `banned_phrase`, `no_question_mark`, `length`, `compound_clause`, `deictic`, `shared_number`, `duplicate_text`, `eval_overlap` (D046, D047). `assemble_queries --verify` now reports False because T14 changed `kept`.

```json
{"qid": "syn-000001", "source_passage_id": "2401.01234::003::002", "voice": "layperson", "query": "...", "generator": "claude-haiku-5-5", "kept": true, "drop_reason": null}
```

### synthetic/ranks/
Written by `train/rank_queries.py` (T14). Refuses to overwrite a non-empty directory without `--force`. `--limit` runs may not write here.

- `ranks.jsonl`: one row per query in `queries.jsonl` order (73,767, kept and dropped). Ranks are 1-based in each system's top 100, `null` if absent. `sec_rank` is the best rank of any passage from the same paper and section as the source (includes the source). Systems: `bm25`, `dense_base`, `hybrid_base` (RRF as D023).
- `{system}_top100.npy`: shape `(73767, 100)`, int32 point IDs, best first, padded with -1. Row `i` is line `i` of `queries.jsonl`.
- `ranks.manifest.json`: `n`, `k` (100), `rrf_k` (60), `systems`, `dense_backend` ("numpy"), `dense_batch_size` (128), `chunk` (4096), `limit`, `rank_base` (1), `qids_sha256` (hash of qids joined by newlines in file order), `passages_sha256`, `passages_lines`, `seconds`, `created`.

```json
{"qid": "syn-000001", "voice": "layperson", "src_point_id": 82889, "bm25_src_rank": null, "bm25_sec_rank": 12, "dense_base_src_rank": 31, "dense_base_sec_rank": 31, "hybrid_base_src_rank": 24, "hybrid_base_sec_rank": 24}
```

### synthetic/eval_overlap.jsonl
Written by `train/apply_filters.py --apply` (T14). One row per eval question (133). dense_base cosine with the query prefix. T22 uses it to report results for the exposed questions (D047).

```json
{"qid": "eval-0008", "max_sim": 0.941, "nearest_qid": "syn-051229", "n_ge_threshold": 11, "threshold": 0.85}
```
`max_sim` is the highest similarity to any synthetic query before filtering. `n_ge_threshold` counts synthetic queries at or above `threshold`.

### synthetic/hn_scored.jsonl
Written by `train/score_negatives.py` (T15). Gitignored. One row per kept query (69,317) in `queries.jsonl` order. The candidate pool before the score threshold and near-duplicate rules, with a base-reranker logit for every candidate and for the source passage. Resumable, never overwrites a complete file.

```json
{"qid": "syn-000001", "pos_ce_score": -3.12, "cands": [[12345, 12, -7.41], [67890, 13, -5.02]]}
```

- `cands`: `[point_id, baseline_rank, ce_score]`, best rank first. `baseline_rank` is the 1-based hybrid_base rank in the full index, 10 to 50.
- Pool: train-split papers only, source passage and source paper excluded. 1,913,454 candidates; 26 queries have none.
- Scores are raw bge-reranker-base logits (D024, D026), batch size 32.
- Manifest: pool rule, model, batch size, `passages_lines`, `passages_sha256`, `qids_sha256` (from the ranks manifest), `created`.

### synthetic/hard_negatives.jsonl
Written by `train/build_hard_negatives.py` (T15). Gitignored. One row per kept query in `queries.jsonl` order. Built from `hn_scored.jsonl` by dropping candidates with `ce_score` above -6 and candidates that are the source passage of another query at dense_base query-to-query cosine of 0.80 or more (D048, D050). Never overwrites an existing file. Used by T16 and T18 (D052, D061).

```json
{"qid": "syn-000001", "pos_ce_score": -3.12, "negatives": [{"passage_id": "2401.04567::002::001", "baseline_rank": 12, "ce_score": -7.41}]}
```

- `negatives` are in baseline-rank order, 0 to 41 per query. About 4,900 queries have none (3.0% / 15.6% / 2.6% of layperson / confused_undergrad / half_remembered_term). Queries are never dropped for this.
- Every negative is a train-split passage from a different paper than the source.
- Manifest: `n_queries`, `n_negatives` (764,375), `pool`, `ce_threshold` (-6.0), `ce_rule`, `near_dup_tau` (0.80), `near_dup_rule`, `scored_file_sha256`, `passages_lines`, `passages_sha256`, `created`.
- `python -m train.build_hard_negatives --check` validates order, train membership, source and source-paper exclusion, rank range, score cap, duplicates.

### synthetic/train_triples.jsonl
Written by `train/build_train_triples.py` (T16). Gitignored (holds query text). One row per kept query in `queries.jsonl` order. IDs only: the trainer joins `embed_text` from `passages.jsonl`. Never overwrites an existing file.

```json
{"qid": "syn-000001", "query": "...", "positive_passage_id": "2401.01234::003::002", "negative_passage_ids": ["2401.04567::002::001"]}
```

- `query` is stored without the prefix. The trainer adds QUERY_PREFIX (D019).
- `negative_passage_ids` has 0 or 1 entries: one drawn uniformly from the query's retained negatives, seed 24 (D055). Empty for 4,874 queries (714 / 3,573 / 587 by voice).
- Manifest: `n`, `n_with_negative`, `n_without_negative`, `k`, `seed`, `rule`, `hard_negatives_sha256`, `passages_sha256`, `created`.

## Val dev set

### synthetic_val/ (T16)
Same pipeline as `synthetic/`, run once on val-split passages (D053). Written by `train/draw_val_sample.py`, `gen_batch_val.py`, `assemble_val.py`, which refuse to overwrite. `raw/` and the query files are gitignored. Never delete `raw/shard_0.api.jsonl`.

- `gen_sample.jsonl`: same format as the train one, `shard` always 0. Manifest keys: `split`, `seed` (18), `cap_per_paper` (10), `min_tokens`, `min_function_word_fraction`, `n_val_papers`, `n_eligible`, `n`, `n_papers`, `shard_sizes`, `passages_lines`, `passages_sha256`, `created`.
- `batches.jsonl`, `raw/shard_0.*`, `gen_log.jsonl`: same formats as `synthetic/` (gen_log has no `shard` field; 2,973 rows).
- `queries.jsonl`: same format, `qid` `val-000001` to `val-006996`. `kept` and `drop_reason` set at assembly with the D046 form rules plus exact duplicate text within val. No eval-overlap rule. 6,524 kept.

### synthetic_val/dev/{tag}.json
Written by `train/dev_eval.py`, one file per evaluated model, never overwritten. Corpus is all 11,536 val passages, queries are the kept val queries, one labelled passage per query (its source, D054).

```json
{"tag": "base", "model": "BAAI/bge-base-en-v1.5", "n_passages": 11536, "n_queries": 6524, "passages_sha256": "...", "metrics": {"all": {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@10": 0.0, "recall@50": 0.0}}, "qids": ["val-000001"], "ranks": [1]}
```
`metrics` keys: `all`, `layperson`, `confused_undergrad`, `half_remembered_term`. `ranks` is the 1-based source rank aligned with `qids`; ties favour the source. Dev numbers select a checkpoint and are never quoted as results.

### models/embedder/{tag}/
Written by `train/train_embedder.py`. Sentence-transformers folders `checkpoint-{step}` (every 135 steps, weights only) and `final` (copy of the last). Refuses to write into a non-empty folder. `emb-v1` holds steps 135 to 810; checkpoint-810 is selected (D058).

### synthetic_val/rr_dev_candidates.jsonl and manifest
Written by `train/build_rr_dev.py --write`, refuses to overwrite. Gitignored (holds query text). One row per kept val query (6,524) in `synthetic_val/queries.jsonl` order. Candidates are global `point_id`s (line index in `passages.jsonl`), so they go straight into `Reranker.rerank`.

```json
{"qid": "val-000001", "voice": "layperson", "query": "...", "src_point_id": 82889, "src_rank": 12, "cands": [123, 456]}
```

- `cands`: top 50 of RRF (D023, k = 60) over a val-only BM25 index (D020 settings) and dense_ft cosine over the val rows, best first. `src_rank` is the 1-based rank of the source passage in `cands`, `null` if absent.
- Manifest keys: `n`, `depth` (50), `rrf_k`, `bm25`, `dense_rows`, `embedder`, `passages_sha256`, `qids_sha256`, `created`.

### synthetic_val/rr_dev/{tag}.json
Written by `train/rr_dev_eval.py`, one file per evaluated reranker, never overwritten. Same lists for every model.

```json
{"tag": "base", "model": "base", "n_queries": 6524, "metrics": {"ALL": {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@10": 0.0, "in_top50": 0.0}}, "qids": ["val-000001"], "ranks": [1], "seconds": 0.0}
```
`metrics` keys: `ALL`, `layperson`, `confused_undergrad`, `half_remembered_term`. `ranks` is the 1-based source rank after reranking, `null` if the source was not in the top 50. Dev numbers select a checkpoint and are never quoted as results.

### models/reranker/{tag}/
Written by `train/train_reranker.py`. Hugging Face folders `epoch-{n}` (model weights and tokenizer files, loadable by `Reranker(model_name=...)`), one per epoch. Refuses to write into a non-empty folder. `rr-v1` holds epochs 1 to 3; epoch-1 is selected (D065). `--smoke N` writes to `models\smoke\rr`.

## Evaluation

### eval/questions.jsonl
133 hand-written questions over 43 topics (T10). Frozen (D033): never edited. LF line endings.

```json
{"qid": "eval-0001", "question": "...", "topic": "transit method", "style": "layperson", "tags": ["acronym"]}
```
`style`: `layperson`, `confused_undergrad` or `half_remembered_term`. `tags`: possibly empty list from `acronym`, `numeric`, `multi_paper`, set at write time before any retrieval output was seen. `topic` matches a row of `eval/topics.txt`.

### data/eval/topics.txt, topics_draft.txt, questions_draft.txt
`topics_draft.txt`: one row per topic, `topic | comma-separated title/abstract keywords`, read by `eval/topic_coverage.py` and `analysis/eval/topic_spotcheck.py`. `topics.txt`: one row per topic, `topic | family | n_questions` (43 topics, 12 families). `questions_draft.txt`: hand-written source, `topic | style | tags | question` (`-` for no tags). Read only by `eval/validate_questions.py`.

### eval/qrels.tsv (planned, T19/T20)
Tab-separated, TREC style, no header: `qid  0  passage_id  grade`. Grades: 0 not relevant, 1 partially relevant, 2 answers the question.

### eval/label_audit.jsonl (planned, T20)
Tracks label source separately from qrels. Path to be confirmed at T19.

```json
{"qid": "eval-0001", "passage_id": "...", "llm_grade": 2, "human_grade": 2, "reviewed": true}
```

### runs/{system}.jsonl
System names: `bm25`, `dense_base`, `dense_ft`, `hybrid_base`, `hybrid_base_rerank_base`, `hybrid_ft_embed`, `hybrid_ft_embed_rerank_ft`.

- bm25, dense_base, dense_ft, hybrid_base, hybrid_ft_embed hold the top 100. `score` is the BM25 score, cosine similarity (dense) or RRF score (hybrid).
- Reranked systems (`hybrid_base_rerank_base`, `hybrid_ft_embed_rerank_ft`) hold the top 50, reranked from the top 50 of the fused list (D024). `score` is the raw reranker logit. `hybrid_ft_embed_rerank_ft` uses `models/reranker/rr-v1/epoch-1`.
- Scores are comparable within a system only.

```json
{"qid": "eval-0001", "system": "hybrid_ft_embed", "results": [{"passage_id": "...", "score": 0.83}]}
```

### results/per_query/{system}.jsonl
Written by `eval/evaluate.py` (T09). One row per question per system. T21 and T22 aggregate from these. `min_grade` is the relevance threshold for recall, MRR and paper_hit (2 primary, 1 appendix variant). `judged@k` is the fraction of the top k with a label in qrels.tsv.

```json
{"qid": "eval-0001", "system": "bm25", "min_grade": 2, "ndcg@10": 0.0, "recall@10": 0.0, "recall@20": 0.0, "mrr@10": 0.0, "paper_hit@10": 0.0, "judged@10": 0.0, "judged@20": 0.0}
```

### results/{experiment}.json (planned, T21)
```json
{"experiment": "ablation-v1", "date": "2026-11-01", "n_questions": 133, "systems": {"bm25": {"ndcg@10": {"mean": 0.0, "ci95": [0.0, 0.0]}, "recall@10": {}, "recall@20": {}, "mrr@10": {}, "paper_hit@10": {}}}}
```

## Audit logs (not contracts)

- `processed/parse_log.jsonl`: one row per attempted paper (3,009), status `parsed`, `dropped` or `error`, with garble ratio, coverage and kept and dropped titles. A paper with garble above 0.01 is dropped on future runs.
- `processed/chunk_drops.jsonl`: one row per dropped unit or passage. `reason`: `unit_lt_50_tokens`, `missing_spaces`, `long_letter_runs`, `low_function_words`, `table_residue`, `comma_density`. `chunk_idx` is `null` for unit-level drops.
- `synthetic/gen_log.jsonl`: one row per sampled passage (30,815). `status`: `ok`, `unusable`, `refusal`, `parse_error`, `request_error`. Only `ok` produces queries.

## Contract-breaking changes

| Date | Change |
|------|--------|
| 2026-10-25 | `passage_id` section index widened from :02d to :03d (244 units above idx 99, max 175) |
| 2026-10-25 | `n_tokens` counts `text` only; `chunk_idx` assigned before filtering |
| 2026-10-27 | Reranked runs hold the top 50 with raw logits as `score` (D024) |
| 2026-10-31 | `eval/questions.jsonl` gained `tags` |
| 2026-11-09 | `synthetic/queries.jsonl`: T14 set `kept` and `drop_reason` |
| 2026-11-17 | Added system `dense_ft` to runs (D060) |
| 2026-11-19 | Added system `hybrid_ft_embed_rerank_ft` to runs (D065) |
