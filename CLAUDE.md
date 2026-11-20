# Plainsight: notes for AI assistants

Operating instructions for Claude Code and similar tools. The project record itself lives in `docs/` and is written for human readers.

@docs/PLAN.md

## Where things are

- `docs/PLAN.md`: purpose, design, task list with status (imported above).
- `docs/DECISIONS.md`: every decision (Dnnn), pipeline commands, numbers by task, open questions. Large (about 25k tokens), so read the relevant section on demand rather than all of it. Grep for `**Dnnn**` or `**Tnn**`.
- `docs/SCHEMAS.md`: file formats and data contracts. Read before writing code that reads or writes any data file.
- `analysis/`: one-off studies that produced the evidence for specific decisions. Pipeline code must never import from it. New one-off surveys and hand-check samplers go here, not in the pipeline packages; delete pure debug printouts once a test covers them.

## Hard rules

- **Never tune on, or look at system output for, the 133 eval questions** (`data/eval/questions.jsonl`, frozen, D033). Dev sets in `data/synthetic_val/` exist for selection.
- **Declare before measuring.** Metrics, selection rules, thresholds and comparisons are written as a DECISIONS entry before the data that tests them is seen. Propose the entry to the user first.
- **No deadlines (D059).** Do not drop, shorten or reorder tasks for time.
- **Raw full text stays local (D006).** Never commit or upload passage text, queries or HTML.
- **`--force` is forbidden** on `draw_gen_sample`, `assemble_queries`, `apply_filters`, `rank_queries`, `draw_val_sample`, `assemble_val`, `make_runs`, `make_runs_ft`. Never rerun `make_runs_ft` or `make_runs_rerank_ft` once T19 starts (D035). Never rebuild `hard_negatives.jsonl` without a new decision.
- **Never delete** `data/synthetic/raw/shard_*.api.jsonl` or `data/synthetic_val/raw/shard_0.api.jsonl`. Batch sampling is not reproducible; these are the only record.
- **`passages.jsonl` is the root of trust.** Every index and manifest is tied to its SHA-256. Any run of `python -m ingest.chunk` (including `--verify` and `--limit N`) rewrites it. Do not run it.
- Ask before spending API money (Batch API runs) or starting long GPU jobs.

## Environment

- Windows 11, Python 3.11.1 in `.venv` (`.venv/Scripts/python.exe`), RTX 5070 12 GB. Run everything from the repo root as modules: `python -m pkg.module`.
- Commands given to the user to run themselves use **cmd syntax** (e.g. `models\reranker\...`, `set VAR=...`). Claude's own shell tools are bash/PowerShell; that is fine for Claude's own use.
- Tests: `.venv/Scripts/python.exe -m pytest tests -q` (58 pass as of 2026-11-19).
- Qdrant 1.20.0 runs in Docker as container `qdrant` on 127.0.0.1:6333. `ranx` is only in `.venv-xcheck`.
- W&B project `plainsight`. Secrets in `.env` (never read aloud or commit).

## Gotchas

- fp16 embeddings must be re-normalised in fp32 before scoring (D022). `qdrant_index --check` overlap compares un-renormalised rows; use `retrieval.check_ft_backends` instead.
- `DenseSearcher` defaults to `backend="numpy"`. Eval code must pass `backend="qdrant"` explicitly. `retrieval/dense.py` reads the query prefix from `manifest["query_prefix"]`, so grepping for `QUERY_PREFIX` there finds nothing.
- Importing `ingest.sparse` or `ingest.dense` pulls in torch (slow). `train.dup_report` and `train.apply_filters` load the model and matrix onto the GPU but need no Qdrant.
- Importing `train.gen_batch_val` redirects `train.gen_batch` paths to `data/synthetic_val`; `assemble_val` relies on that.
- Do not regenerate runs with a different batch composition (D035): dense scores drift about 1e-5 and flip near-ties.
- `assemble_queries --verify` reports False since T14 changed `kept` (expected). Pre-T14 copy: `scratch_synth/queries_pre_t14.jsonl`.
- Training needs an exclusive GPU. Peak allocated above 12 GB means Windows spilled into system RAM and timings are invalid. `train_embedder`, `train_reranker` and `dev_eval` have no resume logic and refuse non-empty output folders: move a failed `models/.../{tag}` aside by hand. `--smoke N` writes to `models/smoke`. `dev_eval --check-cache` is valid for the base model only.
- sentence-transformers: `batch_sampler` must be NO_DUPLICATES explicitly, `dataloader_drop_last` defaults to False, two datasets must be passed as a DatasetDict.
- `rr_dev_eval` prints nothing for 30 to 40 min. The "identical draw in epochs 0 and 1" line from `train.rr_groups` is mis-defined; never quote it.
- Anthropic API: reasoning tokens count against `max_tokens` and the thinking block comes back empty, so a small cap gives empty or truncated text.
- Windows: `open()` needs `encoding='utf-8'`. Inspect UTF-8 with Python, not `findstr`. Pass negative args as `--t=-6`, not `--t -6`. Excel strips leading zeros from IDs in label TSVs (use `zfill`).

## Keeping the docs current

- After finishing a task: add or edit DECISIONS entries in place (IDs are stable, superseded text is deleted), add durable numbers under "Numbers by task", update "Open questions", tick the task and update "Current status" in PLAN.md.
- If code diverges from SCHEMAS.md, update it the same day and add a row to its contract-breaking changes table.
- Publicly quoted metrics go in the DECISIONS results log with date and run ID. Dev numbers are never results.
- Fill the CV bullet only with numbers the project's own results produced.
