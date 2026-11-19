# analysis/

One-off studies behind individual decisions in `docs/DECISIONS.md`. The pipeline (`ingest/`, `retrieval/`, `train/`, `eval/`) never imports from here. Run from the repo root as modules, e.g. `python -m analysis.negatives.make_negative_review analyze`. Most read frozen local data and print tables; the hand-check samplers write review sheets to the gitignored `scratch_synth/`.

| Folder | What it covers | Decisions |
|---|---|---|
| `ingest/` | HTML structure surveys, parser diagnostics (coverage, math, TeX residue), chunk statistics, junk-filter inspection, reranker input lengths | D011, D012, D015, D017, D024 |
| `retrieval/` | Three-question first-stage spot-check from T07 | T07 notes |
| `eval/` | Topic title spot-check | D031 |
| `queries/` | Prompt iteration runner, review sheets and scoring, form checks, source ranks of pilot queries, generation failure listings, per-paper eligibility survey | D037 to D044 |
| `negatives/` | Rank-stratum hand check, reranker-as-filter calibration, threshold sweeps, hard-negative hand checks, pool survey, reranker timing, fresh label draws for judge validation | D045, D048 to D051, D062 |
| `training/` | Batch sampler composition, reranker group fill | D056, D063 |
