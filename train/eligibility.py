"""Eligibility rule for synthetic-query sampling (D038): train or val passages with
n_tokens >= MIN_TOKENS and English function-word fraction >= MIN_FW."""
from ingest.chunk import STOP

PASSAGES = "data/processed/passages.jsonl"
SPLITS = "data/processed/splits.json"
MIN_TOKENS = 100
MIN_FW = 0.15
STRIP = ".,;:!?()[]{}\"'"


def fw_fraction(text):
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower().strip(STRIP) in STOP)
    return hits / len(words)
