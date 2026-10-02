"""The whole pipeline for a list of "Other" returns.

  1. Triage           small model   one reason per row              (pipeline/triage.py)
  2. Escalate         strong model  second opinion on hard rows     (pipeline/escalate.py)
  3. Route + detail   small model   follow-up question per reason   (pipeline/detail.py)
  4. Review flag      code          which rows a person should read

Counting and ranking happen afterwards, in pipeline/aggregate.py.
"""
from pipeline.detail import add_details
from pipeline.escalate import escalate
from pipeline.triage import classify


def _start_record(t: dict) -> dict:
    return {
        "return_id": t["return_id"], "reason": t["reason"], "detail": "", "area": "",
        "confidence": t["confidence"], "evidence": t["evidence"],
        "multiple_reasons": t["multiple_reasons"], "status": t["status"],
        "problem": t["problem"],
        "read_by": "code" if t["status"] == "skipped_junk" else "small model",
        "escalated": False, "strong_failed": False, "review": "",
        "small_tokens_in": t["input_tokens"], "small_tokens_out": t["output_tokens"],
        "strong_tokens_in": 0, "strong_tokens_out": 0,
    }


def _flag_for_review(rec: dict):
    """A row goes to the review queue when the system should not be trusted on it."""
    if rec["status"] == "unclassified":
        rec["review"] = "Could not be labelled"
    elif rec["strong_failed"]:
        rec["review"] = "The stronger model failed, so this is the small model's answer"
    elif rec["status"] == "ok" and rec["confidence"] == "low":
        rec["review"] = "Still unsure after the stronger model"
    elif rec["status"] == "ok" and rec["reason"] == "unclear":
        rec["review"] = "No clear reason in the text"


def run_pipeline(rows: list[dict], small_triage, strong_triage, router,
                 progress=None, chunk: int = 40) -> list[dict]:
    """rows: dicts with return_id, category, size_ordered, other_text.
    progress: optional function(done, total), called after each chunk."""
    out = []
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        records = [_start_record(t) for t in classify(part, small_triage)]
        escalate(part, records, strong_triage)
        add_details(part, records, router)
        for rec in records:
            _flag_for_review(rec)
        out += records
        if progress:
            progress(min(start + chunk, len(rows)), len(rows))
    return out
