"""Escalation: a second opinion from the stronger model, on the hard rows only.

A row is escalated when the small model
  - was not highly confident,
  - said the customer gave more than one reason, or
  - failed to give a usable answer twice.

The strong model gets the same prompt and the same answer shape. It costs more per row, so
it is paid for only on the rows that need it.
"""
from pipeline.triage import classify


def needs_escalation(record: dict) -> bool:
    if record["status"] == "unclassified":
        return True
    return record["status"] == "ok" and (
        record["confidence"] != "high" or record["multiple_reasons"])


def escalate(rows: list[dict], records: list[dict], strong_chain):
    """Re-read the flagged rows with the strong model and update the records in place."""
    flagged = [i for i, r in enumerate(records) if needs_escalation(r)]
    if not flagged:
        return
    second = classify([rows[i] for i in flagged], strong_chain)
    for i, new in zip(flagged, second):
        rec = records[i]
        rec["escalated"] = True
        rec["strong_tokens_in"] += new["input_tokens"]
        rec["strong_tokens_out"] += new["output_tokens"]
        if new["status"] == "ok":
            for key in ("reason", "confidence", "evidence", "multiple_reasons"):
                rec[key] = new[key]
            rec["status"], rec["problem"], rec["read_by"] = "ok", "", "strong model"
        elif rec["status"] == "ok":
            # The strong model failed, so the small model's answer stands, but a person checks it.
            rec["strong_failed"] = True
            rec["problem"] = f"strong model: {new['problem']}"
        else:
            rec["strong_failed"] = True
            rec["problem"] = f"small model: {rec['problem']}; strong model: {new['problem']}"
