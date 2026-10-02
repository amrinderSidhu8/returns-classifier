"""Score the triage step against the answer key.

Run from the project folder:
    python eval/run_eval.py              # all "Other" rows
    python eval/run_eval.py --limit 50   # a quick, cheap check

Prints accuracy and writes eval/results.csv with every row, so you can read the misses.
"""
import argparse
import os
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.triage import build_triage_chain, classify  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"


def load_other_rows():
    df = pd.read_csv(ROOT / "data" / "sample_returns.csv", dtype=str, keep_default_na=False)
    df = df.drop_duplicates("return_id")
    return df[df["dropdown_reason"] == "Other"]


def score(results: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to the answer key and mark each row right or wrong."""
    gold = gold.rename(columns={"multiple_reasons": "true_multiple_reasons"})
    gold["scenario"] = gold["scenario"].str.replace("+duplicate_row", "", regex=False)
    m = results.merge(gold, on="return_id", how="left")
    m["correct"] = m["reason"] == m["true_reason"]
    # A two-reason row is also accepted if the model picked the second reason.
    m["correct_lenient"] = m["correct"] | (
        (m["secondary_reason"] != "") & (m["reason"] == m["secondary_reason"]))
    return m


def report(m: pd.DataFrame):
    n = len(m)
    print(f"\nRows scored: {n}")
    print(f"Accuracy, top-level reason:        {m['correct'].mean():.1%}  "
          f"({int(m['correct'].sum())} of {n})")
    print(f"Accuracy, accepting second reason: {m['correct_lenient'].mean():.1%}")
    print(f"Unclassified (failed twice):       {(m['status'] == 'unclassified').sum()}")
    print(f"Skipped as junk in code:           {(m['status'] == 'skipped_junk').sum()}")
    print(f"Tokens in / out:                   {int(m['input_tokens'].sum())} / "
          f"{int(m['output_tokens'].sum())}")

    by = m.groupby("true_reason")["correct"].agg(rows="count", correct="sum")
    by["accuracy"] = (by["correct"] / by["rows"]).map("{:.0%}".format)
    print("\nBy true reason:")
    print(by.sort_values("rows", ascending=False).to_string())

    by = m.groupby("scenario")["correct"].agg(rows="count", correct="sum")
    by["accuracy"] = (by["correct"] / by["rows"]).map("{:.0%}".format)
    print("\nBy scenario:")
    print(by.sort_values("rows", ascending=False).to_string())

    wrong = m[~m["correct"]]
    print("\nMost common mistakes (true reason -> model's answer):")
    if wrong.empty:
        print("  none")
    else:
        pairs = wrong.groupby(["true_reason", "reason"]).size().sort_values(ascending=False)
        print(pairs.head(10).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    args = ap.parse_args()

    secrets = tomllib.load(open(ROOT / ".streamlit" / "secrets.toml", "rb"))
    os.environ["OPENROUTER_API_KEY"] = secrets["OPENROUTER_API_KEY"]

    rows = load_other_rows()
    if args.limit:
        rows = rows.head(args.limit)
    gold = pd.read_csv(ROOT / "data" / "gold_labels.csv", dtype=str, keep_default_na=False)

    chain = build_triage_chain(args.model)
    records = rows.to_dict("records")
    results = []
    for start in range(0, len(records), 50):          # in chunks, so progress is visible
        results += classify(records[start:start + 50], chain)
        print(f"  classified {min(start + 50, len(records))} of {len(records)}", flush=True)

    merged = score(pd.DataFrame(results), gold)
    merged = merged.merge(rows[["return_id", "category", "size_ordered", "other_text"]],
                          on="return_id")
    report(merged)

    out = ROOT / "eval" / "results.csv"
    cols = ["return_id", "category", "size_ordered", "other_text", "true_reason", "reason",
            "correct", "secondary_reason", "confidence", "evidence", "multiple_reasons",
            "status", "problem", "scenario", "language"]
    merged[cols].to_csv(out, index=False, encoding="utf-8")
    print(f"\nEvery row is in {out.relative_to(ROOT)}. Filter correct = False to read the misses.")


if __name__ == "__main__":
    main()
