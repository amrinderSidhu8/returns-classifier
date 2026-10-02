"""Score the whole pipeline against the answer key.

Run from the project folder:
    python eval/run_eval.py              # all "Other" rows
    python eval/run_eval.py --limit 50   # a quick, cheap check

Prints accuracy for the reason and for the detail, the top of the action list, and the cost.
Writes eval/results.csv with every row, so you can read the misses.
"""
import argparse
import os
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.aggregate import action_list, combine  # noqa: E402
from pipeline.config import SMALL_MODEL, STRONG_MODEL, WEEKLY_OTHER_ROWS, cost_usd  # noqa: E402
from pipeline.detail import build_detail_router  # noqa: E402
from pipeline.run import run_pipeline  # noqa: E402
from pipeline.triage import build_triage_chain  # noqa: E402


def load_returns():
    df = pd.read_csv(ROOT / "data" / "sample_returns.csv", dtype=str, keep_default_na=False)
    return df.drop_duplicates("return_id")


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


def pct(series) -> str:
    return f"{series.mean():.1%}  ({int(series.sum())} of {len(series)})" if len(series) else "no rows"


def report(m: pd.DataFrame):
    print(f"\nRows scored: {len(m)}")
    print(f"Reason correct:                    {pct(m['correct'])}")
    print(f"Reason, accepting second reason:   {m['correct_lenient'].mean():.1%}")

    # Detail is scored only where the reason was right, so each number means one thing.
    fit = m[m["correct"] & (m["true_reason"] == "fit")]
    print(f"Fit direction correct:             {pct(fit['detail'] == fit['true_detail'])}")
    print(f"Fit area correct:                  {pct(fit['area'] == fit['true_area'])}")
    qual = m[m["correct"] & (m["true_reason"] == "quality")]
    print(f"Quality issue correct:             {pct(qual['detail'] == qual['true_detail'])}")
    shown = m[m["correct"] & (m["true_reason"] == "not_as_shown")]
    print(f"Not-as-shown aspect correct:       {pct(shown['detail'] == shown['true_detail'])}")

    print(f"\nSkipped as junk in code:           {(m['status'] == 'skipped_junk').sum()}")
    print(f"Sent to the stronger model:        {int(m['escalated'].sum())}")
    failed = m[m["strong_failed"]]
    print(f"  of which the stronger model failed: {len(failed)}")
    if len(failed):
        print(f"  FIRST ERROR: {failed['problem'].iloc[0]}")
    print(f"In the review queue:               {(m['review'] != '').sum()}")
    print(f"Unclassified (both models failed): {(m['status'] == 'unclassified').sum()}")

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
    print("  none" if wrong.empty else
          wrong.groupby(["true_reason", "reason"]).size().sort_values(ascending=False).head(10).to_string())


def cost_report(m: pd.DataFrame):
    small = cost_usd(SMALL_MODEL, m["small_tokens_in"].sum(), m["small_tokens_out"].sum())
    strong = cost_usd(STRONG_MODEL, m["strong_tokens_in"].sum(), m["strong_tokens_out"].sum())
    print("\nCost:")
    print(f"  {SMALL_MODEL}: {int(m['small_tokens_in'].sum())} in, "
          f"{int(m['small_tokens_out'].sum())} out"
          + (f" = ${small:.4f}" if small is not None else " (no price set)"))
    print(f"  {STRONG_MODEL}: {int(m['strong_tokens_in'].sum())} in, "
          f"{int(m['strong_tokens_out'].sum())} out"
          + (f" = ${strong:.4f}" if strong is not None else " (no price set)"))
    if small is not None and strong is not None:
        total = small + strong
        print(f"  This run: ${total:.4f}. Per row: ${total / len(m):.5f}. "
              f"At {WEEKLY_OTHER_ROWS:,} rows a week: ${total / len(m) * WEEKLY_OTHER_ROWS:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    args = ap.parse_args()

    secrets = tomllib.load(open(ROOT / ".streamlit" / "secrets.toml", "rb"))
    os.environ["OPENROUTER_API_KEY"] = secrets["OPENROUTER_API_KEY"]

    returns = load_returns()
    rows = returns[returns["dropdown_reason"] == "Other"]
    if args.limit:
        rows = rows.head(args.limit)
    gold = pd.read_csv(ROOT / "data" / "gold_labels.csv", dtype=str, keep_default_na=False)

    records = run_pipeline(
        rows.to_dict("records"), build_triage_chain(SMALL_MODEL),
        build_triage_chain(STRONG_MODEL), build_detail_router(SMALL_MODEL),
        progress=lambda done, total: print(f"  read {done} of {total}", flush=True))
    results = pd.DataFrame(records)

    merged = score(results, gold).merge(
        rows[["return_id", "vendor_id", "category", "size_ordered", "other_text"]], on="return_id")
    report(merged)
    cost_report(merged)

    if not args.limit:      # the action list only means something on the whole file
        labelled, _ = combine(returns, results)
        print("\nTop of the action list. The five planted problems are V-07 Kurti, "
              "V-12 Kids T-shirt,\nV-23 Palazzo, V-31 Men's T-shirt and V-18 Saree:")
        print(action_list(labelled).head(8)[
            ["rank", "vendor", "product", "size", "returns", "problem"]].to_string(index=False))

    out = ROOT / "eval" / "results.csv"
    cols = ["return_id", "category", "size_ordered", "other_text", "true_reason", "reason",
            "correct", "true_detail", "detail", "true_area", "area", "secondary_reason",
            "confidence", "evidence", "multiple_reasons", "read_by", "escalated", "strong_failed", "review",
            "status", "problem", "scenario", "language"]
    merged[cols].to_csv(out, index=False, encoding="utf-8")
    print(f"\nEvery row is in {out.relative_to(ROOT)}. Filter correct = False to read the misses.")


if __name__ == "__main__":
    main()
