import os

import pandas as pd
import streamlit as st

from pipeline.aggregate import action_list, combine
from pipeline.config import (MIN_ROWS_TO_RANK, SMALL_MODEL, STRONG_MODEL, WEEKLY_OTHER_ROWS,
                             cost_usd)
from pipeline.detail import build_detail_router
from pipeline.run import run_pipeline
from pipeline.triage import build_triage_chain

st.set_page_config(page_title="Returns Reason Classifier", layout="wide")
st.title("Returns Reason Classifier")
st.caption('Dhaga & Co. · Reads the "Other" box on returns and shows what to fix first')
def get_secret(name):
    """Read a key from Streamlit secrets, or from the environment when running without them."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


@st.cache_data
def load_returns(path_or_file):
    # dtype=str and keep_default_na=False stop pandas turning a typed "NA" into an empty cell
    return pd.read_csv(path_or_file, dtype=str, keep_default_na=False)


@st.cache_resource
def get_chains():
    """Built once and reused: small-model triage, strong-model triage, and the detail router."""
    return (build_triage_chain(SMALL_MODEL), build_triage_chain(STRONG_MODEL),
            build_detail_router(SMALL_MODEL))


def to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")    # utf-8-sig so Excel shows Hindi text


# ------------------------------------------------------------------ 1. load the file
uploaded = st.file_uploader("Upload a returns CSV", type="csv")
df = load_returns(uploaded if uploaded else "data/sample_returns.csv")
if not uploaded:
    st.info("Showing the bundled sample data. It is made up for testing.")

required = {"return_id", "vendor_id", "category", "size_ordered", "dropdown_reason", "other_text"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing columns: {', '.join(sorted(missing))}")
    st.stop()

df = df.drop_duplicates("return_id")
other = df[df["dropdown_reason"] == "Other"]

c1, c2, c3 = st.columns(3)
c1.metric("Returns", len(df))
c2.metric('Filed under "Other"', len(other))
c3.metric('"Other" share', f"{len(other) / len(df):.0%}" if len(df) else "0%")

with st.expander("See the dropdown reasons and what customers typed"):
    st.bar_chart(df["dropdown_reason"].value_counts())
    st.dataframe(other[["return_id", "vendor_id", "category", "size_ordered", "other_text"]],
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ 2. run the pipeline
st.divider()
st.header("Read the returns")

api_key = get_secret("OPENROUTER_API_KEY")
if not api_key:
    st.warning("No API key set yet. Add OPENROUTER_API_KEY to the secrets to read the returns.")
    st.stop()
os.environ["OPENROUTER_API_KEY"] = api_key

if other.empty:
    st.warning('This file has no rows with "Other" in dropdown_reason, so there is nothing to read.')
    st.stop()

scope = st.radio("How much to read", ["A quick sample", 'Every "Other" row in the file'],
                 horizontal=True)
if scope == "A quick sample":
    limit = min(len(other), 100)
    n = st.slider("Rows in the sample", min_value=1, max_value=limit, value=min(20, limit))
else:
    n = len(other)
    st.caption(f"{n} rows. Allow a few minutes for a large file.")

file_key = f"{len(df)}-{df['return_id'].iloc[0]}"
if st.session_state.get("file_key") != file_key:          # a new file clears old results
    st.session_state.pop("results", None)
    st.session_state["file_key"] = file_key

if st.button("Read the returns", type="primary"):
    batch = other.head(n)
    small_triage, strong_triage, router = get_chains()
    bar = st.progress(0.0, text="Starting...")
    records = run_pipeline(
        batch.to_dict("records"), small_triage, strong_triage, router,
        progress=lambda done, total: bar.progress(done / total, text=f"Read {done} of {total} rows"))
    bar.empty()
    # Kept in the session, so the results stay on screen after a download click.
    st.session_state["results"] = batch[
        ["return_id", "vendor_id", "category", "size_ordered", "other_text"]
    ].merge(pd.DataFrame(records), on="return_id")

# ------------------------------------------------------------------ 3. show the results
if "results" in st.session_state:
    res = st.session_state["results"]
    queue = res[res["review"] != ""]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows read", len(res))
    m2.metric("Given a reason", int(((res["status"] == "ok") & (res["reason"] != "unclear")).sum()))
    m3.metric("Sent to the stronger model", int(res["escalated"].sum()))
    m4.metric("For a person to check", len(queue))

    strong_failed = res[res["strong_failed"]]
    if len(strong_failed):
        st.error(f"The stronger model failed on {len(strong_failed)} of "
                 f"{int(res['escalated'].sum())} rows sent to it, so those rows show the small "
                 f"model's answer. First error: {strong_failed['problem'].iloc[0]}")

    tab_action, tab_reasons, tab_review, tab_rows, tab_report = st.tabs(
        ["Action list", "Reasons", "Review queue", "All rows", "Run report"])

    with tab_action:
        with_dropdown = st.checkbox("Also count returns that already have a dropdown reason",
                                    value=True)
        min_rows = st.number_input("Smallest group to rank", min_value=2, max_value=50,
                                   value=MIN_ROWS_TO_RANK)
        labelled, unknown = combine(df, res, include_dropdown=with_dropdown)
        ranked = action_list(labelled, min_rows=int(min_rows))
        st.caption(f"Based on {int((labelled['source'] == 'Other text').sum())} of {len(other)} "
                   f'"Other" returns'
                   + (f" and {int((labelled['source'] == 'Dropdown').sum())} dropdown returns."
                      if with_dropdown else "."))
        if unknown:
            st.warning("These dropdown reasons are not recognised and were left out: "
                       + ", ".join(unknown))
        if ranked.empty:
            st.info("Not enough rows in any group to rank. Read more rows, or lower the "
                    "smallest group.")
        else:
            st.dataframe(ranked, width="stretch", hide_index=True)
            st.download_button("Download the action list as CSV", to_csv(ranked),
                               "action_list.csv", "text/csv")
        st.caption("Ranked by the number of returns. The file has no order counts, so this is "
                   "not a return rate.")

    with tab_reasons:
        st.bar_chart(res["reason"].value_counts())
        detail = res[res["detail"] != ""].groupby(["reason", "detail"]).size()
        st.dataframe(detail.rename("returns").reset_index().sort_values("returns", ascending=False),
                     hide_index=True)

    with tab_review:
        if queue.empty:
            st.success("Nothing needs checking in this run.")
        else:
            st.write("The system does not trust its own answer on these rows. The reason shown "
                     "is its best guess.")
            st.dataframe(queue[["return_id", "vendor_id", "category", "size_ordered", "other_text",
                                "review", "reason", "confidence", "problem"]],
                         width="stretch", hide_index=True)

    with tab_rows:
        cols = ["return_id", "vendor_id", "category", "size_ordered", "other_text", "reason",
                "detail", "area", "confidence", "evidence", "multiple_reasons", "read_by",
                "status", "review", "problem"]
        st.dataframe(res[cols], width="stretch", hide_index=True)
        st.download_button("Download classified rows as CSV", to_csv(res[cols]),
                           "classified_returns.csv", "text/csv")

    with tab_report:
        usage = pd.DataFrame([
            {"model": SMALL_MODEL, "job": "Triage and detail, every row",
             "tokens in": int(res["small_tokens_in"].sum()),
             "tokens out": int(res["small_tokens_out"].sum())},
            {"model": STRONG_MODEL, "job": "Second opinion, hard rows only",
             "tokens in": int(res["strong_tokens_in"].sum()),
             "tokens out": int(res["strong_tokens_out"].sum())},
        ])
        usage["cost (USD)"] = [cost_usd(r["model"], r["tokens in"], r["tokens out"])
                               for r in usage.to_dict("records")]
        st.dataframe(usage, hide_index=True)
        if usage["cost (USD)"].isna().any():
            st.warning("A model has no price in pipeline/config.py, so the total is incomplete.")
        total = float(usage["cost (USD)"].sum())
        r1, r2, r3 = st.columns(3)
        r1.metric("Cost of this run", f"${total:.4f}")
        r2.metric("Cost per row", f"${total / len(res):.5f}")
        r3.metric(f"At {WEEKLY_OTHER_ROWS:,} rows a week", f"${total / len(res) * WEEKLY_OTHER_ROWS:.2f}")
        st.dataframe(res["status"].value_counts().rename("rows").reset_index(), hide_index=True)
        st.caption("Skipped junk rows are blank or symbol-only and cost nothing. Unclassified "
                   "rows failed with both models and are in the review queue.")

# ------------------------------------------------------------------ 4. test one sentence
st.divider()
st.header("Test a single return reason")
st.caption("Type any reason to see how it is read. Useful for checking a case by hand.")
with st.form("try_it"):
    typed = st.text_input("Return reason", placeholder="kamar pe tight hai")
    go = st.form_submit_button("Check")
if go and typed.strip():
    row = {"return_id": "typed", "category": "Kurti", "size_ordered": "M", "other_text": typed}
    small_triage, strong_triage, router = get_chains()
    st.session_state["tried"] = run_pipeline([row], small_triage, strong_triage, router)[0]
if "tried" in st.session_state:
    t = st.session_state["tried"]
    parts = [t["reason"]] + [p for p in (t["detail"], t["area"]) if p]
    st.subheader(" · ".join(parts).replace("_", " "))
    st.write(f"Confidence: **{t['confidence']}** · Read by: **{t['read_by']}**"
             + (f" · Evidence: \"{t['evidence']}\"" if t["evidence"] else ""))
    if t["review"]:
        st.warning(f"This would go to the review queue: {t['review']}. {t['problem']}")
    with st.expander("Full result"):
        st.json(t)
