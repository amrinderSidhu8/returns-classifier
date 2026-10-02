import os

import pandas as pd
import streamlit as st

from pipeline.triage import build_triage_chain, classify

# Pick a model ID from openrouter.ai/models and paste it here.
SMALL_MODEL = "google/gemini-3.5-flash-lite"

st.set_page_config(page_title="Returns Reason Classifier", layout="wide")
st.title("Returns Reason Classifier")
st.caption("Dhaga & Co. · hello-world deploy")


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


uploaded = st.file_uploader("Upload a returns CSV", type="csv")
df = load_returns(uploaded if uploaded else "data/sample_returns.csv")
if not uploaded:
    st.info("Showing the bundled sample data.")

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
c3.metric('"Other" share', f"{len(other) / len(df):.0%}")

st.subheader("Dropdown reasons")
st.bar_chart(df["dropdown_reason"].value_counts())

st.subheader('What customers typed under "Other"')
st.dataframe(other[["return_id", "vendor_id", "category", "size_ordered", "other_text"]],
             width="stretch", hide_index=True)

st.divider()
st.header("Classify with LangChain")

api_key = get_secret("OPENROUTER_API_KEY")
if not api_key:
    st.warning("No API key set yet. Add OPENROUTER_API_KEY to the secrets to use the classifier.")
    st.stop()
if SMALL_MODEL == "PASTE_MODEL_ID_HERE":
    st.warning("Set SMALL_MODEL at the top of app.py to an OpenRouter model ID.")
    st.stop()
os.environ["OPENROUTER_API_KEY"] = api_key


@st.cache_resource
def triage_chain():
    return build_triage_chain(SMALL_MODEL)


st.subheader("Try it")
typed = st.text_input("Type any return reason", placeholder="kamar pe tight hai")
if typed:
    row = {"return_id": "typed", "category": "Kurti", "size_ordered": "M", "other_text": typed}
    st.json(classify([row], triage_chain())[0])

st.subheader('Classify "Other" rows')
n = st.slider("How many rows", min_value=5, max_value=100, value=20, step=5)
if st.button("Run triage"):
    batch = other.head(n)
    with st.spinner(f"Classifying {n} rows..."):
        results = pd.DataFrame(classify(batch.to_dict("records"), triage_chain()))
    shown = batch[["return_id", "other_text"]].merge(results, on="return_id")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Labelled by the model", int((results["status"] == "ok").sum()))
    m2.metric("Junk, skipped in code", int((results["status"] == "skipped_junk").sum()))
    m3.metric("Unclassified", int((results["status"] == "unclassified").sum()))
    m4.metric("Tokens in / out",
              f"{int(results['input_tokens'].sum())} / {int(results['output_tokens'].sum())}")

    st.bar_chart(results["reason"].value_counts())
    st.dataframe(shown[["return_id", "other_text", "reason", "confidence", "evidence",
                        "multiple_reasons", "status", "problem"]],
                 width="stretch", hide_index=True)
