"""Find the settings the strong model accepts.

Run with:  python test_strong.py

It makes up to six small calls to STRONG_MODEL, trying different ways of asking for a
structured answer, and prints which ones work. Total cost is well under one US cent.
"""
import os
import tomllib

from langchain.chat_models import init_chat_model

from pipeline.config import STRONG_MODEL
from pipeline.triage import PROMPT, Triage

secrets = tomllib.load(open(".streamlit/secrets.toml", "rb"))
os.environ["OPENROUTER_API_KEY"] = secrets["OPENROUTER_API_KEY"]

ROW = {"category": "Kurti", "size": "M", "text": "kamar pe tight hai aur colour bhi alag hai"}
TRIES = [   # (method, temperature, max_tokens), the current settings first
    ("function_calling", 0, 500),
    ("function_calling", None, 500),
    ("json_schema", 0, 500),
    ("json_schema", None, 500),
    ("function_calling", None, 4000),
    ("json_schema", None, 4000),
]

print(f"Testing {STRONG_MODEL}\n")
winner = None
for method, temperature, max_tokens in TRIES:
    label = f"method={method}, temperature={temperature}, max_tokens={max_tokens}"
    kwargs = {"model_provider": "openrouter", "max_retries": 0, "timeout": 60_000,
              "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        model = init_chat_model(STRONG_MODEL, **kwargs)
        out = (PROMPT | model.with_structured_output(Triage, method=method,
                                                     include_raw=True)).invoke(ROW)
        if out["parsed"] is None:
            print(f"NO ANSWER  {label}\n           {str(out['parsing_error'])[:300]}\n")
            continue
        print(f"WORKS      {label}\n           -> {out['parsed'].reason}, "
              f"multiple_reasons={out['parsed'].multiple_reasons}\n")
        winner = winner or (method, temperature, max_tokens)
    except Exception as e:
        why = getattr(e, "body", "") or str(e)
        print(f"FAILED     {label}\n           {type(e).__name__}: {str(why)[:500]}\n")

print("-" * 70)
if winner:
    method, temperature, max_tokens = winner
    print("In pipeline/config.py, change the STRONG_MODEL line of MODEL_SETTINGS to:\n")
    print(f'    STRONG_MODEL: {{"method": "{method}", "temperature": {temperature}, '
          f'"max_tokens": {max_tokens}}},')
else:
    print("Nothing worked. In pipeline/config.py set:\n")
    print('    STRONG_MODEL = "google/gemini-3.8-flash"')
