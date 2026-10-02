"""One plain model call, to check the key and the model before running the app.
Run with:  python test_call.py
"""
import time
import tomllib

from langchain.chat_models import init_chat_model

secrets = tomllib.load(open(".streamlit/secrets.toml", "rb"))
MODEL = "google/gemini-3.5-flash-lite"      # use the same ID as SMALL_MODEL in app.py

model = init_chat_model(MODEL, model_provider="openrouter", temperature=0, max_retries=0,
                        timeout=30_000, api_key=secrets["OPENROUTER_API_KEY"])
start = time.time()
try:
    reply = model.invoke("Reply with the single word: ok")
    print("WORKS:", reply.text, f"({time.time() - start:.1f}s)")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:500])
