"""Models and prices in one place, so changing a model is a one-line edit."""

SMALL_MODEL = "google/gemini-3.5-flash-lite"     # reads every row: triage and detail
STRONG_MODEL = "anthropic/claude-sonnet-5.5"     # reads only the rows the small model is unsure of

# US dollars per million tokens (input, output), from openrouter.ai on 2 October 2026.
# Update these when you change a model or when prices move.
PRICES = {
    "google/gemini-3.5-flash-lite": (0.15, 1.25),
    "google/gemini-3.8-flash": (0.375, 1.875),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
}

# How each model is called.
#   method       how the fixed answer shape is requested: "function_calling" or "json_schema"
#   temperature  0 gives the same label for the same text every run. None leaves the
#                provider's default, for models that reject a temperature setting.
#   max_tokens   the longest reply allowed. Answers are a few fields, so this stays small.
MODEL_SETTINGS = {
    SMALL_MODEL: {"method": "function_calling", "temperature": 0, "max_tokens": 500},
    STRONG_MODEL: {"method": "json_schema", "temperature": 0, "max_tokens": 500},
}
DEFAULT_SETTINGS = {"method": "function_calling", "temperature": 0, "max_tokens": 500}

WEEKLY_OTHER_ROWS = 6500     # 48,000 orders x 31% returns x 44% "Other", from the brief
MIN_ROWS_TO_RANK = 5         # groups smaller than this are too thin to act on


def make_model(model_name: str):
    """Return (chat model, structured-output method) for a model, using MODEL_SETTINGS."""
    from langchain.chat_models import init_chat_model

    s = MODEL_SETTINGS.get(model_name, DEFAULT_SETTINGS)
    # timeout is in MILLISECONDS for the OpenRouter package: 30_000 = 30 seconds.
    kwargs = {"model_provider": "openrouter", "max_retries": 1, "timeout": 30_000,
              "max_tokens": s["max_tokens"]}
    if s["temperature"] is not None:
        kwargs["temperature"] = s["temperature"]
    return init_chat_model(model_name, **kwargs), s["method"]


def cost_usd(model: str, tokens_in: int, tokens_out: int):
    """Dollars for a model's token use, or None when the model has no price listed above."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
