"""Triage: give every "Other" return one top-level reason.

Model work : reading the customer's text and choosing a reason (language judgment).
Code work  : skipping junk, checking the output, retrying once, marking failures.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.config import make_model

Reason = Literal["fit", "quality", "not_as_shown", "wrong_or_damaged", "delivery",
                 "changed_mind", "unclear"]


class Triage(BaseModel):
    """The reason a customer returned a clothing item."""

    reason: Reason = Field(description="The main reason for the return.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How sure you are of the reason.")
    evidence: str = Field(
        description="The exact words copied from the customer's text that show the reason. "
                    "Empty string when the reason is unclear.")
    multiple_reasons: bool = Field(
        description="True when the text gives more than one reason.")


SYSTEM = """You label return reasons for Dhaga & Co., an Indian online clothing brand.
Customers write in Hinglish, Hindi, English or another Indian language, often with typos.

Choose exactly one reason:
- fit: the size or fitting is wrong (tight, loose, short, long, chota, bada, dhila).
- quality: the product is poorly made or did not last (stitching, thin cloth, colour runs,
  shrank after washing, tore or broke with use).
- not_as_shown: colour, fabric or design differs from the photo or description.
- wrong_or_damaged: a different item, size or colour was sent, or it arrived torn, stained,
  used or with a piece missing.
- delivery: it arrived too late.
- changed_mind: the customer's own choice, not a fault in the product. They no longer want
  it, ordered by mistake, found it cheaper, someone at home said no, or it was not to their
  taste ("pasand nahi aaya").
- unclear: the text does not say why. Never guess.

Rules:
- "Shrank after washing" is quality, not fit.
- If there are two reasons, pick the one stated first and set multiple_reasons to true.
- evidence must be copied word for word from the customer's text.
- If the text only says the product was not good, not okay or not right, without saying
  what was wrong ("theek nahi tha", "achha nahi hai", "sahi nahi hai", "not good", "bakwas"),
  choose unclear.
- delivery means late arrival only. A complaint about the delivery person or the packaging
  is unclear.
- Use low confidence when you are unsure. Use unclear when there is no reason at all."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Product: {category}, size ordered: {size}\nCustomer text: {text}"),
])


def build_triage_chain(model_name: str):
    """Prompt -> small model -> validated Triage object.

    include_raw=True makes the chain return a dict with "raw", "parsed" and "parsing_error",
    so a bad answer is reported instead of raising.
    """
    model, method = make_model(model_name)
    return PROMPT | model.with_structured_output(Triage, method=method, include_raw=True)


def is_junk(text: str) -> bool:
    """Blank, symbols or emoji only. Decided in code, so no model call is spent on it."""
    return not any(ch.isalpha() for ch in text)


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _check(result, text: str):
    """Return (Triage, None) when the output can be trusted, else (None, why not)."""
    if isinstance(result, Exception):
        # .body holds the provider's own explanation when there is one
        why = getattr(result, "body", "") or str(result)
        return None, f"model call failed: {type(result).__name__}: {str(why)[:400]}"
    parsed = result.get("parsed")
    if parsed is None:
        return None, "output did not match the schema"
    if parsed.reason != "unclear" and _squash(parsed.evidence) not in _squash(text):
        return None, "evidence phrase is not in the customer's text"
    if parsed.reason != "unclear" and not parsed.evidence.strip():
        return None, "no evidence phrase given"
    return parsed, None


def _tokens(result):
    raw = None if isinstance(result, Exception) else result.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def classify(rows: list[dict], chain, max_concurrency: int = 8) -> list[dict]:
    """rows: dicts with return_id, category, size_ordered, other_text.

    Returns one dict per row with reason, confidence, evidence, multiple_reasons and status.
    status is one of: ok, skipped_junk, unclassified.
    """
    out = [None] * len(rows)
    todo = []
    for i, r in enumerate(rows):
        if is_junk(r["other_text"]):
            out[i] = {"return_id": r["return_id"], "reason": "unclear", "confidence": "high",
                      "evidence": "", "multiple_reasons": False, "status": "skipped_junk",
                      "problem": "", "input_tokens": 0, "output_tokens": 0}
        else:
            todo.append(i)

    def ask(indexes):
        inputs = [{"category": rows[i]["category"], "size": rows[i]["size_ordered"] or "not given",
                   "text": rows[i]["other_text"]} for i in indexes]
        return chain.batch(inputs, config={"max_concurrency": max_concurrency},
                           return_exceptions=True)

    spent = {i: [0, 0] for i in todo}
    problems = {}
    for attempt in (1, 2):                      # one retry, then give up visibly
        if not todo:
            break
        failed = []
        for i, result in zip(todo, ask(todo)):
            tin, tout = _tokens(result)
            spent[i][0] += tin
            spent[i][1] += tout
            parsed, problem = _check(result, rows[i]["other_text"])
            if parsed is None:
                problems[i] = problem
                failed.append(i)
                continue
            out[i] = {"return_id": rows[i]["return_id"], **parsed.model_dump(), "status": "ok",
                      "problem": "", "input_tokens": spent[i][0], "output_tokens": spent[i][1]}
        todo = failed

    for i in todo:
        out[i] = {"return_id": rows[i]["return_id"], "reason": "unclassified", "confidence": "low",
                  "evidence": "", "multiple_reasons": False, "status": "unclassified",
                  "problem": problems[i], "input_tokens": spent[i][0],
                  "output_tokens": spent[i][1]}
    return out
