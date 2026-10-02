"""Detail extraction, with routing.

Triage says WHAT kind of reason it is. This step asks a follow-up question that depends on
that answer: fit rows go to the fit extractor, quality rows to the quality extractor, and
"not as shown" rows to the mismatch extractor. Every other reason needs no follow-up and
makes no model call.

Why route: one prompt carrying every sub-label for every row costs more tokens per row and
lets the model mix fit sub-labels with quality ones.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

from pipeline.config import make_model


class FitDetail(BaseModel):
    """How and where a clothing item did not fit."""

    direction: Literal["too_small", "too_large", "unclear"] = Field(
        description="too_small: tight, short, chota. too_large: loose, long, bada, dhila.")
    area: Literal["bust", "waist", "hips", "length", "sleeves", "shoulders", "overall",
                  "unclear"] = Field(description="The part of the garment that did not fit.")


class QualityDetail(BaseModel):
    """What was wrong with the quality of a clothing item."""

    issue: Literal["stitching", "thin_fabric", "colour_bleed", "shrinkage", "tear_or_defect",
                   "unspecified"] = Field(description="The kind of quality problem.")


class MismatchDetail(BaseModel):
    """How a clothing item differed from its photo or description."""

    aspect: Literal["colour", "fabric", "design", "unspecified"] = Field(
        description="What differed from the listing.")


FIT_SYSTEM = """A customer returned a clothing item because it did not fit. They write in
Hinglish, Hindi, English or another Indian language, often with typos. Extract two things.

direction:
- too_small: tight, short, small, chota, fasta hai.
- too_large: loose, long, big, bada, dhila, lamba.
- unclear: the text does not say which.

area, the first one the customer names:
- bust: bust, chest, seene, "upar se" (the upper part).
- waist: waist, kamar, pet.
- hips: hips, kulhe.
- length: the garment is too short or too long (length, lambai, lamba, ankle, ghutne).
- sleeves: sleeves, baju, baazu, armhole.
- shoulders: shoulders, kandhe.
- overall: the whole garment is small or big, or another body part is named.
- unclear: only when direction is also unclear."""

QUALITY_SYSTEM = """A customer returned a clothing item because of its quality. They write in
Hinglish, Hindi, English or another Indian language, often with typos. Choose the issue.

- stitching: seams, silai, threads coming out, dhaga.
- thin_fabric: thin, see-through, transparent, patla, halka.
- colour_bleed: colour ran or faded in the wash, rang nikal gaya.
- shrinkage: it shrank after washing, sikud gaya, chota ho gaya after a wash.
- tear_or_defect: it tore, a hole, a broken zip, button or elastic, bobbles on the cloth.
- unspecified: the text says the quality is bad without saying how."""

MISMATCH_SYSTEM = """A customer returned a clothing item because it was not as shown. They
write in Hinglish, Hindi, English or another Indian language, often with typos. Choose what
differed from the photo or description.

- colour: the colour or shade.
- fabric: the material, for example cotton promised and polyester received.
- design: the print, pattern, embroidery or border.
- unspecified: the text says it looks different without saying how."""

# The triage result is passed in, so this step builds on the one before it (prompt chaining).
HUMAN = ("Product: {category}, size ordered: {size}\n"
         "Customer text: {text}\n"
         "The earlier step found the reason is {reason}, based on: {evidence}")

NO_DETAIL = {"raw": None, "parsed": None, "parsing_error": None, "skipped": True}
ROUTED = ("fit", "quality", "not_as_shown")


def _prompt(system):
    return ChatPromptTemplate.from_messages([("system", system), ("human", HUMAN)])


def make_router(fit_chain, quality_chain, mismatch_chain):
    """Send each row to the extractor for its reason. Other reasons get no model call."""
    return RunnableBranch(
        (lambda x: x["reason"] == "fit", fit_chain),
        (lambda x: x["reason"] == "quality", quality_chain),
        (lambda x: x["reason"] == "not_as_shown", mismatch_chain),
        RunnableLambda(lambda x: NO_DETAIL),
    )


def build_detail_router(model_name: str):
    model, method = make_model(model_name)

    def chain(system, schema):
        return _prompt(system) | model.with_structured_output(schema, method=method,
                                                              include_raw=True)

    return make_router(chain(FIT_SYSTEM, FitDetail), chain(QUALITY_SYSTEM, QualityDetail),
                       chain(MISMATCH_SYSTEM, MismatchDetail))


def _read(result):
    """Return (detail, area, tokens_in, tokens_out, ok) from one router result."""
    if isinstance(result, Exception):
        return "", "", 0, 0, False
    if result.get("skipped"):
        return "", "", 0, 0, True
    usage = getattr(result.get("raw"), "usage_metadata", None) or {}
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    parsed = result.get("parsed")
    if parsed is None:
        return "", "", tin, tout, False
    if isinstance(parsed, FitDetail):
        return parsed.direction, parsed.area, tin, tout, True
    if isinstance(parsed, QualityDetail):
        return parsed.issue, "", tin, tout, True
    return parsed.aspect, "", tin, tout, True


def add_details(rows: list[dict], records: list[dict], router, max_concurrency: int = 8):
    """Fill detail and area on each record, in place. One retry, then detail is left as
    'not extracted' and the row keeps its top-level reason."""
    todo = [i for i, r in enumerate(records) if r["reason"] in ROUTED]

    def ask(indexes):
        inputs = [{"reason": records[i]["reason"], "evidence": records[i]["evidence"],
                   "category": rows[i]["category"],
                   "size": rows[i]["size_ordered"] or "not given",
                   "text": rows[i]["other_text"]} for i in indexes]
        return router.batch(inputs, config={"max_concurrency": max_concurrency},
                            return_exceptions=True)

    for attempt in (1, 2):
        if not todo:
            break
        failed = []
        for i, result in zip(todo, ask(todo)):
            detail, area, tin, tout, ok = _read(result)
            records[i]["small_tokens_in"] += tin
            records[i]["small_tokens_out"] += tout
            if ok:
                records[i]["detail"], records[i]["area"] = detail, area
            else:
                failed.append(i)
        todo = failed
    for i in todo:
        records[i]["detail"] = "not extracted"
