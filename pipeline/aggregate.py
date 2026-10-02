"""Counting and ranking. Plain code: no model is involved in any number shown to the user."""
import pandas as pd

from pipeline.config import MIN_ROWS_TO_RANK

# Returns that already carry a dropdown reason are mapped in code, with no model call.
# These labels match the sample data. A dropdown label that is not listed here is left out
# of the action list and reported, so nothing is silently mislabelled.
DROPDOWN_MAP = {
    "Size too small": ("fit", "too_small"),
    "Size too large": ("fit", "too_large"),
    "Quality not as expected": ("quality", "unspecified"),
    "Different from image": ("not_as_shown", "unspecified"),
    "Wrong item received": ("wrong_or_damaged", ""),
    "Damaged or defective": ("wrong_or_damaged", ""),
    "Delivered late": ("delivery", ""),
    "Changed my mind": ("changed_mind", ""),
}
ACTIONABLE = ("fit", "quality", "not_as_shown")
KEYS = ["vendor_id", "category", "size_ordered"]

WORDS = {
    "too_small": "too small", "too_large": "too large", "stitching": "stitching",
    "thin_fabric": "thin or see-through fabric", "colour_bleed": "colour runs or fades",
    "shrinkage": "shrinks after washing", "tear_or_defect": "tears or breaks",
    "colour": "colour differs from the photo", "fabric": "fabric differs from the description",
    "design": "design differs from the photo",
}
FIX = {
    "too_small": 'Check this vendor\'s size chart. Add "runs small, order one size up" to the listing.',
    "too_large": 'Check this vendor\'s size chart. Add "runs large, order one size down" to the listing.',
    "shrinkage": "Raise with the vendor: the fabric shrinks. Add a wash-care note to the listing.",
    "stitching": "Raise the stitching quality with the vendor.",
    "thin_fabric": "Say the fabric is thin in the listing, or raise it with the vendor.",
    "colour_bleed": "Raise colour fastness with the vendor. Add a wash-care note.",
    "tear_or_defect": "Raise the defects with the vendor and check incoming stock.",
    "colour": "Re-shoot or colour-correct the product photo.",
    "fabric": "Correct the fabric named in the description.",
    "design": "Check the listing photo shows the design being shipped.",
}


def combine(all_returns: pd.DataFrame, results: pd.DataFrame, include_dropdown: bool = True):
    """One labelled row per return: classified "Other" rows, plus mapped dropdown rows."""
    typed = all_returns[KEYS + ["return_id"]].merge(
        results[["return_id", "reason", "detail", "area", "evidence", "status"]], on="return_id")
    typed = typed[typed["status"] == "ok"].drop(columns="status")
    typed["source"] = "Other text"
    if not include_dropdown:
        return typed, []
    picked = all_returns[all_returns["dropdown_reason"] != "Other"]
    known = picked[picked["dropdown_reason"].isin(list(DROPDOWN_MAP))].copy()
    unknown = sorted(set(picked["dropdown_reason"]) - set(DROPDOWN_MAP))
    known["reason"] = known["dropdown_reason"].map(lambda d: DROPDOWN_MAP[d][0])
    known["detail"] = known["dropdown_reason"].map(lambda d: DROPDOWN_MAP[d][1])
    known["area"], known["evidence"], known["source"] = "", "", "Dropdown"
    return pd.concat([typed, known[typed.columns]], ignore_index=True), unknown


SKIP = ("", "unspecified", "unclear", "overall", "not extracted")


def _top(series: pd.Series):
    """The most common specific value, how often it appears, and how many rows gave one."""
    known = series[~series.isin(SKIP)]
    counts = known.value_counts()
    return (counts.index[0], int(counts.iloc[0]), len(known)) if len(counts) else (None, 0, 0)


def action_list(labelled: pd.DataFrame, min_rows: int = MIN_ROWS_TO_RANK) -> pd.DataFrame:
    """Rank the groups that caused the most returns.

    Fit is grouped by vendor + product + size, because a size chart is fixed per size.
    Quality and "not as shown" are grouped by vendor + product, because size does not matter.
    """
    data = labelled[labelled["reason"].isin(ACTIONABLE)].copy()
    data["group_size"] = data["size_ordered"].where(data["reason"] == "fit", "all sizes")
    rows = []
    for (vendor, category, size, reason), g in data.groupby(
            ["vendor_id", "category", "group_size", "reason"]):
        if len(g) < min_rows:
            continue
        detail, n_detail, n_known = _top(g["detail"])
        problem = {"fit": "Fit", "quality": "Quality", "not_as_shown": "Not as shown"}[reason]
        fix = "Read the customer quotes to find the cause."
        quotes = g
        if detail:
            problem += f": {WORDS.get(detail, detail)} ({n_detail} of {n_known} that say)"
            fix = FIX.get(detail, fix)
            quotes = g[g["detail"] == detail]
        if reason == "fit":
            typed = g[g["source"] == "Other text"]
            area, n_area, _ = _top(typed["area"])
            if area and n_area >= 2:              # one mention is not a pattern
                problem += f". Most named area: {area} ({n_area} of {len(typed)} typed)"
        rows.append({
            "vendor": vendor, "product": category, "size": size or "not given",
            "returns": len(g), "problem": problem, "suggested fix": fix,
            "from Other text": int((g["source"] == "Other text").sum()),
            "customer quotes": " | ".join([q for q in quotes["evidence"] if q][:3]),
        })
    cols = ["vendor", "product", "size", "returns", "problem", "suggested fix",
            "from Other text", "customer quotes"]
    out = pd.DataFrame(rows, columns=cols).sort_values("returns", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)
