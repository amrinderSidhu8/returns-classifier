# Discovery note: Dhaga & Co.

Written 3 October 2026 by [group member names]. The ranking below was first set out in our PRD on 2 October, before the first commit. This note was written up after it.

## 1. The problem, in the client's language

Almost a third of our orders come back, and for nearly half of those returns we do not know why, because the reason is sitting unread in the "Other" box.

## 2. Who owns it, and what they do today

Neha, Category Head. She reads the "Other" box by hand, a few hundred entries at a time, out of roughly 6,500 a week.

## 3. Evidence from the brief

- Neha: "Returns are thirty-one percent overall. When I read the Other box by hand, most of it is about fit, but I can only read a few hundred at a time."
- Returns data: the reason is a dropdown with an "Other" free-text box, and 44% of returns land in "Other".
- Catalogue data: "Size charts differ per vendor." There are about 40 vendors.
- Karthik, Data analyst: "Metabase tells us what happened. It never tells us why."

## 4. What it costs them

| Figure | Value | Source |
|---|---|---|
| Returns per week | about 14,900 | Derived: 48,000 orders × 31%, assuming the rate is per order |
| "Other" returns per week | about 6,500 | Derived: 14,900 × 44% |
| Order value returned per week | about ₹1.25 crore | Our estimate: 14,900 × ₹840 average order value |
| Cost of one return | not stated | We would ask the client |

In time: Neha reads a few hundred of 6,500 entries, so most of the signal is never seen.

## 5. What success looks like

- The share of "Other" returns that carry a specific reason, measured on the returns table.
- Agreement between the tool's reason and a hand label, on a sample Neha's team checks.
- Later: the fit-return rate for vendor and size combinations where a size chart was fixed, against those where it was not. Returns and orders tables already hold this.

The tool does not lower the 31% by itself. It tells the Category team where to act.

## 6. Ranked shortlist

| Rank | Problem and owner | Why it is a real candidate | Why it sits here |
|---|---|---|---|
| 1 | Unread "Other" return reasons. Neha | 6,500 rows a week, already collected, read by hand today | A named owner gets an immediate substitute for manual work. It is internal, so a wrong label is cheap. We can show it works without touching live orders |
| 2 | Cash-on-delivery return to origin. Faizan | The only stated unit cost: ₹120 each, about 7,600 a week, about ₹9.1 lakh a week (derived: 48,000 × 61% × 26% × ₹120) | The clearest rupee figure. Fixing it means predicting refusals and changing checkout or calling customers, which we cannot prove works without a live trial |
| 3 | "Where is my order" tickets. Arpita | 58% of 9,000 weekly tickets, nine-hour first response | Replies go to customers, so every answer needs review, and it needs live tracking data from three couriers |
| 4 | Sample-to-live time. Vivek | Six to nine days, and a slipped drop loses the Tuesday traffic spike | The brief does not say how the days split between studio shoot and typing, so the gain is unknown |
| 5 | Repeat purchase and acquisition cost. Ritu, Sameer | The founders' top concerns: 22% repeat rate, cost up 40% | These are outcomes of many causes, not one workflow a named person runs today |

## 7. Biggest assumption

That the "Other" text is specific enough to act on at vendor, product and size level.

It is wrong if a sample of real "Other" entries turns out to be mostly blank or one word, or if fit complaints spread evenly across vendors and sizes. Our build uses synthetic data, so this is untested until Neha shares real rows.
