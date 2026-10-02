# Build note: Returns Reason Classifier

Written  by Group 4. Live app: https://dhaga-returns-classifier.streamlit.app/

## 1. Code or model, step by step

The rule we followed: if a step has one right answer that a rule can find, code does it. A model is used only where someone has to read what a customer meant.

| Step | Done by | Why |
|---|---|---|
| Load the file, check the columns, drop repeated return IDs | Code | Fixed rules. Free, and the same every run |
| Skip blank, emoji or symbol-only text | Code | "Has no letters" is a rule. No model call is spent |
| Triage: one reason, a confidence, the exact words that show it | Small model | Hinglish, typos and mixed languages. A keyword list breaks: "chota ho gaya after wash" is quality, not fit |
| Check the answer: right shape, and the quoted words are really in the text | Code | An exact text match. Catches a made-up quote without paying a model to judge a model |
| Pick the hard rows: not high confidence, two reasons, or failed twice | Code | A rule on fields the model already returned |
| Second opinion on the hard rows | Strong model | The same reading task, on the rows where the small model said it was unsure |
| Send each row to the fit, quality or not-as-shown reader | Code | The route follows from the reason label. Nothing to judge |
| Detail: too small or too large and where, which quality issue, what differed | Small model | Reading again, on a narrower question |
| Review queue, counting by vendor, product and size, ranking, cost report | Code | Arithmetic and lookups. A model would be slower, cost more and could miscount |

## 2. The patterns and why

**Routing.** The triage reason decides which of three detail prompts runs. The other four reasons make no second call. We chose this over one large prompt holding every sub-label because each prompt stays short, the model cannot mix a fit label with a quality label, and rows that need no follow-up cost nothing. Escalation is a second route, by difficulty: only hard rows reach the strong model.

**Prompt chaining.** Triage, then a code check, then detail. The detail prompt is given the reason and the quoted words from triage. Two easy questions are answered better than one hard one, and the code check between them stops a bad triage answer from flowing into the next step.

**Not used.** Evaluator-optimizer: our check on the output is an exact rule, so code does it for free. Parallelization: we send eight rows at a time for speed, but that is not splitting one task or voting, so we do not claim it.

## 3. Models and settings

| Model | Job | Price per million tokens (in, out) | Temperature |
|---|---|---|---|
| `google/gemini-3.5-flash-lite` | Triage and detail, every row | $0.15, $1.25 | 0 |
| `anthropic/claude-sonnet-5.5` | Second opinion, about 7% of rows | $2.00, $10.00 | 0 |

- **Cost.** Sonnet is 13 times the price on input and 8 times on output, so it reads only the rows that need it.
- **Quality.** On our 440 test rows the pipeline got the reason right on 438. Detail: fit direction 99%, fit area 94%, quality issue 98%, not-as-shown aspect 96%.
- **Latency.** This is a weekly batch, so nobody waits on a single row. A full 440-row run took about 25 seconds on our machine.
- **Temperature 0** for both. This is labelling, so the same text should get the same label on every run.
- **Structured output.** Each step returns a fixed shape with a fixed list of allowed values (Pydantic). When an answer fails the check: one retry, then the strong model, then the row is marked `unclassified` and goes to the review queue with the error. No row is dropped.

**What the numbers do not show.** Sonnet read about 7% of rows, made up 53% of the cost, and did not change the measured accuracy. Our test text is generated, so it is cleaner than real customer text. That makes 438 of 440 a ceiling, and it means we cannot yet say whether Sonnet earns its place.

## 4. The cost line

One full run on 440 "Other" rows, with token counts from the run itself:

| Model | Tokens in | Tokens out | Arithmetic | Cost |
|---|---|---|---|---|
| Gemini | 329,650 | 23,750 | (329,650 × $0.15 + 23,750 × $1.25) ÷ 1,000,000 | $0.079 |
| Sonnet | 34,830 | 2,124 | (34,830 × $2 + 2,124 × $10) ÷ 1,000,000 | $0.091 |
| **Total** | | | | **$0.170** |

- Per row: $0.170 ÷ 440 = **$0.00039**
- Dhaga's weekly volume: 48,000 orders × 31% returned × 44% "Other" = about 6,500 rows
- Per week: 6,500 × $0.00039 = **about $2.50, or about ₹240** (at ₹96 to the dollar)
- Per year: about $130, or about ₹12,500

Hosting is on the free tier of Streamlit Community Cloud. Not counted: the time a person spends on the review queue. Real text may be longer than our sample, but most of the tokens are the fixed prompt (about 750 tokens in per row), so the cost should move little.

## 5. The thing that broke

**What happened.** After we added the strong model, the evaluation ran without errors and accuracy looked fine. But the cost report showed Sonnet with 0 tokens. Every call to it was failing.

**Why.** LangChain's default way of asking for a fixed answer shape is a forced tool call. Claude Sonnet through OpenRouter rejects that. Our code caught the error and kept the small model's answer, which was the right fallback. But it labelled those rows "still unsure after the stronger model", which was false, and reported nothing. The system was failing silently.

**How we found it.** Only because the cost report listed tokens per model. Accuracy alone would never have shown it. We then wrote `test_strong.py`, which tries six settings against the model and prints the one that works.

**The fix, in two parts.**

1. The cause: Sonnet now uses the JSON schema method. Settings are per model in `pipeline/config.py`.
2. The silence: every row now carries a `strong_failed` flag. The app shows a red banner with the provider's own error, the evaluation prints a count, and those rows go to the review queue with the true reason.

**What we learned.** A caught error that is not reported is worse than a crash. We now treat "a model shows zero tokens" as a failure signal. One more of the same kind: the OpenRouter package reads `timeout` in milliseconds, so our `timeout=60` made every call time out and the app hung for 20 minutes. It is now 30,000.

## 6. Limits and what comes next

- **Real data first.** Run 200 of Neha's real "Other" rows, hand-label them, and score again.
- **Fewer escalations.** Send only low-confidence and failed rows to Sonnet, then compare accuracy and cost.
- **Return rate, not count.** The action list ranks by number of returns. With order counts per vendor, product and size it can rank by rate.
