# Returns Reason Classifier

Built for Dhaga & Co. (FDE Academy, Mini Project 1).

**Live app:** https://dhaga-returns-classifier.streamlit.app/
(The app sleeps when unused. If you see a "wake up" button, click it and wait about a minute.)

## What it does

44% of Dhaga's returns are filed under "Other", with the real reason typed in a free-text box
that nobody has time to read. This app reads that text, in Hinglish, Hindi or English, and
turns it into:

1. **A reason for every row**: fit, quality, not as shown, wrong or damaged, delivery,
   changed mind, or unclear.
2. **A detail for the reasons you can act on**: too small or too large and where (bust, waist,
   length...), which quality issue, or which part differs from the photo.
3. **A ranked action list**: which vendor, product and size has the most returns for the same
   problem, with a suggested fix and customer quotes.
4. **A review queue**: the rows the system does not trust itself on, for a person to check.

## Run it on your computer (about five minutes)

You need Python 3.11 or newer and an OpenRouter key (https://openrouter.ai/keys).

```
git clone https://github.com/amrinderSidhu8/returns-classifier.git
cd returns-classifier
python -m venv .venv
```

Turn the environment on:

- Windows: `.venv\Scripts\activate`
- Mac or Linux: `source .venv/bin/activate`

Then:

```
pip install -r requirements.txt
```

Add your key: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace
the placeholder with your key.

Check the key works (one small call, a few seconds):

```
python test_call.py
```

Start the app:

```
streamlit run app.py
```

It opens at http://localhost:8501 with the sample data already loaded. Press
**Read the returns**.

## What it expects

A CSV with one row per return and these columns. Extra columns are ignored.

| Column | What it holds |
|---|---|
| `return_id` | A unique ID for the return. Repeated IDs are dropped |
| `vendor_id` | The vendor, for example V-07 |
| `category` | The product type, for example Kurti |
| `size_ordered` | The size the customer ordered. May be blank |
| `dropdown_reason` | The reason picked from the dropdown. Only rows that say `Other` are read |
| `other_text` | What the customer typed |

`data/sample_returns.csv` is a made-up file in this shape (1,000 returns, 440 under "Other").
`data/gold_labels.csv` holds the correct answer for each row and is used only for testing.

## How it works

| Step | Done by | Why |
|---|---|---|
| Drop repeated rows, skip blank or symbol-only text | Code | A fixed rule. No model call is spent |
| Triage: pick one reason, a confidence and the exact words that show it | Small model | Reading messy mixed-language text needs judgment |
| Check the answer: right shape, and the quoted words really are in the text | Code | A rule. Catches made-up evidence |
| Second opinion on rows that are not high confidence, have two reasons, or failed | Strong model | Only the hard rows need the costly model |
| Route to one of three detail readers (fit, quality, not as shown) | Code | The route follows from the reason. No judgment needed |
| Detail: direction and body area, quality issue, or mismatched aspect | Small model | Judgment again, on a narrow question |
| Count by vendor, product and size, rank, attach a fix | Code | Arithmetic and a lookup table |

Patterns used: **routing** (the reason decides which detail reader runs) and **prompt
chaining** (the detail step is given the reason and evidence from triage).

## Models and cost

| Model | Job | Temperature |
|---|---|---|
| `google/gemini-3.5-flash-lite` | Triage and detail, every row | 0 |
| `anthropic/claude-sonnet-5.5` | Second opinion, about 7% of rows | 0 |

Both run through OpenRouter. Temperature is 0 so the same text gets the same label each run.
To change a model or a price, edit `pipeline/config.py`.

On the full sample (440 rows) one run cost $0.17:

- Gemini: 329,650 tokens in × $0.15 per million + 23,750 out × $1.25 per million = $0.079
- Sonnet: 34,830 tokens in × $2 per million + 2,124 out × $10 per million = $0.091
- $0.17 ÷ 440 rows = $0.00039 per row
- At Dhaga's 6,500 "Other" rows a week: about **$2.50 a week**

The app's **Run report** tab shows the same figures for every run.

## What it does when something goes wrong

| Situation | What you see |
|---|---|
| The file is missing a needed column | A red message naming the column. Nothing runs |
| No "Other" rows in the file | A message saying there is nothing to read |
| No API key | A message saying which key to add |
| Blank, emoji or symbol-only text | Marked `skipped_junk`, labelled unclear, no model call |
| The model's answer is the wrong shape, or its quoted words are not in the text | One retry, then the strong model. If that also fails the row is `unclassified` and goes to the review queue with the error |
| The strong model fails | A red banner with the error. Those rows keep the small model's answer and go to the review queue |
| The model is still unsure, or the text gives no reason | The row goes to the review queue with the reason why |
| A dropdown reason the app does not know | Named in a warning and left out of the action list |

No row is dropped silently. Every row appears in **All rows** with a status.

## Test the accuracy

```
python eval/run_eval.py --limit 50     # quick check, about two cents
python eval/run_eval.py                # all 440 rows, about $0.17
```

This runs the whole pipeline against `data/gold_labels.csv` and prints accuracy, cost and the
top of the action list. Every row is written to `eval/results.csv`.

Latest full run: reason correct on 438 of 440 rows. Fit direction 99%, fit area 94%, quality
issue 98%, not-as-shown aspect 96%. The five problems planted in the sample data came out as
ranks 1 to 5 of the action list.

**Read this number with care.** The sample text is generated, so it is cleaner than real
customer text. Treat it as a ceiling until it is tested on real rows.

## Known limits

- The action list ranks by the number of returns, not the return rate, because the file has
  no order counts.
- The dropdown reasons it understands are the ones in the sample file. Others are reported
  and left out.
- "Test a single return reason" assumes a Kurti in size M.
- It has not been run on real Dhaga data.

## What is in the repository

```
app.py                  the Streamlit screen
pipeline/config.py      models, prices, settings
pipeline/triage.py      step 1: the reason
pipeline/escalate.py    step 2: second opinion from the strong model
pipeline/detail.py      step 3: routing and detail
pipeline/run.py         runs the steps in order and builds the review queue
pipeline/aggregate.py   the ranked action list
eval/run_eval.py        accuracy and cost test
data/                   sample returns and the answer key
docs/discovery_note.md  why we picked this problem
test_call.py            checks your key
test_strong.py          checks the strong model's settings
```
