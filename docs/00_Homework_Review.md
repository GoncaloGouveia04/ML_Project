# Review of Part 1 (Homework, 12.1 / 20)

This file is a check of the homework notebooks in `notebooks/` (01 to 04) against the homework brief, the class material in `ML_DSAA_Practicals/`, and the data itself. The goal is to understand where the points went, and to make sure Part 2 does not repeat the same problems.

## Is 12.1 / 20 bad?

It is not a disaster. It is a pass in the middle of the scale (about 60 %). But it is below what the amount of work deserved. The notebooks did a lot, especially in feature selection, yet points were lost in four places:

1. **Consistency.** The notebooks contradict each other (details below).
2. **Missed insights.** Several important facts in the data were never noticed.
3. **Evaluation.** The model was never compared with the simplest possible baseline for its own metric.
4. **Writing.** Many typos and some wrong statements in the exploration notebook.

The homework rubric was: exploration 3, preprocessing 5, feature selection 3, model and assessment 4, PDF 5, plus 1 extra point for the Kaggle top 5. We do not have the PDF or the per-section marks, so the estimates below are about the notebooks only.

## What was good (keep it)

- **The validation idea was right.** Splitting by date and leaving a 90-day gap between training and validation is exactly what this problem needs; a shuffled split would leak. Part 2 keeps this idea and makes it the single design of the whole project.
- **Preprocessing was fitted inside the folds** with a `Pipeline`, never on test data.
- **Invalid values became missing values** (the 1e9 placeholder and the negatives), as in week 3.
- **Feature selection was rigorous**: paired differences with standard errors, stability across splits, the check of rare categories that identify customers (transport 11 used by one customer), and the warning that a Random Forest selected `random_noise`.
- **Small `assert` tests** checked each helper function.

## What cost points

### Exploration (3 points)

| Problem | Why it matters |
|---|---|
| Typos and wrong sentences: "collumns", "unliquely", "targetr", "apropriete"; "1 if the customer bought again **after** 90 days" (it is **within** 90 days); "the **model** is already evenly balanced" (the *data* is balanced) | The rubric explicitly grades writing quality. A wrong definition of the target is a red flag for a reader. |
| The 10 "duplicated" rows were never inspected | They are all among the 65 empty rows (same customer, every field missing). They are not duplicate purchases at all. |
| `Total a Pagar` and `Total Bruto` are exact sums of other columns | Never checked; the selected feature set later kept all four totals. |
| The target is defined by the customer's **next purchase** (99.1 % of rows) | This is the key to leakage and to the validation design, and it was never shown. |
| No look at the target over time | The repeat rate drifts from 0.48 to 0.63 between years; this explains why scores vary between folds. |
| The date was only used for splitting | Purchases follow a four-month cycle (months 3, 7, 11) with a clear effect on the target. |
| Correlations computed with Pearson on very skewed, uncleaned columns | Pearson is dominated by the extreme values; the conclusion "current amounts are unrelated" was right but not well supported. |
| The test file was not checked | It has the same placeholders and negative values as train, which is what justifies a rule instead of row deletion. |

### Preprocessing (5 points)

| Problem | Why it matters |
|---|---|
| **Exploration said "remove the 65 empty rows and the duplicates"; preprocessing kept all 6,534 rows** | Direct contradiction between notebooks. The 65 rows have no features and no date: after imputation they are 65 copies of "the median customer" with random labels. |
| A missing flag for **every** numeric column (29 flags) plus `history_unknown`, which duplicates `prior_purchase_count_missing` | Most flags are identical to each other (the 65 empty rows) or mark random missingness that carries no signal. Feature selection then had to throw them away. |
| `clientes.csv`: exploration said `customer_since_year` and `commercial_region` "could be useful", preprocessing dropped the whole table with "we don't know when it was updated" | The brief asks to *decide whether* the table helps. It was never tested. |
| The same functions (`clean_values`, `add_features`, `build_preprocessor`) were copied into three notebooks | Any change has to be made three times; the copies can drift apart. |
| Three different validation designs: one holdout in 02, five splits in 03, ten date blocks (last four) in 04 | Results from different notebooks cannot be compared, and a reader has to understand three designs. |
| Missing values that could be rebuilt exactly (average purchase value = total / count) were imputed with a median | A guess replaced a value that was known. |

### Feature selection (3 points)

| Problem | Why it matters |
|---|---|
| The chosen selector is a **single unpruned decision tree** with a "mean" importance threshold | A deep tree is unstable (week 9): small changes in the rows give a different tree and different importances. Impurity importances also favour continuous columns, which is the same bias that made the Random Forest pick `random_noise`. |
| The selected set kept `Total a Pagar`, `Total Bruto`, `Total Liquido` and `Total Imp_` together | This contradicts the notebook's own redundancy analysis (correlations up to 1.0). |
| The selector (a tree) and the evaluation model (logistic regression) were different, and F1 at 0.5 was the only score | F1 at a fixed threshold mixes ranking quality with threshold choice, so small differences between methods are mostly noise. |
| About 1,500 lines for a homework section worth 3 points | The brief asks for conciseness; the decision gets lost in the length. |

### Model and assessment (4 points)

| Problem | Why it matters |
|---|---|
| **The F1 floor was never reported.** With 56 % positives, predicting "repeat" for everyone already gives F1 ≈ 0.72 | The reported F1 of 0.76 is only about 0.03–0.05 above "no model at all". Week 5 says this explicitly: "F1 means little without that floor beside it". |
| Only one model (logistic regression) with one setting (`C`) | Allowed for the homework, but there was no comparison to show the choice was reasonable. |
| Metrics shown only for the last fold; no ROC-AUC or balanced accuracy | The course always reports F1, ROC-AUC and balanced accuracy together. |
| No look at the decision threshold | With a majority positive class, F1 is usually higher below 0.5 (+0.01 to +0.02 here). |
| Leftovers: an absolute path from another computer in the output (`/home/yoru/...`), `submission.csv` instead of the naming convention, the README pointing to a `notebooks/main.ipynb` that does not exist | Small, but they make the work look unfinished. |

## How Part 2 fixes each point

| Part 1 problem | Part 2 |
|---|---|
| Contradictions between notebooks | One shared module (`project_pipeline.py`) and one validation design (`development_split`) used by every notebook. |
| Empty rows kept | Dropped, with the evidence (no date, no features, all the "duplicates", none in test). |
| Missed insights | Target construction / leakage audit, exact identities, structural vs random missingness, yearly drift, four-month cycle, rare levels that identify customers. |
| `clientes.csv` not tested | Joined and tested on the five windows with two models; no measurable gain, so not joined. |
| Unstable selector | A vote of three different methods, refitted per window, judged by three models, checked with a noise probe. |
| No floor, F1 only | ROC-AUC to compare models, F1 always next to its floor, balanced accuracy, a business-rule baseline, the threshold chosen on validation, a sealed holdout used once. |
| One model | Twelve algorithms from class, six tuned with 60-candidate random searches, a soft-voting check. |
| Writing | Short markdown conclusions with the actual numbers after every result. |

## Note on what the course rules require

- Notebooks must run top to bottom with ascending execution numbers: all six notebooks were run in order and pass this check.
- "Refrain from providing theoretical explanations of topics covered in class" in the notebook: the theory is in these `docs/` files, not in the notebooks.
- "If something is good enough to be mentioned, it is also good enough to know": every technique used is explained in the docs, including why alternatives were rejected. Read them before the discussion.
