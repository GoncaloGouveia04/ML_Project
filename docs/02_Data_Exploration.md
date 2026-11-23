# Part II.a — Data exploration

Notebook: `01_Business_and_Data_Exploration.ipynb`, Section II.

## 0. What comes from Part 1

| Kept from Part 1 (`notebooks/01_Exploration_notebook.ipynb`) | Changed | New |
|---|---|---|
| Table shapes and periods, missing-value table, the 65 empty rows, the 603 first purchases, the 1e9 placeholders and negatives, class balance, skew, correlation with the target, redundant pairs, categorical repeat rates, the `clientes.csv` checks | The 10 "duplicates" are empty rows; the empty rows are now really dropped; the target is "within 90 days" (Part 1 wrote "after"); the `clientes.csv` join is tested instead of assumed useless | The leakage analysis of the target (next purchase of the same customer), the exact-sum columns, the four-month cycle, the insight → decision table |

## 1. What this part does

It describes the data, finds its problems and extracts the facts that shape every later decision. The brief warns: "avoid adding visualizations and elements that do not address the problem at hand", so every table and figure in the notebook ends with a decision.

## 2. Findings and the decisions they drive

| # | Finding (how it was measured) | Decision | Why this decision |
|---|---|---|---|
| 1 | Train: 6,534 rows, 610 customers, 2015-07 → 2025-02. Test: 986 rows, 2025-05 → 2026-10, 94.5 % of test purchases from known customers | Time-based validation; customer behaviour features matter | The test is a *later* period of mostly *known* customers. |
| 2 | Repeat rate 0.559 overall, **0.48–0.63 between years** | Five yearly windows; F1 always with its floor | Some years are harder; one split could mislead. |
| 3 | **65 rows with every field missing**, including the date. The 16 rows that look duplicated are all among them. Test has none | Drop them | No information, cannot be placed in time, absent from test. |
| 4 | 1e9 placeholders and negative values in 4 columns, **in train and test** | Mark as missing (rule, not deletion) | Week 3: invalid values first become missing; the same rule must work on test rows. |
| 5 | `Total a Pagar = Liquido + Imp` and `Total Bruto = Liquido + discounts` in 100 % of rows | Drop both | Exact copies add only redundancy. |
| 6 | `prior_mean_purchase_value = total / count` (99.2 %), `prior_mean_gap_days = (tenure − days since previous) / (count − 1)` (100 % where valid), `days_since_previous = tenure` when count = 1 (99.3 %) | Rebuild missing/invalid values from these definitions | A known value is better than a median guess. |
| 7 | Missingness is **structural** (first purchase → no history; one earlier purchase → no gap) or **random** (~195 per column, repeat rate like comparable rows) | Median + `first_purchase` flag; `Missing` level for categories | Structural missingness is information; random missingness is not. |
| 8 | **Target = next row of the same customer within 90 days** (99.1 %) | Only past information as features; no `customer_id`; chronological validation with a gap | This is how leakage would happen. |
| 9 | Rhythm dominates: average gap < 30 days → 89 % repeat; > 90 days → 29–38 %; 13+ purchases last year → 98 %; first purchases → 33 % | Ratio features describing rhythm; `log1p` | The model must see frequency and recency clearly. |
| 10 | Current purchase amounts are similar in both classes; long right tails everywhere | `log1p` before scaling | Compress tails without deleting rows. |
| 11 | **Four-month cycle**: months 3, 7, 11 are the busiest every year 2018–2024, and purchases there repeat less (44–49 % vs up to 72 %). In test, July/November 2025 and July 2026 are busy, March 2026 is not | `purchase_month` as a categorical candidate | Anonymised dates keep intervals, so a cycle survives; tested later because it may be weaker in test. |
| 12 | `transport_method` has 80 levels, 40 used by one customer; `warehouse`, `document_series`, `commercial_zone` almost uninformative | One-hot inside the pipeline; rare-level merging tuned | Rare levels can identify customers; let validation decide. |
| 13 | `random_noise`: r = −0.012, p = 0.34 | Not an input; used as a probe in feature selection | No signal; a selector that keeps it is selecting noise. |
| 14 | `clientes.csv`: mostly empty or identifying columns; `customer_since_year` ~ tenure (r = 0.92); second country ≈ zero-tax purchases | Test the join on the windows (notebook 02) | Do not accept or reject it blindly. |

## 3. Core concepts

### Four different data problems (week 3)
- **Missing**: no value recorded.
- **Invalid**: a value that cannot be true (negative days, the 1e9 placeholder). Turn into missing *first*, before counting missing values.
- **Extreme (outlier)**: unusual but possibly real (a very large order). Investigate, do not auto-delete; the IQR rule is a screening tool, not a deletion rule. `log1p` reduces their weight while keeping every row.
- **Repeated**: the same entity appears more than once. Here the "duplicates" were empty rows; the real repeated entity is the *customer*, which is handled by the time-based validation (a model never sees a customer's future).

### Structural vs random missingness
- *Structural*: the value does not exist because of what the row is (a first purchase has no "days since previous purchase"). The fact that it is missing tells you something (new customer). Keep that information with a flag or with a feature that already encodes it (`prior_purchase_count = 0`).
- *Random*: the value existed but was not recorded, independently of the outcome. We checked this by comparing the repeat rate of rows with a missing value against comparable rows: they are similar, so a flag would carry no signal and a simple median is enough.

### How the target can leak
A feature leaks if it uses information that only exists after the prediction moment. Here the label is "is there another purchase in the next 90 days". Examples of leaky features we deliberately did not build: number of rows per customer in the whole table, customer averages over the whole table, "days until the next row", target encoding of `customer_id` computed on all rows. The supplied history columns are safe because they only look backwards.

The same structure exists inside `test.csv`: for a test purchase, the next purchase of the same customer may also be in `test.csv`. Using it would mean reading the future at prediction time, so we do not, even though it could raise the Kaggle score.

### Relevance indices used
- **Point-biserial correlation**: Pearson correlation between a numeric column and a 0/1 target.
- **Cramér's V**: strength of association between two categorical variables, from the chi-square statistic, between 0 and 1. The chi-square p-value is unreliable when expected counts are below about 5 (many rare transport levels), so V is used as a description, not a test.
- Both measure one variable at a time; they cannot see interactions or redundancy.

### Why `log1p`
`log1p(x) = log(1 + x)` is defined at 0 (many zero counts) and compresses long right tails: a purchase of 100,000 no longer weighs a thousand times more than one of 100. It is stateless (learns nothing), so it cannot leak. Trees are unaffected by it (they only use the order of values); linear models, KNN and MLPs benefit.

### Why the seasonality survives anonymisation
The brief says the dates are anonymised but "temporal order and intervals are preserved". Shifting every date by the same amount changes the month labels but keeps a cycle of four months a cycle of four months, and train and test are shifted together.

## 4. Alternatives considered

| Alternative | Why not |
|---|---|
| Delete outliers with the IQR rule | Week 3: it would delete real large orders; the long tails are normal for a wholesaler. `log1p` instead. |
| Keep the 65 empty rows (homework) | They contain only the label; after imputation they are noise with random labels. |
| Missing flags for every column (homework) | Most missingness is random or duplicates the empty rows; flags would add ~30 useless columns. |
| Use `customer_id` | An identifier, not a behaviour; would let a model memorise customers. |
| Use future rows of the same customer (in train or test) | Target leakage. |
| Profiling packages (ydata, pandas-profiling) | Allowed, but only insights that are acted upon count; we kept every analysis tied to a decision instead. |

## 5. Questions you may get in the discussion

- *How did you find the duplicates?* `train.drop(columns="ID").duplicated(keep=False)`: 16 rows; all 16 are among the 65 empty rows.
- *Why not delete rows with invalid values?* The same invalid values appear in test; we cannot delete test rows, so the treatment must be a rule that works on any row.
- *What does a correlation of 0.92 between `customer_since_year` and tenure mean for the join?* The column repeats information already in `customer_tenure_days`; the join was tested anyway and gave no measurable gain.
- *Why is the four-month cycle not leakage?* The month of the current purchase is known at the prediction moment; it describes *when* the purchase happens, not what happens next.
- *Why are first purchases important?* They repeat only 33 % of the time and have no history; notebook 05 shows the model is almost blind on them (AUC 0.55).
