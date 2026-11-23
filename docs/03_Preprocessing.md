# Part II.b — Cleaning and preparation

Notebook: `02_Preprocessing.ipynb`.
Code: `clean_purchases`, `add_features`, `prepare_frame`, `FeatureBuilder`, `build_preprocessor` in `project_pipeline.py`.

## 0. What comes from Part 1

| Kept from Part 1 (`notebooks/02_Preprocessing.ipynb`) | Changed | New |
|---|---|---|
| The invalid-value rule of `clean_values` (week 3); `log1p` before median imputation; median for numbers, a constant missing label and one-hot for categories, `StandardScaler` (`fill_missing` / `fold_matrices`, week 4); `first_purchase` and value per order line; leaving out `customer_id`, `purchase_date`, `random_noise` | Empty rows dropped (Part 1 kept them); the 28 `_missing` flags removed (Part 1's own `VarianceThreshold` showed they were constant inside the splits); the hand-written fold function became a scikit-learn `Pipeline` + `ColumnTransformer` (week 5) | Rebuilding values from their definitions; the exact-sum columns dropped; six ratio features and `purchase_month`; every optional choice (including `clientes.csv`) tested on the windows |

## 1. What this part does

It turns the raw rows into a numeric matrix that any of the twelve algorithms can use, in a way that is (a) reproducible, (b) identical for train, test and a future purchase, and (c) never learns anything from validation or test rows.

## 2. The two kinds of steps

| Kind | Steps | Learns from data? | Where it runs |
|---|---|---|---|
| **Stateless** | cleaning rules, rebuilt values, engineered features, `log1p`, dropping columns | No | `FeatureBuilder` (first step of the pipeline) |
| **Learned** | median imputation, scaling, `Missing` level, one-hot categories, rare-level merging | Yes (medians, means, standard deviations, the list of levels) | `ColumnTransformer` inside the pipeline, **refitted on the training rows of every split** |

Week 3's golden rule: anything *computed* from the data must be computed on training rows only, then applied to other rows. What we save is the **recipe** ("fill with the training median"), not a number. A stateless step is safe anywhere because it computes nothing from other rows.

## 3. Every step, and why

### Cleaning (stateless)
1. **Invalid → missing.** Values ≥ 1e9 (a placeholder) and negative amounts/days become `NaN`. Same rule for every row.
2. **Rebuild by definition** (only where the value is missing):
   - `prior_mean_purchase_value = prior_total_spend / prior_purchase_count`: recovers all 32 invalid and all 194 randomly missing values; the 603 left are exactly the first purchases.
   - `prior_mean_gap_days = (tenure − days_since_previous) / (count − 1)`: 203 values.
   - `days_since_previous_purchase = tenure` when there is one earlier purchase: 22 values.
   - `spend_365d = 0` if no purchase in 365 days, `= prior_total_spend` if all earlier purchases were in the last 365 days: 82 values.

   *Why:* week 3 says cleaning comes from understanding what each column means. When a column is a definition of other columns, the true value is known; a median would replace it with a guess.
3. **Drop** `Total a Pagar` and `Total Bruto` (exact sums), and the non-features `ID`, `customer_id`, `purchase_date` (after taking the month), `random_noise`.

### Engineered features (stateless)
Week 4: engineering creates or transforms features; every change must prove itself on held-out data; engineer **before** selecting, because a selector cannot keep a feature that does not exist.

| Feature | Idea | Univariate |r| |
|---|---|---|
| `purchases_per_year` | long-run frequency | 0.37 (4th of all columns) |
| `discount_rate` | commercial conditions (discounts mark established accounts) | 0.22 |
| `first_purchase` | structural-missingness flag | 0.15 |
| `recent_activity_share` | speeding up or slowing down | 0.11 |
| `value_per_detail`, `tax_rate` | order-line size; exports pay no tax | 0.08–0.09 |
| `recency_ratio` | late compared with the usual rhythm | 0.04 alone, but non-linear |
| `ticket_vs_history` | unusually large order | 0.02 |
| `purchase_month` (categorical) | four-month cycle | Cramér's V 0.18 |

All are ratios of columns in the **same row**, so they only use the past.

### `log1p` (stateless)
Applied to all amounts, counts, day counts and the skewed ratios. Keeps every row, compresses long tails; needed by linear models, KNN and MLPs, harmless for trees (they use only the order of values).

### Imputation (learned)
- Numeric: **median of the training rows**. Robust to the long tails (a mean would be pulled by large orders).
- Categorical: a constant **`Missing` level**. It keeps the fact that the value was not recorded and lets the model decide whether it matters.
- The `first_purchase` flag tells the models when a median stands in for a history that does not exist.

### Scaling (learned)
`StandardScaler` (subtract the training mean, divide by the training standard deviation) on numeric columns. Needed by models based on **distances** (KNN), **penalties on coefficients** (logistic regression with L1/L2, MLP with `alpha`) and gradient-based fitting (MLP). Trees ignore it. One recipe for all models keeps the comparison fair.

### One-hot encoding (learned)
- One 0/1 column per level, because the categories have no order (week 4: ordinal encoding would invent one).
- `handle_unknown="infrequent_if_exist"`: a level never seen in training goes to the "infrequent" column if there is one, otherwise all zeros. Test has 8 such levels (notebook 06).
- `min_frequency`: levels seen in fewer training rows are merged into one "infrequent" column. Default 1 (no merging); tuned in notebook 04 (chosen value 10 for the final model).
- We keep every level (no `drop="first"`): the models are regularised or tree-based, so the redundant column does not cause problems, and an unknown level stays distinguishable from the reference level.

## 4. Optional decisions tested on the five windows

Reference models: logistic regression (linear) and ExtraTrees (tree ensemble). Score: ROC-AUC, paired by window.

| Choice | Result | Decision |
|---|---|---|
| Join 3 columns of `clientes.csv` (country, commercial region, years as customer) | −0.0013 / +0.0016, both inside the noise | Not joined |
| Remove `purchase_month` | ExtraTrees −0.0053, worse in 5/5 windows | Keep the month |
| Remove the 8 engineered ratios | inside the noise for both | Keep as candidates; feature selection decides |
| Merge levels seen < 30 times | worse for both (−0.0084, −0.0014) | No merging by default; tuned later |

**Why not join the customer table if it does not hurt?** No measurable gain, and three costs: a dependency on a table whose update date is unknown (a customer's region could have been recorded *after* some purchases), fine-grained location fields that mainly identify customers, and one more thing to explain and maintain. Week 6/10 advice: when something more complex only matches something simpler, keep the simpler one.

**Why keep rare transport levels if they identify customers?** Because the validation is chronological: the windows test on *later* periods, and the rare levels still help there. Most test customers are known (94.5 %), so customer-specific information does carry into the future. The cost — the model partly relies on customer identity — is measured in notebook 05 (O1).

## 5. Alternatives considered

| Alternative (class-covered) | Why not |
|---|---|
| KNN or iterative imputation | Week 3: fancier is not automatically better. Most numeric gaps are either rebuilt exactly or structural (first purchases), where any imputation is a placeholder anyway and the flag carries the information. |
| Mean imputation | Pulled by the long tails. |
| Mode / proportional imputation for categories | Would hide that the value was missing and invent a level for ~3 % of rows. |
| Ordinal encoding | Invents an order between payment types or transports. |
| Count (frequency) encoding | Compact, but loses which level it was; two different transports with the same frequency become identical. |
| Target encoding | Week 4: "powerful but leaks easily". Here the target is built from future purchases, so any encoding that averages labels by customer-like groups is especially risky. |
| Winsorising (clipping at 1 % / 99 %) | `log1p` already compresses the tails without changing the order of values. |
| PCA | Week 4: keeps variance, not signal; it lost accuracy on every split in class. |
| Missing flags for every column | Random missingness carries no signal (notebook 01, 2.3.4). |
| A time-aware "past repeat rate" of each customer (tested during development, not in the notebook) | +0.002 AUC for logistic regression, −0.0004 for a forest, both inside the noise, and it is a form of target encoding with leakage risk. Not worth it. |

## 6. Questions you may get in the discussion

- *Where is the preprocessing fitted?* Inside the `Pipeline`, on the training rows of each window, and on all training rows for the final model. `test.csv` is only transformed.
- *Why does the number of columns change between folds (113 → 147)?* The one-hot encoder learns its levels from the training rows; later periods contain more levels.
- *Is `log1p` before imputation a problem?* No: it is stateless, and the median commutes with a monotonic transformation (exactly for an odd number of rows, almost exactly otherwise), so imputing before or after the log gives practically the same value.
- *Why not drop rows with missing values?* Structural missingness affects every first purchase (603 rows); dropping them would remove the hardest and most important group.
- *What happens to a payment type never seen in training?* It is mapped to the "infrequent" column (or zeros if the variable has none); the other columns are unaffected.
