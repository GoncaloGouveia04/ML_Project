# Part III.a — Feature selection

Notebook: `03_Feature_Selection.ipynb`.
Code: `target_correlation`, `CorrelationFilter`, `VoteSelector`, `filter_selector`, `wrapper_selector`, `embedded_selector`, `make_vote_selector` in `project_pipeline.py`.

## 0. What comes from Part 1

| Kept from Part 1 (`notebooks/03_Feature_Selection.ipynb`) | Changed | New |
|---|---|---|
| Correlation filter with redundancy, RFE, L1, tree importance with the **mean** threshold (Part 1 found that the median keeps everything), the vote of three families, `paired_delta` with the 2 × SEM rule, refitting selection inside every split, testing `random_noise` | ROC-AUC instead of F1; three reference models instead of one logistic regression; wide budgets instead of 10–30 columns; the exported strategy is the vote, not tree importance; VarianceThreshold, SelectKBest and TypedFilter not repeated (reasons in Section 8) | The noise probe in every run; stability (Jaccard) of the kept sets |

## 1. What this part does

After preprocessing, each window has 113–147 columns (about 34 numeric, the rest one-hot levels). This part decides which of them the models should see, using methods from week 4, and exports a **strategy** that is refitted inside every training split.

## 2. Result in one paragraph

No selection method improved accuracy (as in class). Narrow selections (20–40 columns) **lost** accuracy for the tree ensemble. A **vote** of three methods with wide budgets (keep a column if at least 2 of: correlation filter k=60, RFE k=60, L1 with C=0.3) kept about 55 of 130 columns, was **inside the noise for all three reference models**, and rejected the pure-noise probe in every window. That vote is the exported strategy.

## 3. Protocol

- Selection reads the target, so it is **fitted on the training rows of each window only** (fitting it on all rows would leak — week 4).
- Each strategy is judged with **three reference models** (logistic regression, ExtraTrees, histogram gradient boosting): the final algorithm is not chosen yet, and different families use features differently.
- Metric: ROC-AUC per window, compared with "all columns" (paired difference, 2 × SEM rule), plus the number of columns kept, **stability** (mean Jaccard overlap |A ∩ B| / |A ∪ B| between the sets kept in the five windows) and whether `random_noise` was kept.
- **Noise probe**: `random_noise` (r = 0.02 with the target) is added as an extra input in every run. A method that keeps it is selecting noise.

## 4. The methods (week 4)

### Filter — correlation filter
1. Score each column's **relevance** = |correlation with the target|. For a numeric column that is the point-biserial r; for a 0/1 one-hot column it is the phi coefficient, which equals Cramér's V of its 2×2 table — so one formula gives the class's type-matched index for every column.
2. Walk down the ranking; **skip** a column whose |correlation| with an already kept column exceeds 0.8 (**redundancy**).
3. Stop at *k* columns.

Strength: fast, model-free, handles redundancy. Weakness: one column at a time, linear, cannot see interactions ("late *and* usually frequent").

### Wrapper — Recursive Feature Elimination (RFE)
Fit a logistic regression, drop the 5 columns with the smallest absolute coefficients, refit, repeat until *k* remain. Judges columns *together* through a fitted model. Weakness: depends on one model's view (linear), slow, least stable (Jaccard 0.51–0.54).

*Why RFE and not RFECV / SequentialFeatureSelector?* Both cross-validate internally with shuffled folds, which would mix time inside our training windows (week 4 warns about internal CV leakage). RFE has no internal CV; the budget *k* is chosen outside, on the windows.

*Why `liblinear`?* Same logistic-regression model, faster solver for small data (about 20× faster, 93 % identical selections); speed matters because the vote is refitted for every window and every tuning candidate.

### Embedded — L1 logistic regression
The L1 penalty (sum of absolute coefficients) pushes weak coefficients to **exactly zero**, so fitting the model *is* the selection; `SelectFromModel(threshold=1e-8)` keeps the non-zero ones. Smaller `C` = stronger penalty = fewer columns. Geometric intuition: the L1 constraint region is a diamond whose corners lie on the axes, so the optimum often lands where some coefficients are 0; L2's round region does not.

**Finding:** L1 kept `random_noise` in 3/5 windows at C = 0.05 and 5/5 at C = 0.3. With correlated inputs, L1 picks somewhat arbitrarily among them, and a small spurious correlation of the noise in a training window can be enough to keep it. L1 alone is therefore not a safe selector here.

### Embedded — tree importance
`SelectFromModel(ExtraTrees, threshold="mean")` keeps columns with above-average impurity importance. The most stable method (0.85), but it lost accuracy for ExtraTrees (−0.009). Week 9: impurity importance is not cause and can be "stolen" between correlated columns; it also favours continuous columns with many split points. The homework's median threshold failed because half of the importances were exactly zero.

### Combining — vote vs sequence
- **Sequence (funnel)**: each method only sees what the previous kept; any step can veto a column for good.
- **Vote**: each method chooses independently; a column needs ≥ 2 of 3 votes. One method's blind spot cannot remove a useful column, and one method's mistake (L1 keeping noise) is out-voted.
The class exported the vote because it was the only combination with no measurable loss; we did the same. We did not run the sequence: in class it lost accuracy, and every narrow single method already lost here, so a funnel (which narrows further) had no chance of passing.

## 5. Results

| Strategy | Columns | ΔAUC LogReg | ΔAUC ExtraTrees | ΔAUC HistGB | Stability | Noise kept |
|---|---|---|---|---|---|---|
| all columns | 130 | 0 | 0 | 0 | 0.88 | 5/5 |
| filter k=30 | 30 | −0.001 | **−0.009** | **−0.002** | 0.71 | 0/5 |
| filter k=60 | 60 | +0.002 | −0.002 | +0.005 | 0.70 | 0/5 |
| RFE k=30 | 30 | **−0.018** | **−0.027** | −0.002 | 0.51 | 0/5 |
| RFE k=60 | 60 | −0.003 | −0.006 | −0.002 | 0.54 | 0/5 |
| L1 C=0.05 | 23 | −0.004 | **−0.012** | **−0.007** | 0.63 | 3/5 |
| L1 C=0.3 | 54 | −0.004 | −0.003 | +0.000 | 0.69 | 5/5 |
| tree importance | 40 | −0.002 | **−0.009** | −0.004 | 0.85 | 0/5 |
| **vote (≥ 2 of filter 60, RFE 60, L1 0.3)** | **55** | **+0.000** | **−0.002** | **+0.001** | 0.61 | **0/5** |

Bold differences are "worse" beyond 2 × SEM.

**Why do narrow selections hurt the tree ensemble?** ExtraTrees spreads its splits over many columns and averages many trees; removing columns that are individually weak but add small, different pieces of information (rare levels, month dummies, current-purchase details) removes diversity it uses. Linear models suffer less because they already shrink weak columns.

**Why choose the vote over the filter at 60, which looks similar?** Both are inside the noise. The filter alone is a bit more stable (0.70 vs 0.61) and simpler. We chose the vote for the reason given in class: the algorithm is not chosen yet, and a column agreed on by two different instruments (a correlation ranking, a fitted linear model, a penalty) depends less on one method's blind spot — the filter cannot see joint effects, L1 keeps noise. The filter at 60 is a valid alternative to mention.

**Honest caveat.** The wide budgets were chosen after seeing the narrow ones lose on the same windows, so the "no loss" verdict is slightly optimistic. The sealed holdout (notebook 04) is the independent check, and the final model there is fine.

## 6. What the vote keeps (on all development rows)

61 of 155 columns: most history and current-purchase information, 9 of 12 month levels, and only 27 of 108 one-hot levels (most single-customer transport levels go). Dropped numeric columns are mostly redundant copies: `purchase_count_180d` (vs 365d, r = 0.92), `prior_mean_gap_days` (vs `purchases_per_year`, 0.86), `customer_tenure_days` (vs `prior_total_spend`, 0.91), plus weak ones (`Total Portes`, `N_ Detalhes`, `ticket_vs_history`). `random_noise` gets one vote (from L1) and is rejected.

## 7. Identifier-like fields (asked explicitly by the brief)

| Field | Treatment | Evidence |
|---|---|---|
| `ID` | Row label only | Unique per row, no meaning. |
| `customer_id` | Not an input | An identifier: its numeric value means nothing; as a category it would let a model memorise customers; and the target is built from the same customer's future rows. |
| `purchase_date` | Not an input directly; used to split and to derive the month | A raw date would let a model learn "this period was good", which cannot extrapolate to 2025–2026. |
| `random_noise` | Not an input; used as a probe | r = −0.01 (p = 0.34); the vote rejects it in every window. |

## 8. Alternatives considered

| Alternative | Why not |
|---|---|
| No selection at all | Inside the noise too, but 130 columns including the probe and dozens of single-customer levels. Selection halves the width at no measurable cost and removes noise. |
| `SelectKBest` (k best by a univariate score) | Same ranking idea as the filter but without the redundancy rule; the homework showed redundancy matters. |
| `VarianceThreshold` | Only removes constant columns; there are none after one-hot fitting per window. (In Part 1 it removed the `_missing` flags, which is why Part 2 no longer creates them.) |
| `TypedFilter` (Part 1, 3.5.3) | Keeps or drops a categorical variable as a whole. Here the vote can keep useful single levels (for example one payment type) and drop the rest, which the whole-variable rule cannot do. Part 1 found it inside the noise. |
| PCA / dimensionality reduction | Keeps variance, not signal, and loses the column names needed to explain the model (week 4). |
| RFECV / SequentialFeatureSelector | Internal shuffled CV mixes time inside the training window. |
| Random-forest importance alone | Week 9 bias towards continuous and noisy columns; the homework saw it select `random_noise`. |

## 9. Questions you may get in the discussion

- *Why refit the selection in each fold?* Because it uses the target. Selecting on all rows and then validating would let validation rows choose the columns (leakage). We export the strategy, not a list of columns.
- *What is the difference between relevance and redundancy?* Relevance: is a column related to the target? Redundancy: does it repeat another column? A filter needs both; a univariate ranking alone cannot see redundancy.
- *Why does L1 set coefficients to exactly zero and L2 not?* The L1 penalty has a corner at zero (its derivative does not vanish there), so small coefficients are pushed exactly to zero; L2's penalty is smooth and only shrinks them.
- *What does stability 0.61 mean?* On average, two windows' kept sets share 61 % of their union. Lower stability is expected when RFE is a member; what matters is that the core history columns are always kept.
- *Selection did not improve accuracy — so why do it?* A narrower model is cheaper, easier to explain and less exposed to noise and to single-customer levels; the brief also requires a justified strategy. We show it costs nothing measurable.
