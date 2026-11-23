# Part IV — Open-ended section: can the model be trusted to drive commercial actions?

Notebook: `05_Open_Ended.ipynb`.

Grading (4 points): formulation and adequacy of the objectives 0.5, difficulty 1, correctness/efficiency 1, discussion of results 1, alignment between results and objectives 0.5. The brief also says the section must go beyond adding or removing a pipeline step; comparing feature sets, analysing errors and studying calibration are given as valid examples.

## 0. What comes from Part 1

Nothing directly: the homework had no open-ended section. The tools are reused from Part 1 and the earlier notebooks (the same windows, pipeline and paired comparison); calibration comes from week 7.

## 1. Why this question

A Kaggle score says how many labels are right. A sales manager needs to know three different things before acting on the model: *what is it looking at*, *can I believe its percentages*, and *who should I call*. Each objective answers one of them, and each ends with a business recommendation, so results and objectives stay aligned.

All analyses use the final configuration of notebook 04 and **out-of-window predictions** from the five development windows (every prediction comes from a model that never saw that purchase or anything after it). The holdout is used once, only to confirm the campaign result.

## 2. O1 — What information does the model rely on? (ablation study)

### Method
An **ablation study** removes one part of a system and measures how much worse it gets. Here, for each of the seven information groups, the whole pipeline (preprocessing, vote selection, tuned Extra Trees) is refitted **without that group** in every window, and the ROC-AUC is compared with the full model (paired by window). Two block-level variants remove all non-history inputs ("history only") or all history inputs ("no customer history").

This is a *comparison of feature sets*, the example the brief gives. It is different from feature selection: selection asks which columns to keep; the ablation asks what each kind of information is worth to the final model.

### Results
| Removed | ΔAUC | Verdict |
|---|---|---|
| All customer history (only current purchase, context, month left) | **−0.066** | worse, 0/5 |
| Current purchase + context + month (history only) | −0.018 | worse, 0/5 |
| Current purchase | −0.007 | worse, 0/5 |
| Sale context (categoricals) | −0.006 | inside the noise |
| Purchase month | −0.005 | inside the noise |
| Rolling counts | −0.004 | worse |
| Rolling spend | −0.003 | inside the noise |
| History timing | −0.003 | worse |
| History value | −0.001 | inside the noise |

### Interpretation
- **History is the backbone** (−0.066 without it), but **no single history group is indispensable**: timing, rolling counts, rolling spend and value describe the same rhythm in overlapping ways (the redundancy measured in notebook 03), so removing one lets the others stand in.
- **The current purchase is small but unique**: −0.007 in every window. Nothing else carries its information (amount, discounts, tax, lines).
- Without history the model still reaches 0.734, partly because transport/payment levels identify regular customers.
- **Business meaning:** the quality of the history columns matters most; a simpler "history only" model would lose about 0.02 AUC.

### Why ablation and not feature importance
Impurity importances are biased and can be "stolen" between correlated columns (week 9); with four redundant history groups they would split the credit arbitrarily. Refitting without a group measures what is actually lost, with the same validation as everything else. (Permutation importance or SHAP were not covered in class.)

## 3. O2 — Are the probabilities honest? (calibration)

### Concepts (week 7)
- A model is **calibrated** if, among purchases it scores near *p*, a share near *p* repeat. F1 and AUC cannot see this: F1 checks labels, AUC checks order.
- **Reliability table / curve:** group predictions into bins (here 10 bins of width 0.1), compare the mean prediction with the observed repeat rate in each bin. Perfect calibration = the diagonal.
- **Calibration gap:** our summary number — the average |observed − predicted| over the bins, weighted by the number of purchases in each bin (also known as expected calibration error).
- **Confidence vs accuracy:** average probability given to the predicted label vs share of correct labels (the measure used in the week 7 notebook). Confidence ≫ accuracy = over-confident.
- **Log-loss:** −mean[y log p + (1 − y) log(1 − p)], the loss a probability model minimises (week 13); it punishes confident wrong probabilities heavily.
- **`CalibratedClassifierCV`:** splits the training rows into 5 internal folds, fits the model on four, and learns on the fifth a map from score to observed frequency: **sigmoid** (Platt scaling, an S-curve with two parameters) or **isotonic** (any non-decreasing step function, more flexible, needs more data). Wrapped around the whole pipeline, so validation rows are still never seen.

### Results
| | ROC-AUC | Log-loss | Gap | Confidence | Accuracy |
|---|---|---|---|---|---|
| Uncalibrated | 0.795 | **0.538** | 0.038 | 0.69 | 0.72 |
| Sigmoid | 0.795 | 0.543 | 0.040 | 0.71 | 0.72 |
| Isotonic | 0.794 | 0.544 | **0.032** | 0.72 | 0.72 |

- The model is already **reasonably calibrated**: a gap of 0.04, predictions between 0.3 and 0.5 a few points too high, the top bins slightly too low. Confidence ≈ accuracy.
- **Calibration does not help**: sigmoid is worse, isotonic slightly better on the gap but worse on log-loss and AUC. A tree ensemble with leaves of ≥ 5 purchases and 300 averaged trees already produces smooth, moderate probabilities (unlike naive Bayes in class, which was very over-confident). **Decision: no calibration step.**
- **The real risk is drift between years:** in window 4 the model expected 0.59 repeats while only 0.49 happened. A year with unusually few repeats cannot be anticipated from past behaviour. Rankings are less affected than absolute numbers.
- **Business meaning:** the probabilities can be read as percentages for planning, but totals ("how many customers will return next quarter") must be monitored and the model refitted regularly.

## 4. O3 — Where does it fail, and how should it be used?

### Error analysis by segment
| Segment | Purchases | Repeat rate | AUC | Share of all errors |
|---|---|---|---|---|
| First purchase | 213 | 0.26 | **0.55** | 7 % |
| 1–4 earlier | 558 | 0.33 | 0.61 | 21 % |
| 5–19 earlier | 1,259 | 0.42 | 0.70 | **43 %** |
| 20+ earlier | 1,501 | 0.77 | **0.81** | 29 % |
| > 90 days since previous | 1,353 | 0.39 | 0.67 | **46 %** |
| Peak month (3, 7, 11) | 1,493 | 0.46 | 0.78 | 44 % |

- **First purchases are almost unpredictable** (AUC 0.55): with no history there is little to learn from.
- **Most errors come from mid-history customers and long gaps**: customers whose rhythm is changing — the hardest cases, and the most valuable for retention.
- **Loyal customers are predicted well** (AUC 0.81).

### Using the ranking for a retention campaign
Instead of labels, contact the purchases with the **lowest** probability, within a budget. "Lift" = share of non-repeats among those contacted ÷ share of non-repeats overall (what random contact would give).

| Contact lowest | Non-repeats among contacted | Share of all non-repeats reached | Lift |
|---|---|---|---|
| 10 % | 80 % | 17 % | 1.74 |
| 20 % | 76 % | 33 % | 1.66 |
| 30 % | 74 % | 49 % | 1.62 |
| Holdout, 20 % (confirmation) | 78 % | 38 % | 1.89 |

**Business meaning:** with a contact budget of 20 % of purchases, three calls out of four reach a customer who would otherwise not return within 90 days, against fewer than one in two at random. The F1 threshold (0.43) is the wrong operating point for this use: it labels most borderline customers "repeat".

## 5. Key takeaways (aligned with the objectives)

| Objective | Answer | Recommendation |
|---|---|---|
| O1 | Customer history carries most of the signal, spread over four interchangeable groups; the current purchase adds a small unique gain. | Protect the quality of history data; a history-only model is a cheaper fallback (−0.02 AUC). |
| O2 | Probabilities are honest on average (gap 0.04); calibration methods do not improve them; yearly drift is the main risk. | Use probabilities as they are; monitor predicted vs actual repeat rate and refit at least yearly. |
| O3 | Nearly blind on first purchases; weakest for customers whose rhythm is changing; the ranking is strong. | Rank and contact the bottom 10–20 % after each purchase; give first-time customers a standard onboarding action instead of a model score. |

## 6. Limitations (say them before you are asked)

- Five windows give only five paired measurements; small effects (< 0.005 AUC) cannot be separated from noise.
- The ablation removes inputs but the engineered ratios of one group are computed from raw columns of another, so groups are not perfectly separated.
- The campaign analysis assumes contacting a customer does not change their behaviour; it measures *who* to contact, not the *effect* of contacting (that would need an experiment).
- Calibration was assessed on 3,531 predictions; the extreme bins have few purchases (53 in the lowest).

## 7. Questions you may get in the discussion

- *Why is this beyond "adding a pipeline step"?* It answers questions about the model's use (what it relies on, whether its probabilities can be believed, who to contact), each with its own method and evidence; calibration was investigated and deliberately *not* added.
- *What is the difference between ablation and feature selection?* Selection decides which columns to keep; ablation measures how much each kind of information is worth to the final model.
- *Platt vs isotonic?* Platt fits a two-parameter S-curve (robust with little data, but assumes that shape); isotonic fits any non-decreasing step function (flexible, can overfit small calibration sets).
- *Why does calibration not change the AUC much?* Both maps are monotonic, so the order of predictions is (almost) unchanged; isotonic can create ties, which changes AUC slightly.
- *What is lift?* The ratio between the hit rate of a targeted selection and the hit rate of a random one.
