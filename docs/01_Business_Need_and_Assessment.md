# Part I — Business need and model assessment strategy

Notebook: `01_Business_and_Data_Exploration.ipynb`, Section I.
Code: `development_split`, `time_windows`, `describe_folds` in `project_pipeline.py`.

## 0. What comes from Part 1

| Kept from Part 1 | Changed | New |
|---|---|---|
| Training on the past and validating on later purchases, with a 90-day gap between them (homework `03_Feature_Selection` 3.4.1 and `04_Model` Section 2) | Part 1 used the same windows to choose its settings and to report the score; Part 2 adds a **sealed holdout** scored once at the end, and uses yearly windows | The written business framing (Section I of the outline did not exist in the homework); the F1 floor; ROC-AUC as the comparison metric |

## 1. What this part does

It turns the company's question into a machine-learning problem and fixes, **before any model is trained**, how models will be judged: which rows are used for what, and which metrics count.

## 2. Framing the problem (week 1)

| Question | Answer | Why it matters |
|---|---|---|
| Observation | One purchase occasion | Each row gets one prediction. |
| Target | `repeat_purchase_90d` | Binary → classification, even though it is stored as 0/1. |
| Features | Current purchase, customer history up to now, sale context, date | Defines what the model may use. |
| **Prediction moment** | When the purchase is registered | Anything known only *later* is leakage. |
| Decision | Rank purchases; contact the lowest, plan stock for the highest | Decides which errors matter. |

**Why the prediction moment is the central idea.** Week 1: "if a piece of information only becomes available after the moment you need the prediction, using it is data leakage: the model looks great in testing and fails in real life". Notebook 01 shows the target is "the same customer's next row is within 90 days" (99.1 % agreement). So the future rows of a customer literally contain the answer. Everything in the validation design follows from this.

**Which errors matter.** With "repeat" as the positive class:
- False positive (predict repeat, customer does not come back): a customer is lost and nobody acted. Costly.
- False negative (predict no repeat, customer comes back): an unnecessary retention contact. Cheaper.

Both matter, and the right balance depends on the campaign budget, so the model must first **rank** well; the label threshold is a separate business choice (notebook 04, Section 4.7 and notebook 05, O3).

## 3. The assessment strategy

### What we did

1. Drop the 65 empty rows and sort the remaining 6,469 by date.
2. **Sealed holdout** = the last 365 days of `train.csv` (742 purchases, 2024-02-17 to 2025-02-15). Used once, at the end of notebook 04.
3. **Development rows** = purchases before `holdout start − 90 days` (5,564 purchases). No development label can depend on a purchase inside the holdout.
4. **Five yearly validation windows** inside the development rows (Nov 2018 → Nov 2023). For each window, training = all development purchases more than 90 days before the window opens (expanding window).
5. **Final model** = refitted on all 6,469 rows before predicting `test.csv`.

### Why a time-based split and not shuffled K-fold

- **Leakage through the target.** With shuffled folds, a validation purchase from March 2020 can be predicted by a model trained on the same customer's purchases from April 2020, whose existence *is* the March label. Scores would be optimistic and the choice of model biased towards models that exploit it.
- **The real question is about the future.** `test.csv` starts 98 days after `train.csv` ends. Week 5: "the split is a modelling assumption — pick it based on how the model will be used in real life, not on which one gives the best score."
- **Drift.** The repeat rate changes between years (0.48 to 0.63). Validating on later periods measures how well a model copes with that, which shuffled folds hide.

### Why the 90-day gap (a "purge")

A training purchase made on day *d* has a label that depends on what happens until day *d + 90*. If *d + 90* falls inside the validation window, the training label already contains information about validation-period purchases. Dropping training purchases from the last 90 days before each window removes this overlap. It also copies the real situation: at the end of `train.csv` the labels of the last 90 days could only be known because the organisers looked ahead; in production you would have to wait.

### Why expanding yearly windows

- `TimeSeriesSplit` (week 5) is the class tool for "always train on the past, validate on the future". Our windows are the same idea, but defined by **dates** (one year each) instead of row counts, because the gap must be 90 *days*, and because a yearly window contains every month of the four-month cycle.
- **Five** windows give five paired measurements per model: enough to compute a standard error and see whether a difference is real, while keeping at least three years of training data in the first window.
- **Expanding** (train on everything before) rather than **sliding** (only the last *k* years): the final model is also trained on all history, so validation mimics it.

### Why a sealed holdout as well

Week 5: "two jobs must use different data: selecting and estimating". Every choice (features, selection, algorithm, 60 candidates × 6 families, threshold) is made on the five windows, so their best score is optimistic ("you picked whatever happened to do best on those rows, luck included"). The holdout was never used to choose anything, so it gives an honest estimate of the whole procedure. It is the **most recent** year, the closest to the Kaggle test period.

### Alternatives we did not use

| Alternative | Why not |
|---|---|
| Shuffled / stratified K-fold | Leaks through the target (see above) and ignores drift. |
| `GroupKFold` by customer | Answers "new customers?", but 94.5 % of test purchases come from known customers, so it asks the wrong question; it would also still mix time. |
| Leave-one-out | Very expensive and still ignores time. |
| One single holdout for everything | A sample of size one (week 5); no way to tell luck from real differences. |
| Nested cross-validation | The gold standard in class. With time order, each outer window would need inner time windows inside its own past; the early outer windows would have only 2–3 years to split again, and the cost multiplies by the number of inner windows. The sealed holdout answers the same question ("how good is the whole selection procedure on unseen data?") for the period that matters most. |

## 4. Metrics (week 5)

Chosen before seeing any result.

### ROC-AUC — to compare models
- The ROC curve plots the true-positive rate against the false-positive rate for every threshold; AUC is the area below it.
- Interpretation: **the probability that a randomly chosen repeat purchase gets a higher score than a randomly chosen non-repeat purchase.** 0.5 = random, 1 = perfect ranking.
- Why it is the comparison metric here: it does not depend on a threshold. Two models can rank equally well but have shifted probabilities (e.g. `class_weight="balanced"` moves them); F1 at 0.5 would punish one of them for something the threshold can fix later.
- Limit: it says nothing about whether the probabilities are *correct* (calibration, notebook 05).

### F1 of the repeat class — the competition metric
- F1 = 2 · precision · recall / (precision + recall). It ignores true negatives.
- **Floor:** predicting "repeat" for everyone gives precision = *p* (the repeat rate) and recall = 1, so F1 = 2*p* / (1 + *p*). With *p* = 0.56, the floor is 0.72. That is why F1 is always shown next to its floor: an F1 of 0.75 sounds good but is only 0.03 above "no model".
- F1 depends on the threshold, so it is reported at the threshold chosen on the validation windows (0.43).

### Balanced accuracy — the check on the other class
- Mean of recall on the repeat class and recall on the non-repeat class. 0.5 = no skill.
- It shows what F1 hides: how many non-repeat purchases (the retention targets) are detected.

### Why not accuracy alone
With 56 % positives, accuracy rewards predicting the majority class and does not show which class the errors fall on.

## 5. Comparing two models fairly

For each window we compute the difference between two models' scores (paired, because they are measured on the same rows). Then:
- mean difference and its standard error, SEM = std / √5;
- **verdict:** "inside the noise" if |mean| ≤ 2 × SEM, otherwise "better" or "worse" (the rule used in class and in the homework);
- "wins" = in how many of the 5 windows the candidate was ahead.

Pairing removes the window-to-window variation (some years are just harder), which is larger than most differences between good models.

## 6. Questions you may get in the discussion

- *Why not shuffled cross-validation?* Because the target is built from each customer's next purchase; shuffled folds would train on the future of the validation purchases. Show the 99.1 % table in notebook 01, Section 2.4.
- *Why 90 days?* It is the target horizon: the label of a purchase is only fully known 90 days later.
- *Why ROC-AUC if Kaggle uses F1?* AUC compares rankings without mixing in the threshold; the threshold is then tuned for F1 on validation data, so the F1 we report is the one we submit.
- *What is the F1 floor?* 2p / (1 + p): the F1 of predicting the majority class for everyone.
- *Why is the holdout F1 (0.794) higher than the windows' (0.750)?* The holdout year has more repeat purchases (58.5 % vs ~54 %), which raises every F1, including the floor (0.738 vs 0.703). The margin over the floor is similar (+0.056 vs +0.047). AUC, which does not depend on the class share, is consistent (0.814 vs 0.800).
