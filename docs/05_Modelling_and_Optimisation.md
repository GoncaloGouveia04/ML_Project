# Part III.b — Classification benchmark and optimisation

Notebook: `04_Modelling_and_Optimisation.ipynb`.
Outputs: `outputs/final_config.json`, `outputs/final_model.joblib`, `outputs/search_results.json`.

This is the largest part of the grade (7 of 20): Kaggle 1, additional preprocessing 0.5, feature selection 1.5, modelling approach (assessment + ≥ 5 algorithms) 2, performance assessment 1.5, optimisation 1.

## 0. What comes from Part 1

| Kept from Part 1 (`notebooks/04_Model.ipynb`) | Changed | New |
|---|---|---|
| Chronological windows with a 90-day gap; the whole pipeline (preprocessing + selection + model) refitted in every window (week 5); logistic regression with `C` searched over 0.001–100; refit on all labelled rows before predicting `test.csv` | Five yearly windows instead of four date blocks; a sealed holdout; ROC-AUC to compare models; F1 at a tuned threshold instead of 0.5, always next to the F1 floor | Baselines; eleven more algorithms from weeks 6–13; `RandomizedSearchCV` over six families; soft voting; threshold tuning |

## 1. The path in one table

| Step | What | Result |
|---|---|---|
| Baselines | Always "repeat"; rule "bought in last 90 days" | F1 0.703 (AUC 0.5); rule F1 0.69, balanced acc. 0.66 |
| Benchmark | 12 algorithms from class, library defaults, same pipeline and windows | Best: logistic regression 0.790 AUC; worst: single tree 0.614 |
| Optimisation | 60-candidate random search for 6 families | Best: ExtraTrees 0.800 AUC; MLP +0.070, KNN +0.054 from tuning |
| Ensemble | Soft vote of the top 3 tuned | 0.798, not better than ExtraTrees alone |
| Choice | Highest mean AUC (rule fixed in advance) | Tuned ExtraTrees |
| Threshold | F1-optimal on out-of-window probabilities | 0.43 (F1 0.751 vs 0.741 at 0.5) |
| Holdout (once) | Last 12 months of train | AUC 0.814, F1 0.794 (floor 0.738) |
| Final fit | All 6,469 rows | Saved with its preprocessing |

## 2. The twelve algorithms (what each does, and why it scored as it did)

| Family (week) | Algorithm | How it works | Default result |
|---|---|---|---|
| Linear (6) | Logistic regression | Weighted sum of the inputs passed through a sigmoid; outputs a probability; L2 penalty by default (`C=1`) | **0.790** — best default. The signal is largely monotonic ("more purchases → more repeats"), which a linear model on log-scaled inputs captures well. |
| Linear (7) | LDA | Models each class as a Gaussian with a shared covariance → linear boundary | 0.784, close to logistic regression, as in class. |
| Bayesian (7) | Gaussian naive Bayes | Bayes' theorem with independent, Gaussian features per class | 0.743 AUC but F1 at 0.5 collapses (0.51 ± 0.17): correlated history columns are counted several times, so its probabilities are extreme. |
| Distance (8) | KNN | Stores the training rows; predicts by vote of the *k* nearest | 0.734 with k = 5 (overfits); 0.788 after tuning. |
| Tree (9) | Decision tree | Greedy yes/no splits minimising impurity (Gini) | 0.614: unpruned, memorises the training rows (week 9). |
| Bagging (10) | Bagging | 10 unpruned trees on bootstrap samples, averaged | 0.726: averaging helps the tree, but 10 members are few. |
| Bagging (10) | Random forest | Bagging + a random subset of columns at every split | 0.779 |
| Bagging (10) | Extra trees | Forest with random split thresholds too, no bootstrap by default | 0.780 |
| Boosting (11) | AdaBoost | Stumps fitted in sequence on reweighted rows | 0.771 |
| Boosting (11) | Gradient boosting | Trees fitted in sequence to the loss gradient | 0.767 |
| Boosting (11) | Histogram gradient boosting | Gradient boosting on binned columns, leaf-wise trees | 0.765 |
| Neural (12–13) | MLP | One hidden layer of 100 ReLU units, Adam, tiny L2 | 0.725 by default (overfits); 0.795 after tuning. |

The brief asks for at least five class-covered algorithms; twelve are compared, and six are tuned.

## 3. Why ROC-AUC to compare, F1 afterwards

F1 at 0.5 changes when a model's probabilities are shifted, even if its ranking is identical (e.g. `class_weight="balanced"`). AUC compares rankings over all thresholds. Then one threshold is chosen for the final model to maximise F1, the competition metric. Balanced accuracy is reported so the non-repeat class is not forgotten.

## 4. Optimisation

### Random search with 60 candidates (weeks 10–13)
- Grid search tries every combination; here the spaces have 96 to 13,500 combinations per family.
- Random search samples **60 combinations without replacement**. If each candidate has a 5 % chance of being in the top 5 % of the space, the chance that none of 60 is: 0.95⁶⁰ ≈ 0.046. So ~95 % of the time at least one top-5 % candidate is tried, **whatever the size of the space** (Bergstra & Bengio, 2012, JMLR 13).
- Each candidate is scored on the same five windows with the **whole pipeline refitted inside each** (preprocessing, vote selection, model). `cv=folds` passes our time windows to `RandomizedSearchCV`.
- `min_frequency` (rare-level merging) is searched together with the model settings, because preprocessing and model interact.
- A `joblib.Memory` cache stores the fitted preprocessing + selection per window and per `min_frequency`, so it is computed 15 times instead of 300 (same idea as the class's `CachedSearchCV`). Results are identical, only faster.
- `refit=False`: the search only chooses settings; the chosen pipeline is refitted explicitly afterwards.

### Search spaces and why
| Family | Settings searched | Reason |
|---|---|---|
| Logistic regression | `C`, `l1_ratio` (L2 ↔ elastic net ↔ L1), `class_weight` | Penalty strength and type (week 6). |
| KNN | `n_neighbors` 5–151, `weights`, `p` (Manhattan/Euclidean) | Week 8: small *k* overfits; distance weighting; metric. |
| Random forest / Extra trees | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `criterion`, `class_weight` | Week 9 pruning controls + week 10 forest controls. |
| Histogram boosting | `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_depth` | Week 11: rate × stages interaction, tree size, shrinkage. |
| MLP | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`, `batch_size`, `early_stopping` | Weeks 12–13: capacity, non-linearity, L2, optimisation. |

### Effect of tuning (paired by window)
| Family | Default → tuned | Verdict | Winning idea |
|---|---|---|---|
| MLP | 0.725 → 0.795 | better 5/5 | one layer of **8** units, **`alpha = 3`** (strong L2) |
| KNN | 0.734 → 0.788 | better 5/5 | **101** neighbours, distance-weighted, Manhattan |
| Hist. boosting | 0.765 → 0.790 | better 5/5 | 7-leaf trees, learning rate 0.03, **L2 = 10** |
| Extra trees | 0.780 → 0.800 | better 5/5 | leaves of ≥ 5 purchases |
| Random forest | 0.779 → 0.793 | better 5/5 | depth 12, leaves ≥ 5, log2 columns |
| Logistic regression | 0.790 → 0.793 | inside the noise | `C = 0.03` (stronger L2) |

**The common thread is regularisation.** The signal is modest (AUC ceiling ≈ 0.80) and the yearly windows shift, so settings that average more and fit less (more neighbours, smaller networks with stronger penalties, bigger leaves, slower boosting) generalise better. Logistic regression gains nothing because it was already a simple, regularised model — week 6: "what tuning is worth depends on the dataset, and sometimes the answer is nothing".

**Optimism.** Each best score is the maximum of 60 tries on the same windows, so it is biased upwards. The sealed holdout is the correction.

## 5. Ensembles (week 11)

Soft voting averages the probabilities of the three best tuned pipelines (ExtraTrees, MLP, logistic regression). Result: 0.798 vs 0.800 for ExtraTrees alone (−0.001, inside the noise). Averaging helps only when members make **different** errors; here all three rely on the same customer-history signal. Following the rule fixed in advance (and weeks 6/10: prefer the simpler model when a complex one only matches it), the single model is kept.

*Why no stacking or blending?* Both need out-of-fold predictions of the members to train the final learner. To respect time they would need another chronological split inside each training window, leaving the early windows with very little data, for a method whose voting version already showed no gain.

## 6. The final model: Extra Trees

### How Extra Trees work (week 10)
- A forest of decision trees; each tree sees **all rows** (no bootstrap by default).
- At every split, a random subset of columns is considered (here √p) **and the threshold of each candidate column is drawn at random**; the best of these random splits is kept.
- Predictions average the trees' class shares.
- Compared with a random forest, the extra randomness makes the trees **more different from each other**, so averaging removes more variance, at the cost of slightly weaker individual trees. Training is also faster (no search for the best threshold).

### Its tuned settings
`n_estimators = 300`, `max_depth = None`, `min_samples_leaf = 5`, `max_features = "sqrt"`, `criterion = "entropy"`, `class_weight = None`, rare levels merged below 10 rows; vote selection in front.
- `min_samples_leaf = 5` is the main regulariser: each leaf answers with at least five purchases, so probabilities are smoother and less "memorised" (week 9).
- Entropy and Gini usually pick the same splits; the search slightly preferred entropy.

### Why it was chosen
- Highest mean validation AUC (0.800), the rule fixed before the comparison.
- Ahead of each other single model in at least 4 of 5 windows; **lowest spread across windows** (std 0.023).
- Most other candidates are inside the noise of it: the data sets the ceiling, not the algorithm. Logistic regression (0.793) is the simple fallback and would be a defensible answer too — know why we did not choose it (behind in 4/5 windows, rule fixed in advance).

## 7. The decision threshold (week 5)

- The model outputs a probability; the threshold turns it into a label. Lower threshold → more "repeat" labels → higher recall of repeats, more false alarms.
- F1 of the positive (majority) class usually peaks below 0.5. On the pooled out-of-window probabilities the peak is **0.43** (F1 0.751 vs 0.741 at 0.5).
- The best threshold of single windows varies from 0.35 to 0.55, so a finer choice would fit noise.
- Chosen on validation windows only, never on the holdout.

## 8. The sealed holdout

| | F1 | ROC-AUC | Balanced acc. | Recall (repeat) | Non-repeats detected |
|---|---|---|---|---|---|
| Final, threshold 0.43 | **0.794** | **0.814** | 0.706 | 0.876 | 54 % (165/308) |
| Final, threshold 0.50 | 0.779 | 0.814 | 0.731 | 0.786 | 67 % |
| Homework-like LogReg, 0.50 | 0.763 | 0.794 | 0.721 | 0.751 | — |
| Always "repeat" | 0.738 | 0.500 | 0.500 | 1.000 | 0 % |

- AUC agrees with the windows (0.814 vs 0.800 ± 0.023): selecting among hundreds of candidates did not overfit the windows.
- F1 is higher than on the windows (0.794 vs 0.750) because the holdout year has more repeats (58.5 % vs ~54 %); the margin over the floor is similar (+0.056 vs +0.047). Expect the Kaggle F1 to follow the share of repeats in 2025–2026; the **margin** should stay around +0.05.
- The F1 threshold sacrifices non-repeat detection (54 %). For retention, use the ranking (notebook 05).

## 9. Final refit and saving

The same pipeline and settings are refitted on all 6,469 usable rows (development + gap + holdout), so the model learns from the most recent purchases before predicting 2025–2026. Week 2: save the model **with** its preprocessing (`joblib`), so new data are treated identically. The configuration (settings, threshold, scores, kept columns) is also saved as JSON so notebook 06 can rebuild the model from scratch and check that it is identical.

## 10. Alternatives considered

| Alternative | Why not |
|---|---|
| Grid search | Too many combinations (up to 13,500 per family); random search with 60 draws is the class standard. |
| Nested cross-validation | See doc 01: the holdout gives the honest estimate for the period that matters, at a fraction of the cost. |
| Choosing the model on the holdout | Would make the holdout optimistic too; it is used once, after the choice. |
| `class_weight="balanced"` | Searched for logistic regression, random forest and extra trees; no winner used it. Classes are nearly balanced. |
| Early stopping (MLP / boosting) | Searched for the MLP (not chosen); for boosting the stage count was searched directly, as in class. |
| Stacking / blending | See Section 5. |
| Many more trees | 300 was chosen among 100–500; week 10: beyond the plateau more trees only cost time. |

## 11. Questions you may get in the discussion

- *Why did logistic regression win the defaults but Extra Trees the tuned comparison?* Logistic regression's default is already regularised and the signal is mostly monotonic; tree ensembles needed regularisation (leaf size) to stop memorising. After tuning, the forest's ability to use thresholds and interactions gives a small edge.
- *What is the difference between Random Forest and Extra Trees?* Both average trees with random columns per split; Extra Trees also draw the split thresholds at random and do not bootstrap by default → more diverse trees.
- *Bagging vs boosting?* Bagging fits members in parallel on resampled data and averages them (reduces variance of unstable models); boosting fits them in sequence, each correcting the previous (reduces bias of weak models).
- *Why 60 candidates?* 1 − 0.95⁶⁰ ≈ 95 % chance of trying a top-5 % configuration.
- *Why is the tuned MLP so small?* With ~4,000 training rows and modest signal, a larger network overfits; the search preferred 8 units with a strong L2 penalty.
- *Is the 0.43 threshold overfitted?* It is chosen on 3,531 out-of-window predictions from five different years and only moves F1 by +0.01; per-window optima scatter around it (0.35–0.55).
- *What F1 do you expect on Kaggle?* Around the floor of the test period plus ~0.05; if the 2025–26 repeat rate is ~0.60, about 0.79–0.80.
