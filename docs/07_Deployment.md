# Part V — Deployment

Notebook: `06_Deployment.ipynb`.
Output: `outputs/ML_Group_XX_Version01.csv`.

Grading (1 point): "take your test data and ensure that it follows the steps implemented in the previous stages".

## 0. What comes from Part 1

| Kept from Part 1 (`notebooks/04_Model.ipynb`, Section 7) | Changed | New |
|---|---|---|
| Refit the final pipeline on all labelled rows, predict the 986 test rows, keep the `ID` order | The fitted pipeline is saved by notebook 04 and loaded here (week 2) instead of being recomputed in the modelling notebook | Reproducibility check, checks on `test.csv` (unseen levels, invalid values), the single-purchase walk-through, the Kaggle file name |

## 1. What this part does

1. Loads the configuration and the fitted pipeline saved by notebook 04.
2. **Rebuilds** the pipeline from the configuration, refits it on the same rows and checks that it gives exactly the same test probabilities as the saved file (reproducibility).
3. Checks the new data: same columns as training, no target, unique IDs, same IDs as the sample submission; lists category levels never seen in training and invalid values.
4. Predicts probabilities, applies the threshold (0.43), builds the submission in the **same row order and columns** as `sample_submission.csv`, asserts its format and writes the CSV.
5. Follows **one purchase** through every stage of the pipeline and checks that it receives the same probability as in the batch prediction.

## 2. Why the pipeline guarantees identical treatment

The saved object is one scikit-learn `Pipeline` with three steps:

| Step | Contains | Fitted on |
|---|---|---|
| `prep` | `FeatureBuilder` (cleaning, rebuilt values, features, `log1p`, dropped columns) + `ColumnTransformer` (median, scaling, `Missing`, one-hot) | all 6,469 training rows |
| `select` | the vote selector (filter + RFE + L1) | all 6,469 training rows |
| `model` | tuned Extra Trees | all 6,469 training rows |

Calling `predict_proba(test)` runs every step's `transform` with the values learned on training rows. Nothing is refitted on test. The stateless part is the same Python function used in every notebook, so there is no second copy of the cleaning code that could differ. Week 2: "save the model together with its preprocessing, so next month's data gets treated exactly the same way".

## 3. Details that matter

- **Unseen categories.** Test contains 8 levels never seen in training (3 payment types, 2 commercial zones, 2 transport methods, 1 warehouse). `OneHotEncoder(handle_unknown="infrequent_if_exist")` maps them to the "infrequent" column of their variable (or to all zeros if that variable has none) instead of failing.
- **Invalid values in test.** 12 values (placeholders and negatives) become missing through the same rule as in training, then are rebuilt or imputed with the training medians.
- **Threshold.** 0.43, chosen on the validation windows in notebook 04 and stored in `final_config.json`.
- **Row order.** The submission is built by merging predictions onto the `ID` column of `sample_submission.csv`, and an `assert` checks that the IDs are identical and in the same order.
- **Reproducibility.** Every random component has `random_state = 42`; the rebuilt pipeline reproduces the saved predictions exactly.

## 4. What the predictions look like

- 77 % of test purchases are labelled "repeat" (mean probability 0.62).
- In the last training year (in-sample, since the final model was trained on it), the same threshold labels 70 % as "repeat" (actual rate 58.5 %).
- The higher share in test is consistent with the rising repeat rate of recent years (0.60 in 2024, 0.63 in early 2025) and with an F1-oriented threshold, which labels borderline purchases as "repeat".

## 5. Before uploading to Kaggle

1. Set `GROUP = "XX"` in the first cell of `06_Deployment.ipynb` to your two-digit group number, and `VERSION` to a new number for each upload (`ML_Group_07_Version03.csv` style, as the competition rules require).
2. Re-run notebook 06 (about 20 seconds).
3. Upload `outputs/ML_Group_XX_VersionNN.csv` and **select it as the final submission**: the rules say the selected Kaggle submission must match the model documented in the notebooks.
4. **Check the competition metric on the Kaggle page.** This project assumes F1 (the metric used in the homework). If the competition uses another label metric (for example accuracy), only the threshold choice changes: in notebook 04, Section 4.7, compute that metric instead of F1 for each candidate threshold, then re-run notebooks 04 to 06. Everything else (features, selection, model) was chosen with ROC-AUC and stays valid.

## 6. Alternatives considered

| Alternative | Why not |
|---|---|
| Re-implementing the cleaning in the deployment notebook | Two copies of the same logic can drift apart (a homework problem). The module is imported instead. |
| Fitting the final model on development rows only | Would ignore the most recent 13 months of purchases, the closest to the test period. The holdout already gave the honest estimate; refitting on all rows is standard practice. |
| Exploiting the later purchases visible inside `test.csv` | Target leakage: at the prediction moment, the next purchase is unknown. |
| Saving only the model (not the pipeline) | New data would need the preprocessing to be rebuilt by hand. |

## 7. Questions you may get in the discussion

- *How do you guarantee test data gets the same preprocessing?* The preprocessing is a step of the saved pipeline, fitted on training rows; test is only transformed. The single-purchase walk-through shows the same probability as the batch.
- *What happens if a new payment type appears?* It goes to the "infrequent" level (or zeros); the model still predicts.
- *Why refit on all rows after evaluating on the holdout?* Evaluation estimates the *procedure*; the deployed model should use all available information, especially the most recent year.
- *How large is the saved model?* About 30 MB (300 fully grown trees). It can be rebuilt from `final_config.json` in seconds, so it does not need to be committed to GitHub.
