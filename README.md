# Machine Learning Project 2026/2027 — 90-Day Repeat Purchase Prediction

**Group 33** · Gonçalo Gouveia · Gonçalo Faustino · Rita Pinto · João Santos

Everything & Then Some, a wholesale distributor, wants to know at the moment of each purchase
whether the same customer will **buy again within the next 90 days**. This repository contains the
whole study, from exploration to the Kaggle submission, as six notebooks that share one Python module.

## Repository structure

| Path | Content |
|---|---|
| `notebooks/01_Business_and_Data_Exploration.ipynb` | Group contribution, abstract, I. business need and assessment strategy, II.a data exploration |
| `notebooks/02_Preprocessing.ipynb` | II.b cleaning and preparation, with every optional choice tested |
| `notebooks/03_Feature_Selection.ipynb` | III.a feature-selection strategy |
| `notebooks/04_Modelling_and_Optimisation.ipynb` | III.b benchmark of 12 algorithms, two-stage tuning, final model, sealed holdout |
| `notebooks/05_Open_Ended.ipynb` | IV. can the model be trusted to drive commercial actions? |
| `notebooks/06_Deployment.ipynb` | V. the final pipeline applied to `test.csv` and the Kaggle file |
| `helpers/project_pipeline.py` | Cleaning, features, the scikit-learn pipeline, the validation design and the scoring used by every notebook |
| `data/` | The supplied files: `train.csv`, `test.csv`, `clientes.csv`, `sample_submission.csv` |
| `results/stage1_search.json` | Saved results of the stage-1 random search of notebook 04 |
| `outputs/` | Created when the notebooks run: the fitted final model, its configuration and the submission CSV (not versioned) |

## Environment

Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or
Anaconda), then open a terminal in the project root:

```bash
conda env create -f environment.yml
conda activate machine_learning_project
jupyter notebook
```

The environment uses Python 3.13.5 with pinned versions of scikit-learn, pandas, NumPy and the
other dependencies. In VS Code, select the `machine_learning_project` environment as the notebook kernel.

## Running the project

Open the notebooks in the `notebooks/` folder and run them in order, **01 to 06**:

- notebook 04 saves the final model and its configuration in `outputs/`, which notebooks 05 and 06 read;
- notebook 06 writes the Kaggle file `outputs/ML_Group_33_Version01.csv`.

Notebook 04 runs the stage-1 random search (about 5 minutes on 12 cores) and saves its results in
`results/stage1_search.json`; set `RUN_SEARCH = False` in notebook 04 to load them instead.

## Main result

The final model is an **Extra Trees** classifier (575 randomised trees, at least 6 purchases per
leaf, 25 % of the columns considered at each split, balanced class weights) behind a vote of three
feature selectors, with a decision threshold of 0.39 chosen for F1 on the validation windows.

| Evaluation | ROC-AUC | F1 | F1 of always predicting "repeat" |
|---|---|---|---|
| Five yearly validation windows (mean ± sd) | 0.801 ± 0.024 | 0.751 | 0.703 |
| Holdout: last 12 months of `train.csv` | 0.816 | 0.802 | 0.738 |
| Kaggle public leaderboard | — | 0.808 | — |

Notebook 04 explains how the model was chosen, including a refinement stage during which the
holdout and public Kaggle scores were also consulted.
