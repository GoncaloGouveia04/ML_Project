# Machine Learning Project 2026/2027

## Repeat Purchase Prediction

This project uses historical wholesale transactions to answer one practical
question: **will a customer make another purchase within 90 days of the current
purchase?**

The repository contains two connected pieces of work:

1. **Homework:** a compact end-to-end version of the modelling workflow.
2. **Final project:** a deeper study that compares, improves, explains and
   deploys the strongest solution.

The central principle is to build a pipeline that is accurate, reproducible
and useful for commercial decision-making. Every important decision should be
supported by an observation from the data and recorded in a clean notebook.

## Environment Setup

Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
(Miniconda or Anaconda), then open a terminal in the project root and create
the environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate machine_learning_project
```

The environment uses Python 3.13.5 and installs the project's pinned Python
dependencies, including Jupyter and scikit-learn. To open the main notebook:

```bash
jupyter notebook
```

Then open `notebooks/main.ipynb`. In VS Code, select the
`machine_learning_project` environment as the notebook kernel.

## Data at a Glance

The main files are in `data/`:

- `train.csv`: 6,534 purchase occasions with the binary target
  `repeat_purchase_90d`.
- `test.csv`: 986 later purchase occasions without the target.
- `clientes.csv`: optional customer-level information that can be joined using
  `customer_id`.
- `sample_submission.csv`: the required Kaggle format with `ID` and
  `repeat_purchase_90d`.

Each row represents one customer purchase occasion. The predictors include:

- current transaction amounts, tax, shipping, discounts and cost;
- purchase detail and invoice counts;
- categorical information such as payment type, salesperson, zone, transport,
  warehouse and document series;
- recency, customer tenure and previous-purchase statistics;
- rolling purchase counts and spend over 30, 90, 180 and 365 days;
- the anonymized purchase date and a `random_noise` field that must be tested,
  not blindly trusted.

The target is `1` when the same customer purchases again within the following
90 days, and `0` otherwise. `ID` is only a row/submission identifier and must
remain unchanged in the final predictions.

## Homework: Compact Pipeline

**Deadline: 26 October 2026, 17:59**

The homework is a small, complete rehearsal for the final project. The goal is
to understand the data, prepare it correctly, select useful variables, train a
simple binary classifier and produce predictions for `test.csv`.

### What to do

1. **Explore the data**
   - Load `train.csv` and inspect `clientes.csv` if the customer join is useful.
   - Describe the rows, columns, target balance and important distributions.
   - Check missing values, duplicates, invalid values and suspicious
     inconsistencies.
   - Use focused univariate and multivariate plots to find relationships that
     can guide modelling decisions.

2. **Clean and preprocess**
   - Define how numerical missing values, categorical missing values and
     anomalous observations are handled.
   - Decide whether `clientes.csv` adds useful information through
     `customer_id`.
   - Create additional features only when they have a clear purpose.
   - Encode categorical variables and scale numerical variables when needed.
   - Fit preprocessing only on the training data. Never use `test.csv` to learn
     imputation, scaling or encoding parameters.

3. **Select features**
   - Use a clear class-based feature-selection method.
   - Treat identifiers and identifier-like fields critically.
   - Explain the final feature set and what was removed or retained.

4. **Build and assess a simple model**
   - Choose an assessment strategy and suitable classification metrics.
   - Train at least one model using `train.csv`.
   - Generate one prediction for every row in `test.csv`.
   - Export predictions using the same columns as `sample_submission.csv`.

### Homework deliverables

- A clean Jupyter notebook, or a zip file containing the notebooks, with code
  and Markdown comments.
- A **two-page PDF** describing the pipeline structure and rationale.
- Files named using the convention `Homework_GroupXX`.

The homework is worth up to 20 points, including a possible extra point for
finishing in the top five Kaggle groups.

## Final Project: Full Study

**Deadline: 22 December 2026, 17:59**

The final project starts from the homework pipeline and turns it into a
defensible modelling study. The emphasis is not only on obtaining a good
Kaggle score, but also on showing why the selected approach is reliable,
interpretable and reproducible.

### Recommended project structure

#### Group contribution and abstract

Begin by identifying each member's contribution and estimated percentage. Add
a 200-300 word abstract covering the business context, objectives, approach,
main results and conclusions.

#### I. Identify the business need

Explain why predicting a 90-day repeat purchase matters. Define the modelling
question, the intended commercial use and the assessment strategy, such as a
holdout split or cross-validation. Because purchase dates have an order, the
validation design should be chosen carefully and justified.

#### II. Explore and preprocess the data

Describe the received data and the insights that influence the pipeline. Clean
the data, handle missing and implausible values, encode categoricals, engineer
useful features and assess whether the optional customer table improves the
information available to the model. Keep the process reproducible and avoid
target leakage.

#### III. Classification and optimization

This is the largest part of the project:

- define metrics that match the business problem and explain them;
- implement and compare at least **five class-covered algorithms**;
- apply a justified feature-selection strategy;
- benchmark candidate models under the same evaluation protocol;
- optimize the strongest approaches and discuss the effect of tuning;
- compare results clearly, including uncertainty or variation where possible;
- train the selected final solution and submit it to Kaggle.

#### IV. Open-ended investigation

Add a question that goes beyond simply adding another standard pipeline step.
Good options include comparing alternative feature sets, analysing errors,
studying probability calibration, investigating prediction confidence,
performing an ablation study, or building a small interface for scoring new
observations. State the objective first, then present the method, results and
key takeaways.

#### V. Deployment

The final notebook must apply the same completed pipeline to `test.csv` and
write the selected solution as a submission-ready CSV. The deployment section
should demonstrate that a new observation receives exactly the same preparation
steps as the training data and that the output follows the sample submission
schema.

## Suggested Work Sequence

1. Establish a reproducible baseline and inspect the target balance.
2. Build one leakage-safe preprocessing pipeline.
3. Complete the homework model and submission first.
4. Reuse the validated pipeline for five-model benchmarking.
5. Compare feature sets and optimize only the strongest candidates.
6. Choose one open-ended research question and evaluate it rigorously.
7. Retrain the final model, generate the Kaggle CSV and run the notebook from
   start to finish before submitting.

## Quality and Submission Rules

- Every notebook cell should have a clear purpose.
- Include only analyses that affect a decision or communicate an important
  result.
- Do not use forbidden AutoML tools such as Lazy Predict, Feature Tools or
  TSFresh.
- Keep all modelling decisions and relevant experiments visible in the
  submitted notebooks.
- Use permitted tools such as vanilla scikit-learn, TensorFlow, Keras,
  PyTorch/skorch or Transformers.
- Cite reliable peer-reviewed sources when external evidence is needed; do not
  cite LLMs.
- Run the final notebook from beginning to end without interruption.
- Submit to Kaggle, since the competition contributes to the evaluation.
- The project may receive an extra point for being publicly available on
  GitHub.

## Definition of Done

The project is complete when a reader can follow the reasoning from business
question to final CSV, reproduce every transformation, understand why the final
model was selected and verify that the predictions were generated without
using unavailable target information.
