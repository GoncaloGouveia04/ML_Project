"""Shared code for the final-project notebooks.

Every notebook imports its cleaning, features, validation design and scoring
from this one file, so a change made here reaches all of them. (In the homework
the same functions were copied into four notebooks and could drift apart.)

The module holds four kinds of code:

1. Stateless, row-by-row steps (`clean_purchases`, `add_features`). They learn
   nothing from the data, so they treat train, test and a brand-new purchase in
   exactly the same way and cannot leak information.
2. The preprocessing that *learns* (medians, category lists, scaling) and the
   feature selectors. They live inside a scikit-learn `Pipeline`, so they are
   refitted on the training rows of every split.
3. The chronological validation design (`development_split`).
4. Scoring helpers shared by every notebook.

Where each piece comes from
---------------------------
Each function says in its docstring where it comes from:
- "Part 1": taken from our homework notebooks (not included in this
  repository), kept or changed (with the reason);
- "Week N": the class notebook in `ML_DSAA_Practicals` that teaches it;
- "New": added in Part 2, with the finding that motivated it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel, SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# scikit-learn's Parallel passes the notebook's warning filters on to the worker processes.
from sklearn.utils.parallel import Parallel, delayed

from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                RandomForestClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42         # one seed everywhere, so every result can be reproduced
TARGET = "repeat_purchase_90d"
ID_COLUMN = "ID"
CUSTOMER = "customer_id"
DATE = "purchase_date"
HORIZON_DAYS = 90          # the target looks 90 days ahead
SENTINEL = 1_000_000_000   # placeholder value found in two columns

# Part 1: the same six categorical columns (they are labels with no order, so one-hot, week 4).
CATEGORICAL = ["payment_type", "salesperson", "commercial_zone",
               "transport_method", "warehouse", "document_series"]
# New: calendar month taken from purchase_date (four-month cycle found in notebook 01).
MONTH = "purchase_month"

# Raw numeric columns grouped by what they describe.
RAW_GROUPS = {
    "current purchase": ["Total a Pagar", "Total Bruto", "Total Liquido", "Total Imp_",
                         "Total Portes", "Total Desc_ Global", "Total Desc_ Linha", "Custo",
                         "N_ Detalhes", "invoice_count_current_purchase"],
    "history timing": ["days_since_previous_purchase", "customer_tenure_days",
                       "prior_purchase_count", "prior_mean_gap_days", "prior_std_gap_days"],
    "history value": ["prior_total_spend", "prior_mean_purchase_value",
                      "prior_std_purchase_value", "prior_min_purchase_value",
                      "prior_max_purchase_value"],
    "rolling counts": ["purchase_count_30d", "purchase_count_90d",
                       "purchase_count_180d", "purchase_count_365d"],
    "rolling spend": ["spend_30d", "spend_90d", "spend_180d", "spend_365d"],
}
RAW_NUMERIC = [c for group in RAW_GROUPS.values() for c in group]

# New: exact copies of other columns (verified in notebook 01), dropped before modelling,
# because a column that is a sum of others adds no information (week 4: redundancy):
#   Total a Pagar = Total Liquido + Total Imp_
#   Total Bruto   = Total Liquido + Total Desc_ Linha + Total Desc_ Global
IDENTITY_COLUMNS = ["Total a Pagar", "Total Bruto"]

# Columns that are never model inputs (Part 1 excluded the same ones): identifiers carry no
# behaviour, the date is used to split and to build the month, random_noise is noise.
NOT_FEATURES = [ID_COLUMN, CUSTOMER, DATE, "random_noise", TARGET]

# Engineered features and the information group each one describes.
# Part 1 already had `first_purchase` and value per line (`net_value_per_detail`); the
# other ratios are new and each expresses a finding of notebook 01 (week 4: feature
# engineering comes before feature selection, which then decides what survives).
ENGINEERED = {
    "first_purchase": "history timing",
    "recency_ratio": "history timing",
    "purchases_per_year": "history timing",
    "ticket_vs_history": "history value",
    "recent_activity_share": "rolling counts",
    "discount_rate": "current purchase",
    "tax_rate": "current purchase",
    "value_per_detail": "current purchase",
}

# Right-skewed, non-negative columns that receive log1p (Part 1 and weeks 3-4: log1p
# compresses long right tails and keeps zeros, unlike log).
LOG_COLUMNS = [c for c in RAW_NUMERIC if c not in IDENTITY_COLUMNS] + [
    "recency_ratio", "purchases_per_year", "ticket_vs_history", "value_per_detail"]

# Prepared input columns by information group (used by the ablation study).
FEATURE_GROUPS = {
    group: [c for c in columns if c not in IDENTITY_COLUMNS]
    + [f for f, g in ENGINEERED.items() if g == group]
    for group, columns in RAW_GROUPS.items()
}
FEATURE_GROUPS["sale context (categoricals)"] = list(CATEGORICAL)
FEATURE_GROUPS["purchase month"] = [MONTH]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_data_folder(start: Path | None = None) -> Path:
    """Return the project's `data/` folder, searching upwards from `start`."""
    here = Path.cwd() if start is None else Path(start)
    for folder in [here, *here.parents]:
        if (folder / "data" / "train.csv").is_file():
            return folder / "data"
    raise FileNotFoundError(f"Could not find data/train.csv above {here}")


def load_raw(data_folder: Path):
    """Read the three supplied tables with `purchase_date` parsed as a date."""
    train = pd.read_csv(data_folder / "train.csv", parse_dates=[DATE])
    test = pd.read_csv(data_folder / "test.csv", parse_dates=[DATE])
    clientes = pd.read_csv(data_folder / "clientes.csv")
    return train, test, clientes


def empty_rows(frame: pd.DataFrame) -> pd.Series:
    """Rows where every purchase field is missing (only ID, customer and target exist).

    Part 1 found these 65 rows (exploration, Section 4.1) and said to remove them, but
    its preprocessing kept them. Changed: they are dropped in every notebook.
    """
    return frame[RAW_NUMERIC + CATEGORICAL + [DATE]].isna().all(axis=1)


# ---------------------------------------------------------------------------
# 1. Stateless cleaning and features
# ---------------------------------------------------------------------------

def clean_purchases(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn impossible values into NaN, then rebuild values that follow from a definition.

    1. Invalid -> missing: the 1e9 placeholder and negative amounts or day counts.
    2. Repair by definition, only where the value is missing:
       - prior_mean_purchase_value = prior_total_spend / prior_purchase_count
       - days_since_previous_purchase = customer_tenure_days when exactly one
         earlier purchase exists (the first purchase *is* the previous one)
       - prior_mean_gap_days = (tenure - days_since_previous) / (prior_count - 1)
       - spend_365d = 0 when purchase_count_365d = 0, and = prior_total_spend
         when every earlier purchase happened in the last 365 days.

    Part 1 (`clean_values`, adapted from week 3): step 1 is the same rule.
    New: step 2. Part 1 filled every missing value with the median; when a value
    can be computed exactly from other columns of the same row, that is better
    than a guess, and it uses no other row, so it cannot leak.
    """
    data = frame.copy()
    numeric = [c for c in RAW_NUMERIC if c in data]
    values = data[numeric]
    data[numeric] = values.mask((values < 0) | (values >= SENTINEL))

    count = data["prior_purchase_count"]
    tenure = data["customer_tenure_days"]

    mean_value = data["prior_total_spend"] / count.where(count > 0)
    data["prior_mean_purchase_value"] = data["prior_mean_purchase_value"].fillna(mean_value)

    one_before = data["days_since_previous_purchase"].isna() & (count == 1)
    data.loc[one_before, "days_since_previous_purchase"] = tenure[one_before]

    gap = (tenure - data["days_since_previous_purchase"]) / (count - 1).where(count >= 2)
    data["prior_mean_gap_days"] = data["prior_mean_gap_days"].fillna(gap.where(gap >= 0))

    no_recent = data["spend_365d"].isna() & (data["purchase_count_365d"] == 0)
    data.loc[no_recent, "spend_365d"] = 0.0
    all_recent = data["spend_365d"].isna() & (data["purchase_count_365d"] == count)
    data.loc[all_recent, "spend_365d"] = data.loc[all_recent, "prior_total_spend"]
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features: ratios of columns in the same row, plus the month.

    Every feature uses only the current row, which describes the past of the
    customer, so it is available at prediction time (no leakage).
    """
    out = data.copy()
    count = out["prior_purchase_count"]

    def ratio(numerator, denominator):
        # A zero or missing denominator gives NaN (imputed later), never a division error.
        return numerator / denominator.where(denominator > 0)

    # Part 1: the same flag. A first purchase has no history, so the median imputed into
    # its history columns must not be mistaken for a real value (structural missingness).
    out["first_purchase"] = (count == 0).astype(float).where(count.notna())
    # Above 1: the customer is already later than their usual rhythm.
    out["recency_ratio"] = ratio(out["days_since_previous_purchase"], out["prior_mean_gap_days"])
    out["purchases_per_year"] = ratio(count, out["customer_tenure_days"] / 365.25)
    # Is this purchase bigger or smaller than the customer's usual ticket?
    out["ticket_vs_history"] = ratio(out["Total Liquido"], out["prior_mean_purchase_value"])
    # Share of last year's purchases made in the last 90 days (activity trend).
    out["recent_activity_share"] = ratio(out["purchase_count_90d"], out["purchase_count_365d"])
    out["discount_rate"] = ratio(out["Total Desc_ Linha"] + out["Total Desc_ Global"],
                                 out["Total Bruto"])
    out["tax_rate"] = ratio(out["Total Imp_"], out["Total Liquido"])
    # Part 1: `net_value_per_detail`, the same ratio under a shorter name.
    out["value_per_detail"] = ratio(out["Total Liquido"], out["N_ Detalhes"])
    # Month as a label, no order assumed between months ("m03" = March): the cycle peaks in
    # months 3, 7 and 11, which a single numeric month column could not express.
    month = pd.to_datetime(out[DATE]).dt.month
    out[MONTH] = ("m" + month.astype("Int64").astype("string").str.zfill(2)).astype(object)
    out.loc[month.isna(), MONTH] = np.nan
    return out


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean, add features, log-transform skewed columns and drop non-features.

    Part 1 order kept: log1p comes before median imputation (Part 1, preprocessing
    Section 5), so the median is taken on the compressed scale.
    """
    data = add_features(clean_purchases(frame))
    present = [c for c in LOG_COLUMNS if c in data]
    data[present] = np.log1p(data[present])
    return data.drop(columns=[c for c in NOT_FEATURES + IDENTITY_COLUMNS if c in data])


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Pipeline step running `prepare_frame`.

    `drop` removes named prepared columns (ablation study); `probe=True` keeps
    `random_noise` as an input (feature-selection sanity check). The step learns
    only the list of output columns, never a statistic.

    Why a class: a scikit-learn step (fit/transform) can sit inside a Pipeline
    (week 5), so the same cleaning runs inside every fold and on test.csv. Part 1
    called its cleaning functions by hand before each split.
    """

    def __init__(self, drop=(), probe=False):
        self.drop = drop
        self.probe = probe

    def _prepare(self, X):
        prepared = prepare_frame(X)
        if self.probe:
            prepared["random_noise"] = X["random_noise"].to_numpy()
        return prepared

    def fit(self, X, y=None):
        self.columns_ = [c for c in self._prepare(X).columns if c not in set(self.drop)]
        return self

    def transform(self, X):
        return self._prepare(X)[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def feature_columns(drop=(), probe=False, extra_numeric=(), extra_categorical=()):
    """Numeric and categorical input columns after FeatureBuilder."""
    numeric = [c for c in RAW_NUMERIC if c not in IDENTITY_COLUMNS] + list(ENGINEERED)
    numeric += (["random_noise"] if probe else []) + list(extra_numeric)
    categorical = CATEGORICAL + [MONTH] + list(extra_categorical)
    drop = set(drop)
    return [c for c in numeric if c not in drop], [c for c in categorical if c not in drop]


# ---------------------------------------------------------------------------
# 2. Learned preprocessing and feature selection (refitted in every split)
# ---------------------------------------------------------------------------

def build_preprocessor(drop=(), min_frequency=1, probe=False,
                       extra_numeric=(), extra_categorical=()):
    """FeatureBuilder -> numeric: median fill + scaling | categorical: 'Missing' + one-hot.

    `min_frequency`: categories seen in fewer training rows are merged into one
    'infrequent' column (1 = no merging). It is tuned in notebook 04.

    Part 1 (`fill_missing` + `fold_matrices`, from week 4): the same steps (median,
    a constant missing label, one-hot, StandardScaler), each fitted on training
    rows only. Changed: written as a Pipeline + ColumnTransformer (week 5) instead
    of a hand-written function, so the fitted object can be refitted per fold,
    tuned by RandomizedSearchCV and saved for test.csv.

    Encoder choices:
    - handle_unknown="infrequent_if_exist": a level never seen in training goes to the
      'infrequent' column when there is one, otherwise to all zeros. With the default
      min_frequency=1 there is none, so this is exactly Part 1's / week 4's "ignore".
    - min_frequency is an option of the same OneHotEncoder, not used in the class
      notebooks. It is the one extra setting in the project: it is kept only because
      it is tested (notebook 02, 2.14) and chosen by the search on the validation
      windows (notebook 04).
    """
    numeric, categorical = feature_columns(drop, probe, extra_numeric, extra_categorical)
    # Scaling matters for KNN, MLP and penalised logistic regression (weeks 6, 8, 12) and
    # does not change tree models, so one recipe serves every algorithm.
    numeric_steps = Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())])
    categorical_steps = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frequency, sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)],
        verbose_feature_names_out=False)
    return Pipeline([("features", FeatureBuilder(drop=drop, probe=probe)),
                     ("columns", columns)])


def target_correlation(X, y):
    """|Pearson r| between each column and the 0/1 target.

    For a numeric column this is the point-biserial correlation; for a 0/1
    one-hot column it is the phi coefficient, which equals Cramer's V of the
    2x2 table. So one formula gives the relevance index the class uses for each
    column type. A constant column scores 0.

    Week 4 (`encoded_effect_scores`) computes the same magnitudes; Part 1 used them
    in its relevance table (feature selection, Section 3.5.3).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    spread = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc.T @ yc) / spread
    return np.abs(np.nan_to_num(r))


class CorrelationFilter(SelectorMixin, BaseEstimator):
    """Week 4 filter: walk down the relevance ranking, skip a column whose |r| with
    an already kept column exceeds `redundancy`, and keep at most `k` columns.

    Week 4 / Part 1: the same algorithm as the class `CorrelationFilter`, which Part 1
    used in Section 3.6. Changed only in form: it inherits SelectorMixin so it can be
    a step of a scikit-learn Pipeline (refitted in every fold, saved with the model).
    """

    def __init__(self, k=60, redundancy=0.8):
        self.k = k
        self.redundancy = redundancy

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.relevance_ = target_correlation(X, y)
        with np.errstate(invalid="ignore", divide="ignore"):
            between = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        keep = []
        for candidate in np.argsort(self.relevance_)[::-1]:
            if self.k is not None and len(keep) == self.k:
                break
            if any(between[candidate, chosen] > self.redundancy for chosen in keep):
                continue
            keep.append(candidate)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def _get_support_mask(self):
        return self.support_


class VoteSelector(SelectorMixin, BaseEstimator):
    """Week 4 vote: keep a column when at least `minimum` member selectors keep it.

    Week 4 / Part 1: the class `Vote` (Part 1, Section 3.9), as a Pipeline step.
    Members are cloned before fitting so a fitted vote never reuses another fold's fit.
    """

    def __init__(self, selectors, minimum=2):
        self.selectors = selectors
        self.minimum = minimum

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.fitted_ = [(name, clone(s).fit(X, y)) for name, s in self.selectors]
        self.votes_ = sum(np.asarray(s.get_support(), dtype=int) for _, s in self.fitted_)
        self.support_ = self.votes_ >= self.minimum
        return self

    def _get_support_mask(self):
        return self.support_


# Settings of the three voters (chosen in notebook 03). Part 1 used narrow budgets
# (10-30 columns); notebook 03 shows those lose accuracy for tree ensembles, so the
# budgets are wider. 0.8 is the redundancy cut-off used in notebook 01 and week 4.
VOTE_BUDGET = 60
VOTE_REDUNDANCY = 0.8
VOTE_L1_C = 0.3


def filter_selector(k=VOTE_BUDGET, redundancy=VOTE_REDUNDANCY):
    """Filter member of the vote (week 4 / Part 1, Section 3.6)."""
    return CorrelationFilter(k=k, redundancy=redundancy)


def wrapper_selector(k=VOTE_BUDGET):
    """RFE with a logistic regression (liblinear solver: same model, faster fit).

    Week 4 / Part 1 (Section 3.7): RFE around a logistic regression. step=5 removes
    five columns per round instead of one, to keep the many refits affordable.
    RFE has no internal cross-validation, so it never mixes time inside a fold
    (RFECV would use shuffled folds).
    """
    return RFE(LogisticRegression(solver="liblinear", max_iter=5000, random_state=RANDOM_STATE),
               n_features_to_select=k, step=5)


def embedded_selector(C=VOTE_L1_C):
    """L1 logistic regression: keeps the columns whose coefficient is not exactly 0.

    Week 4 / Part 1 (Section 3.8): SelectFromModel around an L1 logistic regression
    with the liblinear solver, as in class. threshold=1e-8 means "non-zero".
    """
    return SelectFromModel(
        LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=5000,
                           random_state=RANDOM_STATE),
        threshold=1e-8)


def make_vote_selector(k=VOTE_BUDGET, C=VOTE_L1_C, minimum=2):
    """The exported strategy: filter + wrapper + embedded, kept by >= `minimum` votes."""
    return VoteSelector([("filter", filter_selector(k)), ("wrapper", wrapper_selector(k)),
                         ("embedded", embedded_selector(C))], minimum=minimum)


def make_pipeline(model, *, selector=None, memory=None, **prep_options):
    """Full modelling pipeline: preprocessing -> optional selector -> model.

    Part 1 (model notebook, Section 4) / week 5: preprocessing, selection and model
    in one Pipeline, so fitting it on a fold's training rows fits all three there.
    `memory` (week 11) caches the fitted preprocessing + selection between candidates.
    """
    steps = [("prep", build_preprocessor(**prep_options))]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", model))
    return Pipeline(steps, memory=memory)


def base_model(name):
    """Unfitted model with the fixed settings shared by the notebook-04 searches.

    Fixed settings and why:
    - LogisticRegression (week 6; Part 1's model): saga is the solver that supports
      every penalty the search tries (l1_ratio 0 to 1); max_iter so the fit converges.
    - KNN (week 8), RandomForest / ExtraTrees (week 10), HistGradientBoosting
      (week 11), MLP (weeks 12-13): library defaults; the search tunes the rest.
      ExtraTrees is the random forest variant with random split thresholds presented
      in the week 10 lecture (study guide in ML_DSAA_Practicals/notebooksummaries).
    - early_stopping=False: boosting runs exactly max_iter rounds, a searched setting.
    - n_jobs=1 inside the models: parallelism is used across folds/candidates instead.
    """
    models = {
        "LogisticRegression": LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(early_stopping=False,
                                                               random_state=RANDOM_STATE),
        "MLP": MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
    }
    return models[name]


def final_pipeline(config, memory=None, **prep_options):
    """Rebuild the chosen pipeline (unfitted) from the configuration saved by notebook 04."""
    params = {key: tuple(value) if isinstance(value, list) else value
              for key, value in config["best_params"].items()}   # JSON turns tuples into lists
    pipeline = make_pipeline(base_model(config["model"]), selector=make_vote_selector(),
                             memory=memory, **prep_options)
    return pipeline.set_params(**params)


def prepared_names(pipeline):
    """Names of the columns leaving the preprocessing step of a fitted pipeline."""
    return np.asarray(pipeline.named_steps["prep"].named_steps["columns"].get_feature_names_out())


def selected_names(pipeline):
    """Names of the columns the fitted selector kept (all of them if there is no selector)."""
    names = prepared_names(pipeline)
    if "select" not in pipeline.named_steps:
        return names
    return names[pipeline.named_steps["select"].get_support()]


# ---------------------------------------------------------------------------
# 3. Chronological validation design
# ---------------------------------------------------------------------------

def time_windows(dates, starts, length_days=365, gap_days=HORIZON_DAYS):
    """Return (train_positions, validation_positions) pairs.

    Validation = purchases in [start, start + length). Training = purchases more
    than `gap_days` before `start`, so every training label (which looks 90 days
    ahead) is fully known before the validation window opens.

    Part 1 (feature selection 3.4.1, model Section 2): the same idea, train on the
    past, validate on what follows, with a 90-day gap. Week 5 (time-aware
    splitting): a shuffled K-fold would train on the future. TimeSeriesSplit is not
    used because its folds have equal row counts, not calendar years, and we want
    every window to cover one full year (the buying cycle repeats every 4 months).
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    folds = []
    for start in pd.to_datetime(list(starts)):
        end = start + pd.Timedelta(days=length_days)
        fit_at = np.flatnonzero(dates < start - pd.Timedelta(days=gap_days))
        score_at = np.flatnonzero((dates >= start) & (dates < end))
        folds.append((fit_at, score_at))
    return folds


@dataclass
class Design:
    """Everything a notebook needs about the validation design."""
    X_dev: pd.DataFrame      # development rows (selection and tuning)
    y_dev: pd.Series
    X_hold: pd.DataFrame     # sealed holdout: the latest 12 months of train.csv
    y_hold: pd.Series
    X_all: pd.DataFrame      # every usable labelled row (final fit for Kaggle)
    y_all: pd.Series
    folds: list              # 5 yearly expanding windows inside the development rows
    holdout_start: pd.Timestamp
    development_end: pd.Timestamp
    fold_starts: list


def development_split(train: pd.DataFrame, n_folds=5, window_days=365) -> Design:
    """Build the one validation design used by every notebook.

    - Empty rows are removed and rows are sorted by date.
    - Holdout = the last 365 days of train.csv, scored once at the very end.
    - Development = rows more than 90 days before the holdout starts, so no
      development label looks into the holdout period.
    - Folds = `n_folds` consecutive yearly validation windows that end where
      the development rows end; each trains on rows > 90 days before its window.

    Changed from Part 1: Part 1 used the same windows to choose its settings and
    to report its score, so the score was optimistic. Week 5: reserve the test rows
    first and keep the design fixed; the holdout plays that role here.
    """
    usable = train[~empty_rows(train)].sort_values(DATE, kind="stable").reset_index(drop=True)
    X_all = usable.drop(columns=TARGET)
    y_all = usable[TARGET].astype(int)
    last_day = usable[DATE].max()
    holdout_start = last_day - pd.Timedelta(days=window_days - 1)
    development_end = holdout_start - pd.Timedelta(days=HORIZON_DAYS)
    dev = usable[DATE] < development_end
    hold = usable[DATE] >= holdout_start
    X_dev, y_dev = X_all[dev].reset_index(drop=True), y_all[dev].reset_index(drop=True)
    starts = [development_end - pd.Timedelta(days=window_days * k)
              for k in range(n_folds, 0, -1)]
    folds = time_windows(X_dev[DATE], starts, window_days)
    return Design(X_dev, y_dev, X_all[hold].reset_index(drop=True),
                  y_all[hold].reset_index(drop=True), X_all, y_all, folds,
                  holdout_start, development_end, starts)


class RecentPeriodSplit:
    """One chronological split of the rows it receives: older rows to fit, the latest period to hold out.

    Used as the `cv` of CalibratedClassifierCV (notebook 05). The probability map is learned on the
    last `days` of the training rows, from a model fitted on rows more than `gap_days` earlier, so
    the time rule of the validation windows also holds inside each training window. With an
    integer `cv`, CalibratedClassifierCV uses StratifiedKFold, which ignores the time order.
    """

    def __init__(self, days=365, gap_days=HORIZON_DAYS):
        self.days = days
        self.gap_days = gap_days

    def split(self, X, y=None, groups=None):
        dates = pd.to_datetime(pd.Series(X[DATE]).reset_index(drop=True))
        start = dates.max() - pd.Timedelta(days=self.days)
        yield (np.flatnonzero(dates < start - pd.Timedelta(days=self.gap_days)),
               np.flatnonzero(dates >= start))

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def describe_folds(design: Design) -> pd.DataFrame:
    """One row per fold: training period, validation period, sizes and repeat rate."""
    dates = design.X_dev[DATE]
    rows = []
    for number, (fit_at, score_at) in enumerate(design.folds, start=1):
        rows.append({
            "fold": number,
            "train from": dates.iloc[fit_at].min().date(),
            "train to": dates.iloc[fit_at].max().date(),
            "validate from": dates.iloc[score_at].min().date(),
            "validate to": dates.iloc[score_at].max().date(),
            "train rows": len(fit_at),
            "validation rows": len(score_at),
            "validation repeat rate": round(design.y_dev.iloc[score_at].mean(), 3),
        })
    return pd.DataFrame(rows).set_index("fold")


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------

def scores(y_true, probability, threshold=0.5):
    """F1 plus the companions used in class, and the F1 of 'always predict repeat'.

    Week 5 and weeks 10-13: F1 with ROC-AUC and balanced accuracy as companions.
    New: F1_floor = F1 of predicting "repeat" for every row (= 2p / (1 + p)).
    Part 1 reported F1 0.76 without this reference; on this data the floor is ~0.72.
    """
    y_true = np.asarray(y_true)
    predicted = (np.asarray(probability) >= threshold).astype(int)
    return {
        "F1": f1_score(y_true, predicted, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probability),
        "balanced_acc": balanced_accuracy_score(y_true, predicted),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1_floor": f1_score(y_true, np.ones_like(y_true)),
    }


def _fit_fold(estimator, X, y, fit_at, score_at):
    model = clone(estimator).fit(X.iloc[fit_at], y.iloc[fit_at])
    return model.predict_proba(X.iloc[score_at])[:, 1]


def cross_validate_time(estimator, X, y, folds, threshold=0.5, n_jobs=-1):
    """Fit on each fold's training rows and score its validation window.

    Returns (per-fold score table, out-of-fold probabilities aligned with X).
    The estimator is cloned for every fold (a fresh, unfitted copy), so nothing
    learned in one window reaches another. Windows run in parallel.
    """
    probabilities = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(estimator, X, y, fit_at, score_at) for fit_at, score_at in folds)
    rows, oof = [], pd.Series(np.nan, index=X.index, dtype=float)
    for number, ((_, score_at), proba) in enumerate(zip(folds, probabilities), start=1):
        rows.append({"fold": number, **scores(y.iloc[score_at], proba, threshold)})
        oof.iloc[score_at] = proba
    return pd.DataFrame(rows).set_index("fold"), oof


def paired_delta(candidate, reference):
    """Mean fold-by-fold difference, its standard error, wins and a verdict.

    Verdict rule from class: a difference smaller than 2 standard errors is
    'inside the noise'.

    Week 4 / Part 1 (feature selection 3.4.3): the same paired comparison. Pairing
    by window removes the part of the variation that comes from one window being
    easier than another.
    """
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    sem = delta.std(ddof=1) / np.sqrt(len(delta))
    if np.allclose(delta, 0):
        verdict = "identical"
    elif abs(delta.mean()) <= 2 * sem:
        verdict = "inside the noise"
    else:
        verdict = "better" if delta.mean() > 0 else "worse"
    return {"mean_diff": delta.mean(), "SEM": sem,
            "wins": f"{int((delta > 0).sum())}/{len(delta)}", "verdict": verdict}


def summarise(table: pd.DataFrame, metrics=("ROC_AUC", "F1", "balanced_acc")):
    """Mean and standard deviation across folds for the chosen metrics."""
    return {f"{m} {stat}": getattr(table[m], stat)() for m in metrics for stat in ("mean", "std")}


def f1_by_threshold(y_true, probability, thresholds=None):
    """F1 at each candidate threshold (Week 5: the threshold is a choice too).

    New: Part 1 used the default 0.5. Only validation-window predictions are passed in,
    so the holdout never influences the threshold.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    return pd.Series([f1_score(y_true, (probability >= t).astype(int)) for t in thresholds],
                     index=pd.Index(thresholds, name="threshold"), name="F1")
