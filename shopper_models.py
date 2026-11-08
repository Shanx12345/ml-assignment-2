"""Shared schema, preprocessing, estimators and evaluation for the learning reference."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
TARGET = 'Revenue'
COUNT_AND_TIME = ['Administrative', 'Administrative_Duration', 'Informational',
                  'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']
RATES = ['BounceRates', 'ExitRates', 'SpecialDay']
CATEGORY_CODES = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
CATEGORIES = ['Month', *CATEGORY_CODES, 'VisitorType', 'Weekend']
FEATURES = COUNT_AND_TIME + RATES + CATEGORIES
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib',
}


def _binary(series, label, allow_missing=False):
    text = series.astype(str).str.strip().str.lower()
    value = text.map({'true': 1, 'false': 0, '1': 1, '0': 0,
                      '1.0': 1, '0.0': 0})
    bad = value.isna() & ~(series.isna() if allow_missing else False)
    if bad.any():
        raise ValueError(f'{label}: use True/False or 1/0; invalid row count {bad.sum()}.')
    return value


def normalise_frame(frame):
    """Check raw columns and values, returning a canonical X and binary y.

    Additional columns (including PageValues) are deliberately ignored. Required
    columns must exist. Empty feature cells can be imputed by the saved pipeline.
    """
    if frame.empty:
        raise ValueError('The CSV has no data rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    x = frame.loc[:, FEATURES].copy().replace(r'^\s*$', np.nan, regex=True)
    y = _binary(frame[TARGET], TARGET).astype(int)
    for col in COUNT_AND_TIME + RATES + CATEGORY_CODES:
        values = pd.to_numeric(x[col], errors='coerce')
        if (values.isna() & x[col].notna()).any():
            raise ValueError(f'{col}: contains a non-numeric value.')
        if np.isinf(values).any() or (values.dropna() < 0).any():
            raise ValueError(f'{col}: values must be finite and non-negative.')
        if col in RATES and (values.dropna() > 1).any():
            raise ValueError(f'{col}: values must be between 0 and 1.')
        if col in CATEGORY_CODES + ['Administrative', 'Informational', 'ProductRelated']:
            if (values.dropna() % 1 != 0).any():
                raise ValueError(f'{col}: use whole numbers.')
        x[col] = values
    for col in CATEGORY_CODES:
        x[col] = x[col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    for col in ['Month', 'VisitorType']:
        x[col] = x[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    weekend = _binary(x['Weekend'], 'Weekend', allow_missing=True)
    x['Weekend'] = weekend.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return x, y


def make_preprocessor():
    # Fit imputation, scaling and category vocabulary only on training rows.
    skewed = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ])
    rates = Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())])
    categories = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('count_time', skewed, COUNT_AND_TIME),
        ('rates', rates, RATES),
        ('categories', categories, CATEGORIES),
    ], remainder='drop', sparse_threshold=0)


def make_pipeline(name, seed, params):
    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced',
                                                 random_state=seed),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=seed),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'Naive Bayes': GaussianNB(),
        # AUC uses the margin; probability=True would add expensive calibration.
        'Support Vector Machine': SVC(kernel='rbf', class_weight='balanced',
                                      probability=False, cache_size=256, random_state=seed),
    }
    estimator = estimators[name].set_params(**params)
    return Pipeline([('prepare', make_preprocessor()), ('classifier', estimator)])


def candidate_parameters():
    return {
        'Logistic Regression': [{'C': c} for c in [0.1, 1.0, 10.0]],
        'Decision Tree': [{'max_depth': d, 'min_samples_leaf': 20} for d in [4, 6, 9]],
        'KNN': [{'n_neighbors': k, 'weights': 'distance'} for k in [15, 31, 61]],
        'Naive Bayes': [{'var_smoothing': s} for s in [1e-9, 1e-5, 1e-2]],
        'Support Vector Machine': [{'C': c, 'gamma': 'scale'} for c in [0.5, 2.0, 8.0]],
    }


def evaluate_pipeline(pipeline, x, y):
    predicted = pipeline.predict(x)
    if hasattr(pipeline, 'decision_function'):
        score = pipeline.decision_function(x)
        score_label = 'Decision margin for Revenue=1'
    else:
        positive_index = list(pipeline.classes_).index(1)
        score = pipeline.predict_proba(x)[:, positive_index]
        score_label = 'Estimated probability for Revenue=1'
    metrics = {
        'Accuracy': float(accuracy_score(y, predicted)),
        'AUC': float(roc_auc_score(y, score)) if y.nunique() == 2 else None,
        'Precision': float(precision_score(y, predicted, zero_division=0)),
        'Recall': float(recall_score(y, predicted, zero_division=0)),
        'F1': float(f1_score(y, predicted, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, predicted)),
    }
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    report = classification_report(y, predicted, labels=[0, 1],
                                   target_names=['No purchase', 'Purchase'],
                                   output_dict=True, zero_division=0)
    return metrics, matrix, report, predicted, score, score_label
