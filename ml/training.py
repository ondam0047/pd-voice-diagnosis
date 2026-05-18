"""
Train and cross-validate classifiers for the PD subtype task.

Methodological improvements over the original app.py:

  1. Held-out evaluation via StratifiedKFold. Every metric reported is
     computed on samples the model did NOT see during training.

  2. Comparable model panel — LogisticRegression, LinearDiscriminantAnalysis
     (the two used by the original app, reproduced for comparison),
     RandomForest, SVM (rbf), GradientBoosting.

  3. Class imbalance handled via class_weight='balanced' AND optional SMOTE
     (imbalanced-learn). PD_Rate has only ~5 samples; without resampling
     LR/LDA effectively ignore it.

  4. Feature-set ablation: train separately with acoustic_only,
     perceptual_only, and combined. If perceptual_only >> acoustic_only
     by a large margin, that is empirical evidence of the label-definition
     circularity discussed in docs/CODE_REVIEW.md.

  5. No leakage from imputer / scaler / SMOTE — everything is inside a
     Pipeline so each fold gets a fresh fit on the training portion only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def _build_pipeline(clf, use_smote: bool, smote_k: int = 2):
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]
    if use_smote and HAS_IMBLEARN:
        steps.append(("sm", SMOTE(k_neighbors=smote_k, random_state=42)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)


def model_zoo(use_smote: bool = True) -> dict:
    return {
        "logistic_regression": _build_pipeline(
            LogisticRegression(
                max_iter=4000, class_weight="balanced", random_state=42
            ),
            use_smote,
        ),
        "linear_discriminant": _build_pipeline(
            LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
            use_smote,
        ),
        "random_forest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            use_smote,
        ),
        "svm_rbf": _build_pipeline(
            SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            use_smote,
        ),
        "gradient_boosting": _build_pipeline(
            GradientBoostingClassifier(random_state=42),
            use_smote,
        ),
    }


@dataclass
class CVResult:
    model_name: str
    feature_set: str
    n_classes: int
    classes: list[str]
    fold_f1_macro: list[float]
    mean_f1_macro: float
    sd_f1_macro: float
    mean_balanced_accuracy: float
    confusion_matrix: np.ndarray
    per_class_report: dict


def cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    model_name: str,
    feature_set: str,
    n_splits: int = 5,
) -> CVResult:
    """Stratified k-fold CV producing out-of-fold predictions for every sample."""
    classes = sorted(y.unique().tolist())
    n_classes = len(classes)
    min_cls = int(y.value_counts().min())
    n_splits = min(n_splits, min_cls)
    if n_splits < 2:
        raise ValueError(
            f"Smallest class has only {min_cls} sample(s); CV impossible."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pred = np.empty(len(y), dtype=object)
    fold_f1s: list[float] = []
    bacc_sum = 0.0

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(Xtr, ytr)
        pred = pipeline.predict(Xte)
        oof_pred[test_idx] = pred
        fold_f1s.append(
            f1_score(yte, pred, average="macro", zero_division=0)
        )
        bacc_sum += balanced_accuracy_score(yte, pred)

    cm = confusion_matrix(y, oof_pred, labels=classes)
    report = classification_report(
        y, oof_pred, labels=classes, output_dict=True, zero_division=0
    )

    return CVResult(
        model_name=model_name,
        feature_set=feature_set,
        n_classes=n_classes,
        classes=classes,
        fold_f1_macro=fold_f1s,
        mean_f1_macro=float(np.mean(fold_f1s)),
        sd_f1_macro=float(np.std(fold_f1s)),
        mean_balanced_accuracy=float(bacc_sum / n_splits),
        confusion_matrix=cm,
        per_class_report=report,
    )


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
) -> Pipeline:
    """Refit on the full dataset for deployment, AFTER CV metrics are reported."""
    pipeline.fit(X, y)
    return pipeline
