"""
Load and clean the PD voice training data.

Key decisions vs the original app.py:
  - VHI columns: missing values are kept as NaN (NOT filled with 0). Filling
    with 0 makes 'VHI was recorded?' a stand-in feature for the normal/PD
    label.
  - Three feature subsets are exposed so the model can be evaluated against
    each independently:
        acoustic_only  : F0, Range, dB, SPS, sex                     (objective)
        perceptual_only: 5 perceptual + 4 VHI scores                 (subjective)
        combined       : everything
    Most reports should lead with acoustic_only because the perceptual scores
    were used to define the subtype labels themselves (circular).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ACOUSTIC_COLS = ["F0", "Range", "ਵ도(dB)", "SPS"]
PERCEPTUAL_COLS = [
    "음도(청지각)",
    "음도범위(청지각)",
    "강도(청지각)",
    "말속도(청지각)",
    "조음정확도(청지각)",
]
VHI_COLS = ["VHI총점", "VHI_신체", "VHI_기능", "VHI_정서"]
LABEL_COL = "진단결과 (Label)"
SEX_COL = "성별"

# Fix: typo in the spec used above (ਵ instead of 강). Recreate accurately.
ACOUSTIC_COLS = ["F0", "Range", "강도(dB)", "SPS"]

FeatureSet = Literal["acoustic_only", "perceptual_only", "combined"]


@dataclass
class LoadedData:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    class_names: list[str]


def _encode_sex(s: pd.Series) -> pd.Series:
    # 남 -> 1, 여 -> 0
    return s.astype(str).str.strip().map({"남": 1, "여": 0}).astype(float)


def load_training_data(
    csv_path: Path | str = "training_data.csv",
    feature_set: FeatureSet = "acoustic_only",
    binary_task: bool = False,
) -> LoadedData:
    """
    Load training_data.csv into X/y.

    Args:
        csv_path: path to the training CSV.
        feature_set: which subset of columns to expose to the model.
            'acoustic_only' (recommended) avoids label leakage from perceptual
            scores. See docs/CODE_REVIEW.md C1 for the rationale.
        binary_task: if True, collapse all PD_* into a single 'PD' class
            (Step-1 screening). For subtype classification (Step-2), keep this
            False and additionally call filter_pd_only().
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for col in ACOUSTIC_COLS + PERCEPTUAL_COLS + VHI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip(), errors="coerce"
            )

    if feature_set == "acoustic_only":
        feat_cols = [SEX_COL] + ACOUSTIC_COLS
    elif feature_set == "perceptual_only":
        feat_cols = [SEX_COL] + PERCEPTUAL_COLS + VHI_COLS
    else:
        feat_cols = [SEX_COL] + ACOUSTIC_COLS + PERCEPTUAL_COLS + VHI_COLS

    X = df[feat_cols].copy()
    X[SEX_COL] = _encode_sex(X[SEX_COL])

    y = df[LABEL_COL].astype(str).str.strip()
    if binary_task:
        y = y.where(y == "normal", "PD")

    return LoadedData(
        X=X,
        y=y,
        feature_names=feat_cols,
        class_names=sorted(y.unique().tolist()),
    )


def filter_pd_only(data: LoadedData) -> LoadedData:
    """Keep only PD rows (drop 'normal'). Use for subtype classification."""
    mask = data.y != "normal"
    return LoadedData(
        X=data.X.loc[mask].reset_index(drop=True),
        y=data.y.loc[mask].reset_index(drop=True),
        feature_names=data.feature_names,
        class_names=sorted(data.y.loc[mask].unique().tolist()),
    )
