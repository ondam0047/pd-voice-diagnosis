"""
Human-readable reporting helpers for CVResult objects.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .training import CVResult


def format_confusion_matrix(cm: np.ndarray, classes: list[str]) -> str:
    buf = io.StringIO()
    width = max(max(len(c) for c in classes) + 2, 12)
    header = " " * width + "".join(c.rjust(width) for c in classes)
    buf.write(header + "\n")
    for i, c in enumerate(classes):
        row = c.ljust(width) + "".join(str(int(v)).rjust(width) for v in cm[i])
        buf.write(row + "\n")
    return buf.getvalue()


def format_result(result: CVResult) -> str:
    out: list[str] = []
    out.append(f"=== {result.model_name}  |  features: {result.feature_set} ===")
    folds = "  ".join(f"{f:.3f}" for f in result.fold_f1_macro)
    out.append(
        f"  Macro-F1 (mean +/- sd):  "
        f"{result.mean_f1_macro:.3f} +/- {result.sd_f1_macro:.3f}  "
        f"(folds: {folds})"
    )
    out.append(
        f"  Balanced accuracy:       {result.mean_balanced_accuracy:.3f}"
    )
    out.append("  Per-class precision / recall / F1:")
    for cls in result.classes:
        r = result.per_class_report.get(cls, {})
        out.append(
            f"    {cls:25s}  P={r.get('precision', 0):.2f}  "
            f"R={r.get('recall', 0):.2f}  F1={r.get('f1-score', 0):.2f}  "
            f"n={int(r.get('support', 0))}"
        )
    out.append("  Confusion matrix (rows=true, cols=pred):")
    out.append(format_confusion_matrix(result.confusion_matrix, result.classes))
    return "\n".join(out)


def results_to_dataframe(results: list[CVResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model_name,
                "features": r.feature_set,
                "n_classes": r.n_classes,
                "macro_f1_mean": r.mean_f1_macro,
                "macro_f1_sd": r.sd_f1_macro,
                "balanced_accuracy": r.mean_balanced_accuracy,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["features", "macro_f1_mean"], ascending=[True, False]
    )
