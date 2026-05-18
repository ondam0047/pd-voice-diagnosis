"""
CLI entry point for the improved ML pipeline.

Usage:

    python -m ml.run_pipeline train --csv training_data.csv
        runs 5-fold CV with all models x {acoustic_only, perceptual_only,
        combined} on Step-1 (PD vs Normal) and Step-2 (subtype) tasks,
        prints a report, and writes results to ml_reports/.

    python -m ml.run_pipeline extract --wav sample.wav
        prints acoustic features extracted from one WAV file.

The original Streamlit app.py is untouched. Once you confirm these results,
the model/feature pipeline can be swapped into the app at a later date.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data_loading import filter_pd_only, load_training_data
from .evaluation import format_result, results_to_dataframe
from .training import CVResult, cv_evaluate, model_zoo

REPORT_DIR = Path("ml_reports")
FEATURE_SETS = ["acoustic_only", "perceptual_only", "combined"]


def run_training(csv_path: Path) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    all_results: list[CVResult] = []

    tasks = [
        ("step1_pd_vs_normal", True, False),
        ("step2_subtype", False, True),
    ]

    for task_name, binary, filter_pd in tasks:
        print(f"\n{'#' * 72}\n# TASK: {task_name}\n{'#' * 72}")
        for fs in FEATURE_SETS:
            data = load_training_data(
                csv_path, feature_set=fs, binary_task=binary
            )
            if filter_pd:
                data = filter_pd_only(data)
            print(
                f"\n--- feature_set={fs}  "
                f"n_samples={len(data.y)}  classes={data.class_names}"
            )
            print("    class counts:", data.y.value_counts().to_dict())

            for name, pipe in model_zoo(use_smote=True).items():
                try:
                    result = cv_evaluate(
                        X=data.X,
                        y=data.y,
                        pipeline=pipe,
                        model_name=name,
                        feature_set=f"{task_name}/{fs}",
                    )
                    print(format_result(result))
                    all_results.append(result)
                except Exception as e:
                    print(f"  [SKIP] {name}: {e}")

    summary = results_to_dataframe(all_results)
    print("\n" + "=" * 72)
    print("CROSS-COMPARISON (sorted by macro-F1 within each feature_set):")
    print(summary.to_string(index=False))

    summary_path = REPORT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n-> saved {summary_path}")


def run_extract(wav_path: Path) -> None:
    from .feature_extraction import extract_features

    feats = extract_features(wav_path)
    print(json.dumps(feats.to_flat_dict(), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.run_pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="run full CV training report")
    p_train.add_argument("--csv", type=Path, default=Path("training_data.csv"))

    p_extract = sub.add_parser(
        "extract", help="extract acoustic features from a WAV file"
    )
    p_extract.add_argument("--wav", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        run_training(args.csv)
    elif args.cmd == "extract":
        run_extract(args.wav)
    return 0


if __name__ == "__main__":
    sys.exit(main())
