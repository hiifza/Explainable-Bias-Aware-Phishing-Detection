"""
src/features/pipeline_builder.py
----------------------------------
Constructs, fits, serialises, and loads the complete sklearn preprocessing
pipelines for Track A and Track B.

Pipeline step order
-------------------
1. FeatureEngineer         — add 7 derived columns (uses TLD as string)
2. TLDFrequencyEncoder     — replace TLD string with int frequency rank
3. OutlierClipper          — cap at per-feature P99.9 (train-only fit)
4. Log1pTransformer        — log1p on approved skewed features
5. RobustScalerTransformer — RobustScaler on continuous non-binary cols

The same step sequence is used for both tracks; the only difference is
that Track A includes URLSimilarityIndex in the input DataFrame.

Public API
----------
    build_pipeline(track)                                -> Pipeline
    fit_pipeline(pipeline, X_train)                      -> Pipeline
    transform_data(pipeline, X)                          -> pd.DataFrame
    save_pipeline(pipeline, path)
    load_pipeline(path)                                  -> Pipeline
    run_full_preprocessing(split_results, output_dir, reports_dir) -> dict
"""

import sys
from pathlib import Path
from typing  import Optional

import joblib
import numpy  as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.features.encoding      import TLDFrequencyEncoder, build_tld_encoding_report
from src.features.preprocessing import (
    OutlierClipper, Log1pTransformer, RobustScalerTransformer
)
from src.features.engineering   import FeatureEngineer
from src.features.validation    import run_full_validation
from src.features.feature_catalog import TARGET_COLUMN

logger = get_logger(__name__)


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(track: str = "B") -> Pipeline:
    """
    Construct the sklearn Pipeline for the given track.

    Both tracks use the identical step sequence; the difference is
    handled by which columns are present in the input DataFrame.

    Parameters
    ----------
    track : "A" or "B"

    Returns
    -------
    sklearn.pipeline.Pipeline  (unfitted)
    """
    track = track.upper()
    assert track in ("A", "B"), f"track must be 'A' or 'B', got '{track}'"

    pipeline = Pipeline(
        steps=[
            ("engineer",  FeatureEngineer()),
            ("tld_enc",   TLDFrequencyEncoder(top_n=50)),
            ("clipper",   OutlierClipper(quantile=0.999)),
            ("log1p",     Log1pTransformer()),
            ("scaler",    RobustScalerTransformer()),
        ],
        verbose=False,
    )

    logger.info(
        f"Built pipeline for Track {track} with steps: "
        f"{[s[0] for s in pipeline.steps]}"
    )
    return pipeline


# ── Fit / transform ───────────────────────────────────────────────────────────

def fit_pipeline(
    pipeline: Pipeline,
    X_train : pd.DataFrame,
) -> Pipeline:
    """
    Fit all pipeline steps on the training set only.

    Parameters
    ----------
    pipeline : unfitted Pipeline
    X_train  : training-set feature DataFrame (no label column)

    Returns
    -------
    Fitted Pipeline
    """
    logger.info(f"Fitting pipeline on {len(X_train):,} training rows "
                f"× {X_train.shape[1]} columns …")

    # Custom fit loop — each step is fitted and transformed sequentially
    # so later steps receive already-transformed data.
    # Log1pTransformer uses fit_transform to capture before/after skewness.
    X_current = X_train.copy()
    X_after_clip = None  # captured for intermediate validation

    for name, step in pipeline.steps:
        if hasattr(step, "fit_transform"):
            X_current = step.fit_transform(X_current)
        else:
            step.fit(X_current)
            X_current = step.transform(X_current)
        logger.debug(f"  Step '{name}' fitted — output shape: {X_current.shape}")

        # Capture state immediately after clipping for threshold validation
        if name == "clipper":
            X_after_clip = X_current.copy()

    # Validate that clipping worked on the intermediate (pre-log1p, pre-scale) data
    if X_after_clip is not None:
        clipper = pipeline.named_steps["clipper"]
        for col, thresh in clipper.thresholds_.items():
            if col in X_after_clip.columns:
                actual_max = float(X_after_clip[col].max())
                assert actual_max <= thresh + 1e-9, (
                    f"Outlier clipping FAILED for '{col}': "
                    f"max after clip = {actual_max:.6f}, threshold = {thresh:.6f}"
                )
        logger.info("Intermediate outlier clipping validation PASSED ✓")

    logger.info("Pipeline fitting complete ✓")
    return pipeline


def transform_data(
    pipeline: Pipeline,
    X       : pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a fitted pipeline to a DataFrame (train OR test).

    Parameters
    ----------
    pipeline : fitted Pipeline
    X        : feature DataFrame (no label column)

    Returns
    -------
    pd.DataFrame  — transformed feature matrix
    """
    X_out = X.copy()
    for name, step in pipeline.steps:
        X_out = step.transform(X_out)
        logger.debug(f"  Step '{name}' transform done — shape: {X_out.shape}")
    return X_out


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_pipeline(
    pipeline: Pipeline,
    path    : str | Path,
) -> None:
    """
    Serialise a fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    path     : destination .pkl file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    size_kb = path.stat().st_size / 1024
    logger.info(f"Pipeline saved → {path}  ({size_kb:.1f} KB)")


def load_pipeline(path: str | Path) -> Pipeline:
    """
    Load a serialised pipeline from disk.

    Parameters
    ----------
    path : .pkl file path

    Returns
    -------
    sklearn Pipeline  (already fitted)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    pipeline = joblib.load(path)
    logger.info(f"Pipeline loaded ← {path}")
    return pipeline


# ── Reporting ─────────────────────────────────────────────────────────────────

def _save_tld_report(pipeline: Pipeline, output_dir: Path) -> None:
    enc = pipeline.named_steps["tld_enc"]
    report = build_tld_encoding_report(enc)
    path   = output_dir / "tld_encoding_report.csv"
    report.to_csv(path, index=False)
    logger.info(f"Saved: tld_encoding_report.csv  ({len(report)} rows)")


def _save_skewness_report(pipeline: Pipeline, output_dir: Path) -> None:
    log1p = pipeline.named_steps["log1p"]
    if hasattr(log1p, "get_skewness_report"):
        report = log1p.get_skewness_report()
        path   = output_dir / "skewness_correction_report.csv"
        report.to_csv(path, index=False)
        logger.info(f"Saved: skewness_correction_report.csv  ({len(report)} rows)")


def _save_outlier_report(pipeline: Pipeline, output_dir: Path) -> None:
    clipper = pipeline.named_steps["clipper"]
    if hasattr(clipper, "get_thresholds_df"):
        report = clipper.get_thresholds_df()
        path   = output_dir / "outlier_thresholds.csv"
        report.to_csv(path, index=False)
        logger.info(f"Saved: outlier_thresholds.csv  ({len(report)} rows)")


def _save_scaling_report(pipeline: Pipeline, output_dir: Path) -> None:
    scaler = pipeline.named_steps["scaler"]
    if hasattr(scaler, "get_scaling_report"):
        report = scaler.get_scaling_report()
        path   = output_dir / "scaling_report.csv"
        report.to_csv(path, index=False)
        logger.info(f"Saved: scaling_report.csv  ({len(report)} rows)")


def _save_engineered_report(
    X_before: pd.DataFrame,
    X_after : pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save a distribution + correlation summary of the 7 engineered features.
    """
    from src.features.engineering import ENGINEERED_FEATURE_NAMES

    rows = []
    for feat in ENGINEERED_FEATURE_NAMES:
        if feat not in X_after.columns:
            continue
        s = X_after[feat]
        corr_with_label = float("nan")
        if TARGET_COLUMN in X_after.columns:
            corr_with_label = float(X_after[[feat, TARGET_COLUMN]].corr().iloc[0, 1])
        elif TARGET_COLUMN in X_before.columns:
            combined = X_after[[feat]].copy()
            combined[TARGET_COLUMN] = X_before[TARGET_COLUMN].values
            corr_with_label = float(combined.corr().iloc[0, 1])

        rows.append({
            "feature"         : feat,
            "mean"            : round(float(s.mean()), 6),
            "std"             : round(float(s.std()),  6),
            "min"             : float(s.min()),
            "median"          : float(s.median()),
            "max"             : float(s.max()),
            "n_unique"        : int(s.nunique()),
            "corr_with_label" : round(corr_with_label, 6),
        })

    report = pd.DataFrame(rows)
    path   = output_dir / "engineered_features_report.csv"
    report.to_csv(path, index=False)
    logger.info(f"Saved: engineered_features_report.csv  ({len(report)} rows)")


# ── Full pipeline orchestrator ────────────────────────────────────────────────

def run_full_preprocessing(
    split_results: dict,
    preprocessor_dir: str | Path = "outputs/preprocessors",
    processed_dir   : str | Path = "data/processed",
    reports_dir     : str | Path = "outputs/reports",
) -> dict:
    """
    Fit, transform, validate, serialise, and save both Track A and Track B
    preprocessing pipelines.

    Parameters
    ----------
    split_results    : dict returned by splitter.run_split_pipeline()
    preprocessor_dir : where to save .pkl files
    processed_dir    : base dir for preprocessed CSV splits
    reports_dir      : where to save all report CSVs

    Returns
    -------
    dict  keys: pipeline_A, pipeline_B,
                X_train_A, X_test_A, X_train_B, X_test_B,
                validation_A, validation_B
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info("M3.1 — FULL PREPROCESSING PIPELINE")
    logger.info(sep)

    preprocessor_dir = Path(preprocessor_dir)
    processed_dir    = Path(processed_dir)
    reports_dir      = Path(reports_dir)
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_train = split_results["y_train"]
    y_test  = split_results["y_test"]

    results = {}

    for track in ("A", "B"):
        logger.info(f"\n{'─'*40}")
        logger.info(f"TRACK {track}")
        logger.info(f"{'─'*40}")

        X_train_raw = split_results[f"track_{track}_train"].drop(
            columns=[TARGET_COLUMN], errors="ignore"
        )
        X_test_raw  = split_results[f"track_{track}_test"].drop(
            columns=[TARGET_COLUMN], errors="ignore"
        )

        # 1. Build pipeline
        pipeline = build_pipeline(track)

        # 2. Fit on training set only
        pipeline = fit_pipeline(pipeline, X_train_raw)

        # 3. Transform training set
        X_train_pp = transform_data(pipeline, X_train_raw)

        # 4. Transform test set (no re-fitting)
        X_test_pp  = transform_data(pipeline, X_test_raw)

        logger.info(
            f"Track {track} shapes — "
            f"X_train: {X_train_pp.shape}  X_test: {X_test_pp.shape}"
        )

        # 5. Add label column back for validation
        X_train_with_label = X_train_pp.copy()
        X_train_with_label[TARGET_COLUMN] = y_train[TARGET_COLUMN].values

        X_train_raw_with_label = X_train_raw.copy()
        X_train_raw_with_label[TARGET_COLUMN] = y_train[TARGET_COLUMN].values

        # 6. Validate
        thresholds = pipeline.named_steps["clipper"].thresholds_
        val_result = run_full_validation(
            df_before  = X_train_raw_with_label,
            df_after   = X_train_with_label,
            track      = track,
            thresholds = thresholds,
        )
        results[f"validation_{track}"] = val_result

        if not val_result["all_passed"]:
            raise RuntimeError(
                f"Track {track} validation FAILED — "
                f"{val_result['n_fail']} check(s) did not pass"
            )

        # 7. Save reports
        _save_tld_report(pipeline, reports_dir)
        _save_skewness_report(pipeline, reports_dir)
        _save_outlier_report(pipeline, reports_dir)
        _save_scaling_report(pipeline, reports_dir)
        _save_engineered_report(X_train_raw_with_label, X_train_with_label,
                                reports_dir)

        # 8. Save preprocessed splits to CSV
        track_dir = processed_dir / f"track_{track}"
        track_dir.mkdir(parents=True, exist_ok=True)

        X_train_pp.to_csv(track_dir / "X_train.csv", index=False)
        X_test_pp.to_csv(  track_dir / "X_test.csv",  index=False)
        logger.info(
            f"Saved preprocessed: track_{track}/X_train.csv  "
            f"track_{track}/X_test.csv"
        )

        # 9. Serialise pipeline
        pkl_path = preprocessor_dir / f"preprocessor_{track}.pkl"
        save_pipeline(pipeline, pkl_path)

        results[f"pipeline_{track}"] = pipeline
        results[f"X_train_{track}"]  = X_train_pp
        results[f"X_test_{track}"]   = X_test_pp

    logger.info(sep)
    logger.info("PREPROCESSING COMPLETE — both tracks fitted, validated, saved")
    logger.info(sep)
    return results
