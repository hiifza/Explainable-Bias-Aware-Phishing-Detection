"""
src/features/validation.py
---------------------------
Post-pipeline validation framework for Module M3.1.

Every check raises AssertionError with a descriptive message on failure,
so the pipeline "fails loudly" rather than silently producing corrupt data.

Public API
----------
    validate_output_shape(df, expected_n_features, track)
    validate_no_missing(df)
    validate_feature_types(df, binary_features)
    validate_feature_count(df, track)
    validate_transformations(df_before, df_after, log_feats, clipper)
    validate_tld_encoded(df_before, df_after)
    validate_binary_unchanged(df_before, df_after, binary_features)
    run_full_validation(df_before, df_after, track, clipper) -> dict
"""

import sys
from pathlib import Path
from typing import Optional

import numpy  as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger                import get_logger
from src.features.preprocessing      import BINARY_FEATURES, LOG1P_FEATURES
from src.features.engineering        import ENGINEERED_FEATURE_NAMES

logger = get_logger(__name__)

# Expected feature counts post-pipeline (features only, no label column)
EXPECTED_FEATURES = {"A": 57, "B": 56}
TARGET_COLUMN     = "label"


# ── Individual validators ─────────────────────────────────────────────────────

def validate_output_shape(
    df                : pd.DataFrame,
    track             : str,
    include_label     : bool = False,
) -> None:
    """
    Assert that df has the correct number of columns for the given track.

    Parameters
    ----------
    df            : preprocessed DataFrame (features only, or with label)
    track         : "A" or "B"
    include_label : if True, expected_cols = EXPECTED_FEATURES[track] + 1
    """
    expected = EXPECTED_FEATURES[track.upper()]
    if include_label:
        expected += 1

    actual = df.shape[1]
    assert actual == expected, (
        f"Shape validation FAILED for Track {track}: "
        f"expected {expected} columns, got {actual}. "
        f"Columns present: {list(df.columns)}"
    )
    logger.info(f"Shape validation PASSED ✓  Track {track}: {actual} columns")


def validate_no_missing(df: pd.DataFrame, stage: str = "") -> None:
    """Assert that df contains zero NaN or infinite values."""
    n_null = int(df.isnull().sum().sum())
    assert n_null == 0, (
        f"Missing-value validation FAILED {stage}: "
        f"{n_null} NaN values found in columns: "
        f"{list(df.columns[df.isnull().any()])}"
    )

    # Check for infinities in numeric columns
    num_df = df.select_dtypes(include=[np.number])
    n_inf  = int(np.isinf(num_df.values).sum())
    assert n_inf == 0, (
        f"Infinite-value validation FAILED {stage}: "
        f"{n_inf} inf values found"
    )
    logger.info(f"Missing/inf validation PASSED ✓  {stage}")


def validate_feature_types(
    df             : pd.DataFrame,
    binary_features: Optional[list[str]] = None,
) -> None:
    """
    Assert that:
    - All columns except TLD (after encoding) are numeric.
    - Specified binary features contain only 0 and 1.
    """
    bin_feats = binary_features if binary_features is not None else BINARY_FEATURES

    # All columns should be numeric at this stage (TLD is int after encoding)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    assert len(non_numeric) == 0, (
        f"Type validation FAILED: non-numeric columns found after pipeline: "
        f"{non_numeric}"
    )

    # Binary features must contain only 0 and 1
    for feat in bin_feats:
        if feat not in df.columns:
            continue
        unique_vals = set(df[feat].unique())
        assert unique_vals.issubset({0, 1, np.int8(0), np.int8(1)}), (
            f"Type validation FAILED: binary feature '{feat}' "
            f"contains values other than 0/1: {unique_vals}"
        )

    logger.info("Type validation PASSED ✓")


def validate_feature_count(df: pd.DataFrame, track: str) -> None:
    """
    Assert the feature count (excluding label) matches the expected value.
    """
    feat_cols = [c for c in df.columns if c != TARGET_COLUMN]
    expected  = EXPECTED_FEATURES[track.upper()]
    actual    = len(feat_cols)

    assert actual == expected, (
        f"Feature count validation FAILED for Track {track}: "
        f"expected {expected} features, got {actual}"
    )
    logger.info(
        f"Feature count PASSED ✓  Track {track}: {actual} features"
    )


def validate_tld_encoded(
    df_before: pd.DataFrame,
    df_after : pd.DataFrame,
) -> None:
    """
    Assert that TLD column is object/string before encoding and
    numeric (integer or float) after encoding.

    NOTE: After RobustScaler the TLD column may be negative (centered
    at median).  We only check dtype, not sign.
    """
    tld_col = "TLD"
    if tld_col not in df_before.columns or tld_col not in df_after.columns:
        logger.warning("TLD column not found — skipping TLD encoding check")
        return

    before_dtype = str(df_before[tld_col].dtype)
    after_dtype  = str(df_after[tld_col].dtype)

    # After TLD encoding + optional scaling the column must be numeric
    assert df_after[tld_col].dtype in [np.int64, np.float64,
                                        np.int32, np.float32], (
        f"TLD encoding validation FAILED: "
        f"TLD column dtype after pipeline is '{after_dtype}', "
        f"expected numeric (int or float)"
    )

    logger.info(
        f"TLD encoding PASSED ✓  dtype: {before_dtype} → {after_dtype}"
    )


def validate_log1p_applied(
    df_before: pd.DataFrame,
    df_after : pd.DataFrame,
    log_feats: Optional[list[str]] = None,
) -> None:
    """
    Assert that log1p features have smaller maximum values after transform
    (the log1p transform always reduces large values).
    """
    feats = log_feats if log_feats is not None else LOG1P_FEATURES
    for feat in feats:
        if feat not in df_before.columns or feat not in df_after.columns:
            continue
        max_before = float(df_before[feat].max())
        max_after  = float(df_after[feat].max())
        if max_before > 1:   # skip features already near-zero
            assert max_after < max_before, (
                f"Log1p validation FAILED for '{feat}': "
                f"max_after ({max_after:.2f}) ≥ max_before ({max_before:.2f})"
            )

    logger.info("Log1p transformation PASSED ✓")


def validate_outliers_clipped(
    df_before : pd.DataFrame,
    df_after  : pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    """
    Assert that no value in df_after exceeds its clipping threshold.
    """
    for col, thresh in thresholds.items():
        if col not in df_after.columns:
            continue
        actual_max = float(df_after[col].max())
        assert actual_max <= thresh + 1e-9, (
            f"Outlier clipping FAILED for '{col}': "
            f"max after clip = {actual_max:.4f}, threshold = {thresh:.4f}"
        )

    logger.info("Outlier clipping PASSED ✓")


def validate_binary_unchanged(
    df_before      : pd.DataFrame,
    df_after       : pd.DataFrame,
    binary_features: Optional[list[str]] = None,
) -> None:
    """
    Assert that binary (0/1) features are identical before and after
    the full pipeline (they should not be scaled or log-transformed).
    """
    bin_feats = binary_features if binary_features is not None else BINARY_FEATURES

    for feat in bin_feats:
        if feat not in df_before.columns or feat not in df_after.columns:
            continue
        before_vals = df_before[feat].values
        after_vals  = df_after[feat].values
        assert np.array_equal(before_vals, after_vals), (
            f"Binary feature validation FAILED: '{feat}' changed "
            f"after pipeline (should be unchanged)"
        )

    logger.info("Binary features unchanged PASSED ✓")


def validate_engineered_features(df_after: pd.DataFrame) -> None:
    """Assert all 7 engineered features are present and non-null."""
    for feat in ENGINEERED_FEATURE_NAMES:
        assert feat in df_after.columns, \
            f"Engineered feature '{feat}' missing from pipeline output"
        n_null = int(df_after[feat].isnull().sum())
        assert n_null == 0, \
            f"Engineered feature '{feat}' has {n_null} NaN values"

    logger.info(
        f"Engineered features PASSED ✓  "
        f"all {len(ENGINEERED_FEATURE_NAMES)} present and non-null"
    )


# ── Full validation orchestrator ──────────────────────────────────────────────

def run_full_validation(
    df_before : pd.DataFrame,
    df_after  : pd.DataFrame,
    track     : str,
    thresholds: Optional[dict[str, float]] = None,
) -> dict:
    """
    Execute all validation checks and return a summary dict.

    Parameters
    ----------
    df_before  : raw feature DataFrame BEFORE pipeline (with label column)
    df_after   : preprocessed DataFrame AFTER pipeline (with label column)
    track      : "A" or "B"
    thresholds : OutlierClipper.thresholds_ dict (for clipping validation)

    Returns
    -------
    dict  keys: all_passed, n_checks, results (list of check dicts)
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info(f"M3.1 PIPELINE VALIDATION — Track {track.upper()}")
    logger.info(sep)

    checks = []

    # Helpers to run a check and record result
    def run_check(name: str, fn, *args, **kwargs) -> bool:
        try:
            fn(*args, **kwargs)
            checks.append({"check": name, "status": "PASS", "error": None})
            return True
        except AssertionError as e:
            checks.append({"check": name, "status": "FAIL", "error": str(e)})
            logger.error(f"FAIL: {name} — {e}")
            return False
        except Exception as e:
            checks.append({"check": name, "status": "ERROR", "error": str(e)})
            logger.error(f"ERROR: {name} — {e}")
            return False

    # Separate feature frames (no label) for shape/type checks
    feat_after  = df_after.drop(columns=[TARGET_COLUMN], errors="ignore")
    feat_before = df_before.drop(columns=[TARGET_COLUMN], errors="ignore")

    run_check("output_shape",         validate_output_shape,      feat_after, track)
    run_check("no_missing_values",    validate_no_missing,        feat_after, f"Track {track}")
    run_check("feature_types",        validate_feature_types,     feat_after)
    run_check("feature_count",        validate_feature_count,     df_after, track)
    run_check("tld_encoded",          validate_tld_encoded,       feat_before, feat_after)
    run_check("log1p_applied",        validate_log1p_applied,     feat_before, feat_after)
    run_check("engineered_features",  validate_engineered_features, feat_after)
    run_check("binary_unchanged",     validate_binary_unchanged,  feat_before, feat_after)
    # NOTE: outlier clipping is validated at the intermediate pipeline stage
    # (after clipper, before log1p+scaler) in pipeline_builder._validate_clip_stage().
    # Comparing pre-scale thresholds against post-RobustScaler values is invalid.

    n_pass  = sum(1 for c in checks if c["status"] == "PASS")
    n_fail  = sum(1 for c in checks if c["status"] != "PASS")
    all_ok  = n_fail == 0

    logger.info(sep)
    if all_ok:
        logger.info(f"ALL {n_pass} CHECKS PASSED ✓  Track {track.upper()}")
    else:
        logger.error(
            f"{n_fail} CHECK(S) FAILED — "
            f"{n_pass}/{len(checks)} passed  Track {track.upper()}"
        )
    logger.info(sep)

    return {
        "all_passed": all_ok,
        "n_checks"  : len(checks),
        "n_pass"    : n_pass,
        "n_fail"    : n_fail,
        "track"     : track.upper(),
        "results"   : checks,
    }
