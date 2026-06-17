"""
src/data/loader.py
------------------
Loads the PhiUSIIL Phishing URL Dataset from disk and runs a full suite
of structural validation checks before any downstream processing occurs.

Public API
----------
    load_dataset(filepath)          -> pd.DataFrame
    validate_shape(df)              -> bool
    validate_columns(df)            -> (bool, list[str])
    validate_missing_values(df)     -> pd.Series
    validate_target_column(df)      -> dict
    validate_dtypes(df)             -> dict
    run_full_validation(filepath)   -> (pd.DataFrame, dict)
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Dataset constants ─────────────────────────────────────────────────────────

EXPECTED_ROWS: int = 235_795
EXPECTED_COLS: int = 56
TARGET_COLUMN: str = "label"
URL_COLUMN: str = "URL"
VALID_LABEL_VALUES: set = {0, 1}

# Canonical column order as documented in the roadmap
EXPECTED_COLUMNS: list[str] = [
    "FILENAME", "URL", "URLLength", "DomainLength", "TLDLength", "TLD",
    "IsDomainIP", "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio", "IsHTTPS", "LineOfCode",
    "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "Domain", "Title", "label",
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV from *filepath* into a DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Absolute or relative path to ``PhiUSIIL_Phishing_URL_Dataset.csv``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all 56 columns intact.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist on disk.
    ValueError
        If pandas cannot parse the file as CSV.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    size_mb = filepath.stat().st_size / 1_000_000
    logger.info(f"Loading dataset  : {filepath.name}")
    logger.info(f"File size        : {size_mb:.2f} MB")

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        logger.error(f"Failed to parse CSV: {exc}")
        raise ValueError(f"Failed to parse CSV file at {filepath}: {exc}") from exc

    logger.info(f"Loaded           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_shape(df: pd.DataFrame) -> bool:
    """
    Confirm the DataFrame has exactly EXPECTED_ROWS × EXPECTED_COLS.

    Returns
    -------
    bool
        True when column count matches (row mismatch raises only a warning).

    Raises
    ------
    ValueError
        If the column count does not match EXPECTED_COLS.
    """
    rows, cols = df.shape
    logger.info(f"Shape check      : {rows:,} rows × {cols} columns")
    logger.info(f"Expected         : {EXPECTED_ROWS:,} rows × {EXPECTED_COLS} columns")

    if rows != EXPECTED_ROWS:
        logger.warning(
            f"Row count differs from spec ({EXPECTED_ROWS:,}). "
            "This is acceptable if the file was pre-filtered."
        )
    else:
        logger.info("Row count        : PASSED ✓")

    if cols != EXPECTED_COLS:
        logger.error(f"Column count FAILED — expected {EXPECTED_COLS}, got {cols}")
        raise ValueError(
            f"Expected {EXPECTED_COLS} columns, found {cols}. "
            "Check that the correct dataset file is being used."
        )

    logger.info("Column count     : PASSED ✓")
    return True


def validate_columns(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Verify that every expected column is present.

    Returns
    -------
    (is_valid, missing_columns)
    """
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)

    missing = sorted(expected - actual)
    extra   = sorted(actual - expected)

    if extra:
        logger.warning(f"Extra columns not in spec ({len(extra)}): {extra}")
    if missing:
        logger.error(f"Missing columns ({len(missing)}): {missing}")
    if not missing:
        logger.info(f"All {len(EXPECTED_COLUMNS)} columns present : PASSED ✓")

    return (len(missing) == 0), missing


def validate_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Count missing values per column.

    Returns
    -------
    pd.Series
        Per-column missing-value counts (zero for a clean dataset).
    """
    logger.info("Checking missing values …")
    missing = df.isnull().sum()
    total   = int(missing.sum())

    if total == 0:
        logger.info("Missing values   : NONE — dataset is complete ✓")
    else:
        bad_cols = missing[missing > 0]
        logger.warning(f"Found {total:,} missing values in {len(bad_cols)} columns:")
        for col, cnt in bad_cols.items():
            logger.warning(f"  {col}: {cnt:,} ({cnt / len(df) * 100:.2f}%)")

    return missing


def validate_target_column(df: pd.DataFrame) -> dict:
    """
    Confirm the target column contains only 0 and 1, and log class balance.

    Returns
    -------
    dict
        Keys: phishing_count, legitimate_count, phishing_pct,
              legitimate_pct, total, class_ratio.

    Raises
    ------
    ValueError
        If TARGET_COLUMN is absent or contains unexpected values.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    found_vals = set(df[TARGET_COLUMN].unique())
    unexpected = found_vals - VALID_LABEL_VALUES
    if unexpected:
        raise ValueError(
            f"Unexpected values in '{TARGET_COLUMN}': {unexpected}. "
            f"Expected only {{0, 1}}."
        )

    vc = df[TARGET_COLUMN].value_counts().sort_index()
    n  = len(df)

    stats = {
        "phishing_count"   : int(vc.get(0, 0)),
        "legitimate_count" : int(vc.get(1, 0)),
        "phishing_pct"     : round(vc.get(0, 0) / n * 100, 4),
        "legitimate_pct"   : round(vc.get(1, 0) / n * 100, 4),
        "total"            : n,
        "class_ratio"      : round(vc.get(0, 0) / max(vc.get(1, 1), 1), 6),
    }

    logger.info("Target column validation : PASSED ✓")
    logger.info(f"  label=0  Phishing   : {stats['phishing_count']:>10,}  ({stats['phishing_pct']:.2f}%)")
    logger.info(f"  label=1  Legitimate : {stats['legitimate_count']:>10,}  ({stats['legitimate_pct']:.2f}%)")
    logger.info(f"  Class ratio (0/1)   : {stats['class_ratio']:.6f}")

    return stats


def validate_dtypes(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Summarise column data types.

    Returns
    -------
    dict
        Maps dtype string to list of column names.
    """
    dtype_groups: dict[str, list[str]] = {}
    for col in df.columns:
        key = str(df[col].dtype)
        dtype_groups.setdefault(key, []).append(col)

    logger.info("Data type summary:")
    for dtype, cols in sorted(dtype_groups.items()):
        logger.info(f"  {dtype:<12}: {len(cols):>3} columns")

    return dtype_groups


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_full_validation(filepath: str | Path) -> Tuple[pd.DataFrame, dict]:
    """
    Load the dataset and run every validation check in sequence.

    This is the primary entry point for M1.1 / notebook cell execution.

    Parameters
    ----------
    filepath : str | Path
        Path to the raw CSV file.

    Returns
    -------
    (df, validation_report)
        *df* is the raw (unmodified) DataFrame.
        *validation_report* is a dict summarising every check result.
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("MODULE M1.1  —  DATASET VALIDATION")
    logger.info(sep)

    df = load_dataset(filepath)

    report: dict = {}

    # 1. Shape
    validate_shape(df)
    report["rows"]   = df.shape[0]
    report["cols"]   = df.shape[1]

    # 2. Columns
    cols_ok, missing_cols = validate_columns(df)
    report["columns_valid"]   = cols_ok
    report["missing_columns"] = missing_cols

    # 3. Missing values
    missing_series = validate_missing_values(df)
    report["total_missing"]          = int(missing_series.sum())
    report["columns_with_missing"]   = list(
        missing_series[missing_series > 0].index
    )

    # 4. Target column
    class_stats = validate_target_column(df)
    report["class_distribution"] = class_stats

    # 5. Dtypes
    dtype_summary = validate_dtypes(df)
    report["dtype_counts"] = {k: len(v) for k, v in dtype_summary.items()}

    logger.info(sep)
    logger.info("VALIDATION COMPLETE")
    logger.info(sep)

    return df, report
