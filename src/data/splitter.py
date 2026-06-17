"""
src/data/splitter.py
--------------------
Stratified 80/20 train / test split for the PhiUSIIL dataset.

The split is computed ONCE and the resulting index arrays are shared
between Track A and Track B so that both tracks are evaluated on
identical test rows — an essential requirement for fair comparison.

Public API
----------
    stratified_split(df, test_size, random_state) -> (train_idx, test_idx)
    apply_split(df, train_idx, test_idx)           -> (df_train, df_test)
    save_split(df_train, df_test, output_dir, prefix)
    run_split_pipeline(df, track_features, output_dir) -> dict
"""

import sys
from pathlib import Path
from typing import Optional

import numpy  as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "label"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20


# ── Core split logic ──────────────────────────────────────────────────────────

def stratified_split(
    df          : pd.DataFrame,
    test_size   : float = TEST_SIZE,
    random_state: int   = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute stratified train/test indices preserving class proportions.

    Parameters
    ----------
    df           : full DataFrame containing TARGET_COLUMN
    test_size    : fraction of data for the test set (default 0.20)
    random_state : reproducibility seed (default 42)

    Returns
    -------
    (train_idx, test_idx) — 1-D integer numpy arrays of row positions
    """
    sss = StratifiedShuffleSplit(
        n_splits     = 1,
        test_size    = test_size,
        random_state = random_state,
    )
    y = df[TARGET_COLUMN].values
    train_idx, test_idx = next(sss.split(df, y))

    n_train = len(train_idx)
    n_test  = len(test_idx)
    total   = len(df)

    # Validate class balance preserved
    y_train = y[train_idx]
    y_test  = y[test_idx]

    train_ph_pct = (y_train == 0).mean() * 100
    test_ph_pct  = (y_test  == 0).mean() * 100
    original_pct = (y       == 0).mean() * 100

    logger.info(f"Stratified split: {n_train:,} train / {n_test:,} test "
                f"(test_size={test_size}, seed={random_state})")
    logger.info(f"  Phishing % — original: {original_pct:.2f}%  "
                f"train: {train_ph_pct:.2f}%  test: {test_ph_pct:.2f}%")

    # Assert balance is preserved within 0.5%
    assert abs(train_ph_pct - original_pct) < 0.5, \
        f"Train class balance drifted: {train_ph_pct:.2f}% vs {original_pct:.2f}%"
    assert abs(test_ph_pct  - original_pct) < 0.5, \
        f"Test class balance drifted:  {test_ph_pct:.2f}% vs {original_pct:.2f}%"

    logger.info("Class balance PASSED ✓")
    return train_idx, test_idx


def apply_split(
    df        : pd.DataFrame,
    train_idx : np.ndarray,
    test_idx  : np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice df using pre-computed index arrays.

    Parameters
    ----------
    df        : DataFrame to split
    train_idx : row positions for training set
    test_idx  : row positions for test set

    Returns
    -------
    (df_train, df_test)  — reset-indexed copies
    """
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx ].reset_index(drop=True)
    logger.debug(
        f"Split applied: train={len(df_train):,}  test={len(df_test):,}"
    )
    return df_train, df_test


def save_split(
    df_train  : pd.DataFrame,
    df_test   : pd.DataFrame,
    output_dir: str | Path,
    prefix    : str = "",
) -> tuple[Path, Path]:
    """
    Save train and test DataFrames to CSV files.

    Parameters
    ----------
    df_train   : training-set DataFrame
    df_test    : test-set DataFrame
    output_dir : destination directory
    prefix     : optional filename prefix (e.g. "raw_" or "preprocessed_")

    Returns
    -------
    (train_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{prefix}X_train.csv"
    test_path  = output_dir / f"{prefix}X_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv( test_path,  index=False)

    logger.info(
        f"Saved → {train_path.name}  ({len(df_train):,} rows)  |  "
        f"{test_path.name}  ({len(df_test):,} rows)"
    )
    return train_path, test_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_split_pipeline(
    df             : pd.DataFrame,
    track_A_feats  : list[str],
    track_B_feats  : list[str],
    output_dir     : str | Path = "data/processed",
    test_size      : float      = TEST_SIZE,
    random_state   : int        = RANDOM_STATE,
) -> dict:
    """
    Compute a single stratified split and save raw (pre-pipeline) splits
    for both Track A and Track B.

    The same train/test indices are used for both tracks.

    Parameters
    ----------
    df             : full clean DataFrame (from M1.1)
    track_A_feats  : Track A feature list (50 features)
    track_B_feats  : Track B feature list (49 features)
    output_dir     : base directory; sub-dirs track_A/ and track_B/ created
    test_size      : test fraction
    random_state   : reproducibility seed

    Returns
    -------
    dict  keys: train_idx, test_idx, n_train, n_test,
                track_A_paths, track_B_paths, y_train_path, y_test_path
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info("M3.1 — STRATIFIED TRAIN/TEST SPLIT")
    logger.info(sep)

    output_dir = Path(output_dir)

    # 1. Compute shared split indices
    train_idx, test_idx = stratified_split(df, test_size, random_state)

    # 2. Save label vectors (shared)
    y_train = df.iloc[train_idx][[TARGET_COLUMN]].reset_index(drop=True)
    y_test  = df.iloc[test_idx ][[TARGET_COLUMN]].reset_index(drop=True)

    y_dir        = output_dir
    y_dir.mkdir(parents=True, exist_ok=True)
    y_train_path = y_dir / "y_train.csv"
    y_test_path  = y_dir / "y_test.csv"
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv( y_test_path,  index=False)
    logger.info(f"Saved y_train.csv ({len(y_train):,})  y_test.csv ({len(y_test):,})")

    # 3. Save raw (pre-pipeline) Track A split
    keep_A         = [c for c in track_A_feats if c in df.columns]
    df_A           = df[keep_A + [TARGET_COLUMN]]
    df_A_train, df_A_test = apply_split(df_A, train_idx, test_idx)
    track_A_paths  = save_split(df_A_train, df_A_test,
                                output_dir / "track_A", prefix="raw_")

    # 4. Save raw Track B split
    keep_B         = [c for c in track_B_feats if c in df.columns]
    df_B           = df[keep_B + [TARGET_COLUMN]]
    df_B_train, df_B_test = apply_split(df_B, train_idx, test_idx)
    track_B_paths  = save_split(df_B_train, df_B_test,
                                output_dir / "track_B", prefix="raw_")

    result = {
        "train_idx"      : train_idx,
        "test_idx"       : test_idx,
        "n_train"        : len(train_idx),
        "n_test"         : len(test_idx),
        "track_A_train"  : df_A_train,
        "track_A_test"   : df_A_test,
        "track_B_train"  : df_B_train,
        "track_B_test"   : df_B_test,
        "y_train"        : y_train,
        "y_test"         : y_test,
        "track_A_paths"  : track_A_paths,
        "track_B_paths"  : track_B_paths,
        "y_train_path"   : y_train_path,
        "y_test_path"    : y_test_path,
    }

    logger.info(sep)
    logger.info(f"SPLIT COMPLETE — train: {len(train_idx):,}  test: {len(test_idx):,}")
    logger.info(sep)
    return result
