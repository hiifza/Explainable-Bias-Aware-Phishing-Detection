"""
src/features/encoding.py
------------------------
TLD frequency encoding for the PhiUSIIL dataset.

The encoder fits exclusively on the training set (train-only fit):
  1. Compute TLD value-counts on training data.
  2. Preserve the top-N TLDs by frequency.
  3. Bucket all others as a single 'rare_tld' category.
  4. Assign integer frequency-rank: rank 1 = most common TLD.
  5. Unseen TLDs at transform time receive the rare-TLD rank.

The result is a single numeric column that replaces the original TLD
string column, with no information leakage from the test set.

Public API
----------
    TLDFrequencyEncoder   — sklearn-compatible transformer
    build_tld_report(enc) -> pd.DataFrame
"""

import sys
from pathlib import Path
from typing import Optional

import numpy  as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

TLD_COLUMN  = "TLD"
RARE_LABEL  = "rare_tld"


class TLDFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency-rank encoder for the TLD string column.

    After fitting on the training set, every TLD is mapped to its
    frequency rank (1 = most frequent).  TLDs outside the top-N are
    collapsed to a single *rare_tld* bucket and assigned rank ``top_n + 1``.

    Parameters
    ----------
    top_n       : int   — number of TLDs to preserve individually (default 50)
    rare_label  : str   — internal name for the rare bucket
    col         : str   — name of the TLD column (default "TLD")

    Fitted attributes
    -----------------
    tld_to_rank_   : dict[str, int]  — TLD string → integer rank
    rare_rank_     : int             — rank assigned to out-of-vocabulary TLDs
    top_tld_list_  : list[str]       — ordered list of preserved TLDs
    freq_table_    : pd.Series       — raw training-set TLD frequencies
    """

    def __init__(
        self,
        top_n     : int = 50,
        rare_label: str = RARE_LABEL,
        col       : str = TLD_COLUMN,
    ) -> None:
        self.top_n      = top_n
        self.rare_label = rare_label
        self.col        = col

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None) -> "TLDFrequencyEncoder":
        """
        Compute frequency ranks from the training set.

        Parameters
        ----------
        X : pd.DataFrame  — must contain self.col
        y : ignored

        Returns
        -------
        self
        """
        if self.col not in X.columns:
            raise ValueError(
                f"TLDFrequencyEncoder: column '{self.col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        # Compute training-set TLD frequencies
        self.freq_table_ = X[self.col].value_counts()

        # Preserve top-N by count
        top_tlds = self.freq_table_.head(self.top_n)
        self.top_tld_list_ = top_tlds.index.tolist()

        # Map TLD → rank (1-based, 1 = most frequent)
        self.tld_to_rank_: dict[str, int] = {
            tld: rank + 1 for rank, tld in enumerate(self.top_tld_list_)
        }
        self.rare_rank_ = self.top_n + 1

        n_unique = len(self.freq_table_)
        n_rare   = n_unique - self.top_n
        logger.info(
            f"TLDFrequencyEncoder fitted: "
            f"{n_unique} unique TLDs → top-{self.top_n} preserved, "
            f"{max(n_rare, 0)} bucketed as rare_tld (rank={self.rare_rank_})"
        )
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Replace the TLD string column with integer frequency ranks.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame  — same shape, TLD column now int64
        """
        if not hasattr(self, "tld_to_rank_"):
            raise RuntimeError("TLDFrequencyEncoder has not been fitted yet.")

        X = X.copy()

        n_before   = len(X)
        n_unseen   = int((~X[self.col].isin(self.tld_to_rank_)).sum())

        X[self.col] = (
            X[self.col]
            .map(self.tld_to_rank_)
            .fillna(self.rare_rank_)          # unseen TLDs → rare rank
            .astype(np.int64)
        )

        if n_unseen > 0:
            logger.debug(
                f"TLDFrequencyEncoder: {n_unseen}/{n_before} rows had "
                f"unseen TLDs → assigned rare rank {self.rare_rank_}"
            )

        return X

    # ── fit_transform ─────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ── Report helper ─────────────────────────────────────────────────────────────

def build_tld_encoding_report(enc: TLDFrequencyEncoder) -> pd.DataFrame:
    """
    Build a human-readable report of the fitted TLD encoding.

    Parameters
    ----------
    enc : fitted TLDFrequencyEncoder

    Returns
    -------
    pd.DataFrame  columns: tld, frequency, rank, bucket
    """
    if not hasattr(enc, "tld_to_rank_"):
        raise RuntimeError("Encoder has not been fitted.")

    rows = []
    for tld, rank in enc.tld_to_rank_.items():
        freq = int(enc.freq_table_.get(tld, 0))
        rows.append({
            "tld"      : tld,
            "frequency": freq,
            "rank"     : rank,
            "bucket"   : "top_tld",
        })

    # Add rare bucket summary
    rare_tlds   = enc.freq_table_.iloc[enc.top_n:]
    rare_total  = int(rare_tlds.sum())
    rows.append({
        "tld"      : enc.rare_label,
        "frequency": rare_total,
        "rank"     : enc.rare_rank_,
        "bucket"   : "rare_tld",
    })

    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    logger.info(
        f"TLD encoding report: {len(df)-1} top TLDs + 1 rare bucket"
    )
    return df
