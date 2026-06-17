"""
src/features/preprocessing.py
------------------------------
Three sklearn-compatible transformers for the PhiUSIIL preprocessing
pipeline.  All transformers accept and return pandas DataFrames to
preserve column names throughout the pipeline.

Classes
-------
    OutlierClipper        — caps values at per-feature P99.9 quantile
    Log1pTransformer      — applies numpy.log1p to specified count features
    RobustScalerTransformer — RobustScaler on continuous/non-binary features

Constants
---------
    LOG1P_FEATURES   — columns that receive log1p transform (approved in M2.1)
    BINARY_FEATURES  — strictly 0/1 columns that are never scaled
"""

import sys
from pathlib import Path
from typing import Optional

import numpy  as np
import pandas as pd
from sklearn.base       import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Approved feature lists ────────────────────────────────────────────────────

# Features receiving log1p transform (high |skew| confirmed in M2.1)
LOG1P_FEATURES: list[str] = [
    "URLLength",
    "DomainLength",
    "LineOfCode",
    "LargestLineLength",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfOtherSpecialCharsInURL",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
    "NoOfPopup",
    "NoOfiFrame",
]

# Strictly binary (0/1) features — excluded from RobustScaler
BINARY_FEATURES: list[str] = [
    "IsDomainIP",
    "HasObfuscation",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",     # only 0/1 despite "NoOf" name
    "NoOfSelfRedirect",    # only 0/1 despite "NoOf" name
    "HasDescription",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    # Engineered binary
    "TLD_is_gov_edu",
]


# ── OutlierClipper ────────────────────────────────────────────────────────────

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips each numeric column at a per-feature quantile threshold.

    Thresholds are computed from the training set only (train-only fit).
    At transform time any value exceeding the stored threshold is capped.

    Parameters
    ----------
    quantile : float — upper clipping quantile (default 0.999 = P99.9)

    Fitted attributes
    -----------------
    thresholds_ : dict[str, float]  — per-column upper bounds
    """

    def __init__(self, quantile: float = 0.999) -> None:
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierClipper":
        """Compute per-column clipping thresholds from training data."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.thresholds_: dict[str, float] = {}
        for col in num_cols:
            thresh = float(X[col].quantile(self.quantile))
            self.thresholds_[col] = thresh

        n_cols = len(self.thresholds_)
        logger.info(
            f"OutlierClipper fitted: P{self.quantile*100:.1f} "
            f"thresholds computed for {n_cols} numeric columns"
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Clip each column at its fitted threshold."""
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("OutlierClipper has not been fitted yet.")

        X = X.copy()
        for col, thresh in self.thresholds_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=thresh)
        return X

    def get_thresholds_df(self) -> pd.DataFrame:
        """Return a DataFrame of column → threshold for reporting."""
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("OutlierClipper has not been fitted yet.")
        rows = [{"feature": col, "p999_threshold": thresh}
                for col, thresh in self.thresholds_.items()]
        return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


# ── Log1pTransformer ──────────────────────────────────────────────────────────

class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Applies numpy.log1p to each column in *features* that exists in X.

    log1p(x) = log(1 + x) — handles zero values gracefully and compresses
    the right tail of highly-skewed count distributions.

    Parameters
    ----------
    features : list[str] | None
        Columns to transform.  Defaults to LOG1P_FEATURES constant.
        Columns not present in X are silently skipped.

    Fitted attributes
    -----------------
    features_present_ : list[str]  — subset of features that exist in X
    skewness_before_  : dict[str, float]  — pre-transform skewness
    skewness_after_   : dict[str, float]  — post-transform skewness
    """

    def __init__(self, features: Optional[list[str]] = None) -> None:
        self.features = features

    def fit(self, X: pd.DataFrame, y=None) -> "Log1pTransformer":
        """Record which features are present and compute before-skewness."""
        feat_list = self.features if self.features is not None else LOG1P_FEATURES
        self.features_present_ = [f for f in feat_list if f in X.columns]
        self.skewness_before_  = {
            f: float(X[f].skew()) for f in self.features_present_
        }
        logger.info(
            f"Log1pTransformer fitted: "
            f"{len(self.features_present_)}/{len(feat_list)} features present"
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply log1p to all fitted features present in X."""
        if not hasattr(self, "features_present_"):
            raise RuntimeError("Log1pTransformer has not been fitted yet.")

        X = X.copy()
        for feat in self.features_present_:
            if feat in X.columns:
                # Clip to 0 first — log1p is undefined for negative values
                X[feat] = np.log1p(X[feat].clip(lower=0))

        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **kw) -> pd.DataFrame:
        self.fit(X, y)
        X_t = self.transform(X)
        # Compute after-skewness for reporting
        self.skewness_after_ = {
            f: float(X_t[f].skew()) for f in self.features_present_
        }
        return X_t

    def get_skewness_report(self) -> pd.DataFrame:
        """
        Return before/after skewness comparison for all transformed features.
        Only available after fit_transform() has been called.
        """
        if not hasattr(self, "skewness_before_"):
            raise RuntimeError("Log1pTransformer has not been fitted yet.")

        rows = []
        for feat in self.features_present_:
            before = self.skewness_before_.get(feat, float("nan"))
            after  = self.skewness_after_.get(feat, float("nan")) \
                if hasattr(self, "skewness_after_") else float("nan")
            rows.append({
                "feature"       : feat,
                "skewness_before": round(before, 4),
                "skewness_after" : round(after, 4),
                "reduction_pct"  : round(
                    (1 - abs(after) / max(abs(before), 1e-9)) * 100, 2
                ) if not np.isnan(before) and not np.isnan(after) else float("nan"),
            })
        return pd.DataFrame(rows)


# ── RobustScalerTransformer ───────────────────────────────────────────────────

class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps sklearn.preprocessing.RobustScaler to work with pandas DataFrames
    while leaving binary and categorical columns unchanged.

    Binary features listed in BINARY_FEATURES are excluded from scaling.
    After transformation all column names are preserved.

    Parameters
    ----------
    binary_features : list[str] | None
        Columns to exclude from scaling.  Defaults to BINARY_FEATURES constant.

    Fitted attributes
    -----------------
    scale_cols_  : list[str]    — columns that were actually scaled
    scaler_      : RobustScaler — fitted sklearn scaler
    center_      : np.ndarray   — per-column medians
    scale_       : np.ndarray   — per-column IQR values
    """

    def __init__(
        self,
        binary_features: Optional[list[str]] = None,
    ) -> None:
        self.binary_features = binary_features

    def _get_binary_set(self) -> set[str]:
        bf = self.binary_features
        return set(bf) if bf is not None else set(BINARY_FEATURES)

    def fit(self, X: pd.DataFrame, y=None) -> "RobustScalerTransformer":
        """
        Identify scalable columns and fit RobustScaler on them.
        All fitting is done on the training set only.
        """
        binary_set = self._get_binary_set()

        # Scale all numeric columns that are NOT binary
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scale_cols_ = [
            c for c in num_cols
            if c not in binary_set and X[c].nunique() > 2
        ]

        self.scaler_ = RobustScaler(unit_variance=False)
        if self.scale_cols_:
            self.scaler_.fit(X[self.scale_cols_].values)
            self.center_ = self.scaler_.center_
            self.scale_  = self.scaler_.scale_
        else:
            self.center_ = np.array([])
            self.scale_  = np.array([])

        logger.info(
            f"RobustScalerTransformer fitted: "
            f"{len(self.scale_cols_)} columns scaled, "
            f"{len(binary_set)} binary columns unchanged"
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply fitted RobustScaler to scalable columns."""
        if not hasattr(self, "scaler_"):
            raise RuntimeError("RobustScalerTransformer has not been fitted yet.")

        X = X.copy()
        if self.scale_cols_:
            X[self.scale_cols_] = self.scaler_.transform(
                X[self.scale_cols_].values
            )
        return X

    def get_scaling_report(self) -> pd.DataFrame:
        """Return per-column center and scale values."""
        if not hasattr(self, "scale_cols_"):
            raise RuntimeError("RobustScalerTransformer has not been fitted yet.")

        rows = [
            {
                "feature"    : col,
                "center_median": round(float(self.center_[i]), 6),
                "scale_iqr"  : round(float(self.scale_[i]),  6),
                "scaled"     : True,
            }
            for i, col in enumerate(self.scale_cols_)
        ]

        # Add unscaled binary columns for completeness
        binary_set = self._get_binary_set()
        for col in binary_set:
            rows.append({
                "feature"     : col,
                "center_median": 0.0,
                "scale_iqr"   : 1.0,
                "scaled"      : False,
            })

        return (
            pd.DataFrame(rows)
            .sort_values(["scaled", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
