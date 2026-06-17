"""
src/features/engineering.py
----------------------------
Adds 7 engineered features to the feature matrix as the first step
of the preprocessing pipeline.

Feature engineering must come BEFORE TLD encoding so that
TLD_is_gov_edu can use the original TLD string.

Engineered Features
-------------------
1.  ContentComplexityScore  = log1p(NoOfCSS + NoOfJS + NoOfImage)
2.  FormDangerIndex         = HasExternalFormSubmit + HasHiddenFields + HasPasswordField
3.  TrustBadgeScore         = HasFavicon + Robots + HasCopyrightInfo + HasSocialNet
4.  RedirectActivity        = NoOfURLRedirect + NoOfSelfRedirect
5.  ExternalRefDensity      = NoOfExternalRef / (NoOfSelfRef + 1)
6.  TLD_is_gov_edu          = 1 if TLD in {gov, edu, mil} else 0
7.  SubdomainRatio          = NoOfSubDomain / (DomainLength + 1)

These 7 features bring:
  Track B base (49) + 7 = 56 features
  Track A base (50) + 7 = 57 features

Public API
----------
    FeatureEngineer           — sklearn-compatible transformer
    ENGINEERED_FEATURE_NAMES  — ordered list of new column names
    GOV_EDU_TLDS              — set of TLD strings considered trusted
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


# ── Constants ─────────────────────────────────────────────────────────────────

# TLDs considered trusted (government / education)
GOV_EDU_TLDS: frozenset[str] = frozenset({
    ".gov", ".edu", ".mil",   # with dot prefix
    "gov",  "edu",  "mil",    # without dot prefix (handles both formats)
})

# Ordered list of engineered feature names
ENGINEERED_FEATURE_NAMES: list[str] = [
    "ContentComplexityScore",
    "FormDangerIndex",
    "TrustBadgeScore",
    "RedirectActivity",
    "ExternalRefDensity",
    "TLD_is_gov_edu",
    "SubdomainRatio",
]


# ── FeatureEngineer transformer ───────────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds 7 engineered columns to the input DataFrame.

    This transformer has no trainable parameters — fit() is a no-op.
    All transformations are deterministic functions of the input columns.

    Must be the FIRST step in any pipeline because TLD_is_gov_edu
    requires TLD as a string, before TLDFrequencyEncoder converts it
    to an integer.

    Parameters
    ----------
    tld_col : str  — name of the TLD column (default "TLD")
    """

    def __init__(self, tld_col: str = "TLD") -> None:
        self.tld_col = tld_col

    # ── fit: no-op ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """
        No parameters to learn.  Records which source columns are present.
        """
        self._source_cols_present_ = list(X.columns)
        logger.info(
            f"FeatureEngineer fitted (no-op): "
            f"input has {len(X.columns)} columns"
        )
        return self

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Compute and append 7 engineered features.

        Parameters
        ----------
        X : pd.DataFrame  — feature matrix with original columns

        Returns
        -------
        pd.DataFrame  — original columns + 7 new columns appended
        """
        if not hasattr(self, "_source_cols_present_"):
            raise RuntimeError("FeatureEngineer has not been fitted yet.")

        X = X.copy()

        # ── 1. ContentComplexityScore ─────────────────────────────────────────
        # log1p of the sum of CSS, JS, and image counts.
        # Captures overall page resource richness on a compressed scale.
        css  = X.get("NoOfCSS",   pd.Series(0, index=X.index))
        js   = X.get("NoOfJS",    pd.Series(0, index=X.index))
        img  = X.get("NoOfImage", pd.Series(0, index=X.index))

        X["ContentComplexityScore"] = np.log1p(
            (css + js + img).clip(lower=0)
        ).astype(np.float64)

        # ── 2. FormDangerIndex ────────────────────────────────────────────────
        # Sum of three form-danger binary indicators (range 0–3).
        # Higher = more dangerous form configuration.
        efs  = X.get("HasExternalFormSubmit", pd.Series(0, index=X.index))
        hf   = X.get("HasHiddenFields",       pd.Series(0, index=X.index))
        hpf  = X.get("HasPasswordField",      pd.Series(0, index=X.index))

        X["FormDangerIndex"] = (efs + hf + hpf).astype(np.int8)

        # ── 3. TrustBadgeScore ────────────────────────────────────────────────
        # Sum of four trust-signal binary indicators (range 0–4).
        # Higher = more trust signals present (more likely legitimate).
        fav  = X.get("HasFavicon",      pd.Series(0, index=X.index))
        rob  = X.get("Robots",          pd.Series(0, index=X.index))
        copy = X.get("HasCopyrightInfo",pd.Series(0, index=X.index))
        soc  = X.get("HasSocialNet",    pd.Series(0, index=X.index))

        X["TrustBadgeScore"] = (fav + rob + copy + soc).astype(np.int8)

        # ── 4. RedirectActivity ───────────────────────────────────────────────
        # Sum of the two redirect binary flags (range 0–2).
        redir_url  = X.get("NoOfURLRedirect",  pd.Series(0, index=X.index))
        redir_self = X.get("NoOfSelfRedirect", pd.Series(0, index=X.index))

        X["RedirectActivity"] = (redir_url + redir_self).astype(np.int8)

        # ── 5. ExternalRefDensity ─────────────────────────────────────────────
        # Ratio of external to self references; +1 in denominator avoids /0.
        # High density = more reliance on external resources.
        ext_ref  = X.get("NoOfExternalRef", pd.Series(0, index=X.index))
        self_ref = X.get("NoOfSelfRef",     pd.Series(0, index=X.index))

        X["ExternalRefDensity"] = (
            ext_ref / (self_ref.clip(lower=0) + 1)
        ).astype(np.float64)

        # ── 6. TLD_is_gov_edu ─────────────────────────────────────────────────
        # Binary flag: 1 if the TLD belongs to a trusted institutional
        # namespace (government / education / military), else 0.
        # Must be computed BEFORE TLDFrequencyEncoder replaces the string.
        if self.tld_col in X.columns:
            X["TLD_is_gov_edu"] = (
                X[self.tld_col].isin(GOV_EDU_TLDS)
            ).astype(np.int8)
        else:
            logger.warning(
                f"TLD column '{self.tld_col}' not found — "
                "TLD_is_gov_edu set to 0 for all rows"
            )
            X["TLD_is_gov_edu"] = np.int8(0)

        # ── 7. SubdomainRatio ─────────────────────────────────────────────────
        # Subdomain count normalised by domain length.
        # Captures depth of subdomain nesting relative to domain complexity.
        n_sub  = X.get("NoOfSubDomain", pd.Series(0, index=X.index))
        d_len  = X.get("DomainLength",  pd.Series(1, index=X.index))

        X["SubdomainRatio"] = (
            n_sub / (d_len.clip(lower=0) + 1)
        ).astype(np.float64)

        logger.debug(
            f"FeatureEngineer: added {len(ENGINEERED_FEATURE_NAMES)} features "
            f"→ output shape {X.shape}"
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **kw) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ── Validation helper ─────────────────────────────────────────────────────

    def validate_output(self, X_out: pd.DataFrame) -> None:
        """
        Assert that all engineered features are present and have no NaN.

        Parameters
        ----------
        X_out : DataFrame returned by transform()

        Raises
        ------
        AssertionError if any check fails
        """
        for feat in ENGINEERED_FEATURE_NAMES:
            assert feat in X_out.columns, \
                f"Engineered feature '{feat}' missing from output"
            n_null = X_out[feat].isnull().sum()
            assert n_null == 0, \
                f"Engineered feature '{feat}' has {n_null} NaN values"

        # Range checks
        assert X_out["FormDangerIndex"].between(0, 3).all(), \
            "FormDangerIndex out of range [0, 3]"
        assert X_out["TrustBadgeScore"].between(0, 4).all(), \
            "TrustBadgeScore out of range [0, 4]"
        assert X_out["RedirectActivity"].between(0, 2).all(), \
            "RedirectActivity out of range [0, 2]"
        assert X_out["TLD_is_gov_edu"].isin([0, 1]).all(), \
            "TLD_is_gov_edu contains values other than 0 and 1"
        assert (X_out["ContentComplexityScore"] >= 0).all(), \
            "ContentComplexityScore has negative values"
        assert (X_out["ExternalRefDensity"] >= 0).all(), \
            "ExternalRefDensity has negative values"

        logger.info("FeatureEngineer output validation PASSED ✓")
