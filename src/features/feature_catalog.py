"""
src/features/feature_catalog.py
--------------------------------
Single source of truth for every feature-related constant and audit
function used throughout the project.

All downstream modules — preprocessing, model training, SHAP, LIME,
bias analysis, deployment — must import feature lists from here.
Never hard-code column names in any other module.

Public API
----------
    load_config()                        -> dict
    get_feature_lists()                  -> FeatureLists (namedtuple)
    run_feature_audit(df, output_dir)    -> dict
    save_feature_lists(lists, cfg_path)  -> None
"""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy  as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = ROOT / "config" / "feature_config.yaml"

# ── NamedTuple returned by get_feature_lists() ────────────────────────────────

class FeatureLists(NamedTuple):
    """Immutable container for every approved feature list."""
    drop_identifiers    : list[str]
    drop_multicollinear : list[str]
    drop_all            : list[str]
    leakage_critical    : list[str]
    leakage_advisory    : list[str]
    track_A             : list[str]   # 50 features  (includes URLSimilarityIndex)
    track_B             : list[str]   # 49 features  (excludes URLSimilarityIndex)
    target              : str
    categories          : dict[str, list[str]]


# ── Hardcoded constants (mirrors feature_config.yaml exactly) ─────────────────
# These are the authoritative definitions. The YAML is the human-readable form;
# this module is the machine-readable form. They must stay in sync.

TARGET_COLUMN = "label"

DROP_IDENTIFIERS: list[str] = [
    "FILENAME",   # internal file reference — no predictive value
    "URL",        # raw string identifier used only for deduplication
    "Domain",     # raw domain string derived from URL
    "Title",      # free multilingual text — not a structured feature
]

DROP_MULTICOLLINEAR: list[str] = [
    "URLTitleMatchScore",  # r=0.961 with DomainTitleMatchScore — fully redundant
]

DROP_ALL: list[str] = DROP_IDENTIFIERS + DROP_MULTICOLLINEAR

LEAKAGE_CRITICAL: list[str] = [
    "URLSimilarityIndex",  # ALL label=1 = 100.0 exactly — encodes the label
]

LEAKAGE_ADVISORY: list[str] = [
    "IsHTTPS",  # ALL label=1 = 1; 49.2% of phishing also uses HTTPS
]

# ── Feature categories (Track B — 49 features) ───────────────────────────────

FEATURE_CATEGORIES: dict[str, list[str]] = {
    "URL Structure": [
        "URLLength", "DomainLength", "TLDLength", "TLD",
        "IsDomainIP", "NoOfSubDomain", "IsHTTPS",
    ],
    "URL Statistical": [
        "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    ],
    "URL Character Composition": [
        "NoOfLettersInURL", "LetterRatioInURL",
        "NoOfDegitsInURL", "DegitRatioInURL",
        "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    ],
    "Obfuscation": [
        "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    ],
    "HTML Structure": [
        "LineOfCode", "LargestLineLength", "HasTitle",
        "DomainTitleMatchScore", "HasFavicon", "Robots",
        "IsResponsive", "HasDescription",
    ],
    "Redirects & Navigation": [
        "NoOfURLRedirect", "NoOfSelfRedirect", "NoOfPopup", "NoOfiFrame",
    ],
    "Forms & Interaction": [
        "HasExternalFormSubmit", "HasSubmitButton",
        "HasHiddenFields", "HasPasswordField",
    ],
    "Content & Trust": [
        "HasSocialNet", "HasCopyrightInfo", "Bank", "Pay", "Crypto",
    ],
    "External Resources": [
        "NoOfImage", "NoOfCSS", "NoOfJS",
        "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
    ],
}

# Build ordered Track B list from categories dict
TRACK_B_FEATURES: list[str] = [
    f for cat_feats in FEATURE_CATEGORIES.values() for f in cat_feats
]

# Track A adds URLSimilarityIndex after URL Statistical group
TRACK_A_FEATURES: list[str] = (
    FEATURE_CATEGORIES["URL Structure"]
    + FEATURE_CATEGORIES["URL Statistical"]
    + LEAKAGE_CRITICAL                          # URLSimilarityIndex here
    + FEATURE_CATEGORIES["URL Character Composition"]
    + FEATURE_CATEGORIES["Obfuscation"]
    + FEATURE_CATEGORIES["HTML Structure"]
    + FEATURE_CATEGORIES["Redirects & Navigation"]
    + FEATURE_CATEGORIES["Forms & Interaction"]
    + FEATURE_CATEGORIES["Content & Trust"]
    + FEATURE_CATEGORIES["External Resources"]
)

# Validation
assert len(TRACK_B_FEATURES) == 49, f"Track B should have 49 features, got {len(TRACK_B_FEATURES)}"
assert len(TRACK_A_FEATURES) == 50, f"Track A should have 50 features, got {len(TRACK_A_FEATURES)}"
assert "URLSimilarityIndex" in TRACK_A_FEATURES
assert "URLSimilarityIndex" not in TRACK_B_FEATURES
assert len(set(TRACK_B_FEATURES)) == len(TRACK_B_FEATURES), "Duplicate in Track B"
assert len(set(TRACK_A_FEATURES)) == len(TRACK_A_FEATURES), "Duplicate in Track A"

# ── Helper to build a reverse category lookup ─────────────────────────────────

def feature_to_category() -> dict[str, str]:
    """Return a dict mapping every feature name to its category string."""
    mapping = {}
    for cat, feats in FEATURE_CATEGORIES.items():
        for f in feats:
            mapping[f] = cat
    mapping["URLSimilarityIndex"] = "URL Statistical"
    return mapping


# ── Public loaders ────────────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load and return the feature_config.yaml as a dict.

    Returns
    -------
    dict
    """
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found at {CONFIG_PATH}. Using hardcoded constants.")
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.debug(f"Config loaded from {CONFIG_PATH}")
    return cfg


def get_feature_lists() -> FeatureLists:
    """
    Return the single-source-of-truth FeatureLists namedtuple.

    Every module that needs feature lists should call this function.

    Returns
    -------
    FeatureLists
    """
    return FeatureLists(
        drop_identifiers    = DROP_IDENTIFIERS,
        drop_multicollinear = DROP_MULTICOLLINEAR,
        drop_all            = DROP_ALL,
        leakage_critical    = LEAKAGE_CRITICAL,
        leakage_advisory    = LEAKAGE_ADVISORY,
        track_A             = TRACK_A_FEATURES,
        track_B             = TRACK_B_FEATURES,
        target              = TARGET_COLUMN,
        categories          = FEATURE_CATEGORIES,
    )


# ── Audit functions ───────────────────────────────────────────────────────────

def compute_correlation_with_target(df: pd.DataFrame) -> pd.Series:
    """
    Compute |Pearson r| between every numeric feature and the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain TARGET_COLUMN.

    Returns
    -------
    pd.Series  — sorted descending by |r|
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = (
        df[num_cols]
        .corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .abs()
        .sort_values(ascending=False)
    )
    logger.info(f"Correlation-with-target computed for {len(corr)} numeric features")
    return corr


def compute_pairwise_correlations(
    df       : pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Return all feature-feature pairs with |r| >= threshold.

    Parameters
    ----------
    df        : pd.DataFrame (features only, no target)
    threshold : float

    Returns
    -------
    pd.DataFrame  columns: feat_A, feat_B, abs_r
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_mat = df[num_cols].corr().abs()

    upper = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
    )
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_A", "level_1": "feat_B", 0: "abs_r"})
    )
    pairs = pairs[pairs["abs_r"] >= threshold].sort_values("abs_r", ascending=False)
    logger.info(
        f"Pairwise correlations ≥ {threshold}: {len(pairs)} pairs found"
    )
    return pairs.reset_index(drop=True)


def compute_variance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-feature variance statistics to identify near-constant features.

    Parameters
    ----------
    df : pd.DataFrame  (features only, no target)

    Returns
    -------
    pd.DataFrame  with columns: feature, nunique, top_value_pct, dtype, is_near_constant
    """
    rows = []
    for col in df.columns:
        vc           = df[col].value_counts(normalize=True)
        top_pct      = float(vc.iloc[0]) * 100 if len(vc) > 0 else 100.0
        near_const   = top_pct > 95.0
        rows.append({
            "feature"        : col,
            "nunique"        : int(df[col].nunique()),
            "top_value_pct"  : round(top_pct, 2),
            "dtype"          : str(df[col].dtype),
            "is_near_constant": near_const,
        })
    result = pd.DataFrame(rows).sort_values("top_value_pct", ascending=False)
    n_near = result["is_near_constant"].sum()
    logger.info(
        f"Variance audit: {n_near} near-constant features "
        f"(top-value > 95% of samples)"
    )
    return result.reset_index(drop=True)


def compute_skewness(df: pd.DataFrame) -> pd.Series:
    """
    Return absolute skewness for all numeric columns, sorted descending.

    Parameters
    ----------
    df : pd.DataFrame  (features only, no target)

    Returns
    -------
    pd.Series
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skew     = df[num_cols].skew().abs().sort_values(ascending=False)
    n_high   = (skew > 5).sum()
    logger.info(f"Skewness audit: {n_high} features with |skew| > 5")
    return skew


def build_feature_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive per-feature audit table including correlation with
    target, variance stats, skewness, and category assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain TARGET_COLUMN plus the full feature set.

    Returns
    -------
    pd.DataFrame  — one row per feature
    """
    logger.info("Building full feature audit table …")

    feat_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    cat_map = feature_to_category()

    corr_series = compute_correlation_with_target(df)
    var_df      = compute_variance_stats(feat_df)
    skew_series = compute_skewness(feat_df)

    rows = []
    for col in feat_df.columns:
        row = {
            "feature"         : col,
            "category"        : cat_map.get(col, "—"),
            "dtype"           : str(feat_df[col].dtype),
            "nunique"         : int(feat_df[col].nunique()),
            "missing_pct"     : round(float(feat_df[col].isnull().mean()) * 100, 4),
            "top_value_pct"   : float(
                var_df.loc[var_df["feature"] == col, "top_value_pct"].values[0]
                if col in var_df["feature"].values else np.nan
            ),
            "is_near_constant": bool(
                var_df.loc[var_df["feature"] == col, "is_near_constant"].values[0]
                if col in var_df["feature"].values else False
            ),
            "abs_r_with_label": round(float(corr_series.get(col, np.nan)), 6),
            "skewness"        : round(float(skew_series.get(col, np.nan)), 4),
            "in_track_A"      : col in TRACK_A_FEATURES,
            "in_track_B"      : col in TRACK_B_FEATURES,
            "is_leakage"      : col in LEAKAGE_CRITICAL,
            "is_advisory"     : col in LEAKAGE_ADVISORY,
            "is_dropped"      : col in DROP_ALL,
        }

        # Add class-conditional means for numeric features
        if feat_df[col].dtype in [np.int64, np.float64]:
            row["mean_phishing"]   = round(float(df[df[TARGET_COLUMN] == 0][col].mean()), 4)
            row["mean_legitimate"] = round(float(df[df[TARGET_COLUMN] == 1][col].mean()), 4)
        else:
            row["mean_phishing"]   = None
            row["mean_legitimate"] = None

        rows.append(row)

    audit_df = pd.DataFrame(rows)
    logger.info(f"Feature audit table built: {len(audit_df)} rows × {len(audit_df.columns)} columns")
    return audit_df


def run_feature_audit(
    df         : pd.DataFrame,
    output_dir : str | Path = "outputs/reports",
) -> dict:
    """
    Execute the complete M1.2 feature audit and return all results.

    Parameters
    ----------
    df         : clean DataFrame (output of M1.1 cleaning pipeline)
    output_dir : directory for saving intermediate CSV outputs

    Returns
    -------
    dict  with keys:
        audit_table, corr_with_target, pairwise_high, pairwise_medium,
        variance_stats, skewness, feature_lists
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    logger.info(sep)
    logger.info("MODULE M1.2  —  FEATURE AUDIT")
    logger.info(sep)

    # 1. Build full audit table
    audit_table = build_feature_audit_table(df)
    audit_table.to_csv(output_dir / "feature_audit_table.csv", index=False)
    logger.info("Saved: outputs/reports/feature_audit_table.csv")

    # 2. Correlation with target
    corr_target = compute_correlation_with_target(df)

    # 3. Feature-feature correlations
    feat_cols   = [c for c in df.columns if c not in DROP_ALL + [TARGET_COLUMN]]
    feat_df_num = df[feat_cols].select_dtypes(include=[np.number])

    pairs_high   = compute_pairwise_correlations(feat_df_num, threshold=0.75)
    pairs_medium = compute_pairwise_correlations(feat_df_num, threshold=0.50)
    pairs_medium = pairs_medium[pairs_medium["abs_r"] < 0.75]

    pairs_high.to_csv(output_dir / "high_correlation_pairs.csv", index=False)
    pairs_medium.to_csv(output_dir / "medium_correlation_pairs.csv", index=False)
    logger.info(f"High-corr pairs (|r|≥0.75) : {len(pairs_high)}")
    logger.info(f"Medium-corr pairs (0.50–0.75): {len(pairs_medium)}")

    # 4. Variance stats
    var_stats = compute_variance_stats(df[feat_cols])
    var_stats.to_csv(output_dir / "variance_stats.csv", index=False)

    # 5. Skewness
    skewness  = compute_skewness(df[feat_cols])

    # 6. Feature lists
    feat_lists = get_feature_lists()

    # 7. Log summary
    logger.info(sep)
    logger.info("FEATURE AUDIT SUMMARY")
    logger.info(f"  Columns in raw dataset     : 56")
    logger.info(f"  Identifier columns dropped : {len(DROP_IDENTIFIERS)}")
    logger.info(f"  Multicollinear dropped     : {len(DROP_MULTICOLLINEAR)}")
    logger.info(f"  Track A features           : {len(feat_lists.track_A)}")
    logger.info(f"  Track B features           : {len(feat_lists.track_B)}")
    logger.info(f"  Leakage (critical)         : {LEAKAGE_CRITICAL}")
    logger.info(f"  Leakage (advisory)         : {LEAKAGE_ADVISORY}")
    logger.info(f"  High-corr pairs (≥0.75)    : {len(pairs_high)}")
    logger.info(f"  Near-constant features     : {var_stats['is_near_constant'].sum()}")
    logger.info(sep)

    return {
        "audit_table"     : audit_table,
        "corr_with_target": corr_target,
        "pairwise_high"   : pairs_high,
        "pairwise_medium" : pairs_medium,
        "variance_stats"  : var_stats,
        "skewness"        : skewness,
        "feature_lists"   : feat_lists,
    }


def apply_column_removal(
    df   : pd.DataFrame,
    track: str = "B",
) -> pd.DataFrame:
    """
    Return a DataFrame with only the approved columns for a given track.

    Parameters
    ----------
    df    : clean DataFrame (output of M1.1)
    track : "A" or "B"

    Returns
    -------
    pd.DataFrame  with feature columns + target column only
    """
    fl        = get_feature_lists()
    features  = fl.track_A if track.upper() == "A" else fl.track_B
    keep      = features + [TARGET_COLUMN]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from DataFrame: {missing}")

    df_out = df[keep].copy()
    logger.info(
        f"Track {track.upper()}: {len(features)} features + target "
        f"→ {df_out.shape[1]} columns total"
    )
    return df_out
