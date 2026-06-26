"""
src/blindspots/failure_archetype_discovery.py
----------------------------------------------
Automatically discovers recurring failure archetypes from the uncertainty
zone using unsupervised clustering.  Categories are never hardcoded —
they emerge from the data.

Algorithm
---------
1. Extract feature vectors for all failure-zone samples.
2. Standardise features; apply PCA to 10 components.
3. Run K-Means (k auto-selected via silhouette score, k∈[2,8]).
4. Assign a descriptive label to each cluster based on dominant features.
5. Return archetype DataFrame and cluster metadata.

Public API
----------
    ArchetypeResult        — dataclass with cluster labels + metadata
    discover_archetypes(fcs, X_test, feature_names, shap_values) -> ArchetypeResult
"""

import sys
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

import numpy  as np
import pandas as pd
from sklearn.cluster          import KMeans
from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.blindspots.failure_case_extractor import FailureCaseSet

logger = get_logger(__name__)


@dataclass
class ArchetypeResult:
    """Clustering output for failure archetype discovery."""
    labels          : np.ndarray        # cluster label per failure case
    n_clusters      : int
    silhouette_score: float
    cluster_meta    : list[dict]        # per-cluster description dicts
    X_pca           : np.ndarray        # PCA-reduced features (for plotting)
    sample_indices  : np.ndarray        # original test-set indices
    archetype_df    : pd.DataFrame      # full per-sample DataFrame
    explained_var   : float             # PCA explained variance ratio (sum)


def _auto_k(X: np.ndarray, k_range: range) -> int:
    """Return k that maximises silhouette score."""
    best_k, best_s = k_range[0], -1.0
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbl = km.fit_predict(X)
        s   = float(silhouette_score(X, lbl)) if len(set(lbl)) > 1 else -1.0
        if s > best_s:
            best_s, best_k = s, k
    logger.info(f"Auto-k: best k={best_k}  silhouette={best_s:.4f}")
    return best_k


def _cluster_label(
    cluster_idx    : int,
    cluster_members: pd.DataFrame,
    feature_names  : list[str],
    shap_values    : Optional[np.ndarray],
    sample_indices : np.ndarray,
) -> str:
    """
    Auto-generate a descriptive archetype label.

    Uses dominant feature values and mean SHAP attribution to name the cluster.
    """
    if cluster_members.empty:
        return f"Cluster {cluster_idx}"

    # Dominant features: those farthest from the overall mean
    feat_cols = [c for c in cluster_members.columns
                 if c in feature_names and cluster_members[c].std() >= 0]
    if not feat_cols:
        return f"Archetype {cluster_idx+1}"

    # Feature means within cluster
    means = cluster_members[feat_cols].mean()

    # Map high-value features to descriptive labels
    desc_parts = []
    feature_hints = {
        "IsHTTPS"           : ("HTTPS", "Non-HTTPS"),
        "URLLength"         : ("Long-URL", None),
        "TLD_is_gov_edu"    : ("Gov/Edu domain", None),
        "HasFavicon"        : ("Favicon-present", None),
        "HasSocialNet"      : ("Social-linked", None),
        "HasPasswordField"  : ("Password-form", None),
        "HasExternalFormSubmit": ("Ext-form-submit", None),
        "NoOfExternalRef"   : ("High-ext-refs", None),
        "IsDomainIP"        : ("IP-based-domain", None),
        "HasObfuscation"    : ("Obfuscated-URL", None),
        "HasCopyrightInfo"  : ("Copyright-present", None),
        "ContentComplexityScore": ("Complex-content", None),
        "TrustBadgeScore"   : ("High-trust-signals", None),
        "FormDangerIndex"   : ("Dangerous-form", None),
        "LetterRatioInURL"  : ("Alpha-heavy-URL", None),
        "NoOfDegitsInURL"   : ("Digit-heavy-URL", None),
    }

    for feat, (hi_label, lo_label) in feature_hints.items():
        if feat not in means.index:
            continue
        v    = means[feat]
        all_v= cluster_members[feat].dropna() if feat in cluster_members.columns else pd.Series([])
        if all_v.empty:
            continue
        p75 = all_v.quantile(0.75)
        p25 = all_v.quantile(0.25)
        if v >= p75 and hi_label:
            desc_parts.append(hi_label)
        elif v <= p25 and lo_label:
            desc_parts.append(lo_label)
        if len(desc_parts) >= 3:
            break

    if not desc_parts:
        # Fall back to top-SHAP feature name if SHAP is available
        if shap_values is not None and len(sample_indices) > 0:
            valid_shap_idx = [i for i in sample_indices if i < len(shap_values)]
            if valid_shap_idx:
                sv   = shap_values[valid_shap_idx]
                top_f= int(np.abs(sv).mean(axis=0).argmax())
                desc_parts.append(f"Feature:{feature_names[top_f]}")

    label = "+".join(desc_parts[:3]) if desc_parts else f"Archetype {cluster_idx+1}"
    return label


def discover_archetypes(
    fcs          : FailureCaseSet,
    X_test       : pd.DataFrame,
    feature_names: list[str],
    shap_values  : Optional[np.ndarray] = None,
    n_pca_components: int = 10,
    k_range      : tuple = (2, 8),
    random_state : int   = 42,
) -> ArchetypeResult:
    """
    Discover failure archetypes via PCA + K-Means clustering.

    Parameters
    ----------
    fcs           : FailureCaseSet from extractor
    X_test        : full test feature DataFrame
    feature_names : column names
    shap_values   : optional SHAP array for richer labelling
    n_pca_components : PCA dimensionality
    k_range       : (min_k, max_k) for silhouette search
    random_state  : seed

    Returns
    -------
    ArchetypeResult
    """
    logger.info(f"Discovering archetypes from {len(fcs):,} failure cases …")

    if len(fcs) < 3:
        logger.warning("Too few failure cases for clustering — returning trivial result")
        dummy_df = pd.DataFrame({"sample_idx": [c.sample_idx for c in fcs],
                                  "archetype": ["single_cluster"] * len(fcs)})
        return ArchetypeResult(
            labels=np.zeros(len(fcs), dtype=int), n_clusters=1,
            silhouette_score=0.0, cluster_meta=[],
            X_pca=np.zeros((len(fcs), 2)),
            sample_indices=np.array([c.sample_idx for c in fcs]),
            archetype_df=dummy_df, explained_var=0.0,
        )

    # ── Feature matrix ────────────────────────────────────────────────────────
    indices    = np.array([c.sample_idx for c in fcs])
    valid_mask = indices < len(X_test)
    indices    = indices[valid_mask]
    X_fail     = X_test.iloc[indices][feature_names].fillna(0).values.astype(np.float64)

    if len(X_fail) < 3:
        logger.warning("Insufficient aligned failure cases after masking")
        indices = np.array([c.sample_idx for c in fcs])
        X_pca_2 = np.random.default_rng(random_state).standard_normal((len(fcs), 2))
        dummy_df = pd.DataFrame({"sample_idx": indices.tolist(),
                                  "archetype": ["archetype_0"]*len(fcs),
                                  "cluster": [0]*len(fcs)})
        return ArchetypeResult(
            labels=np.zeros(len(fcs),dtype=int), n_clusters=1,
            silhouette_score=0.0, cluster_meta=[],
            X_pca=X_pca_2, sample_indices=indices,
            archetype_df=dummy_df, explained_var=0.0,
        )

    # ── Standardise ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fail)

    # ── PCA ───────────────────────────────────────────────────────────────────
    n_comp    = min(n_pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca       = PCA(n_components=n_comp, random_state=random_state)
    X_pca     = pca.fit_transform(X_scaled)
    exp_var   = float(pca.explained_variance_ratio_.sum())

    # ── Auto-select k ─────────────────────────────────────────────────────────
    k_min = k_range[0]
    k_max = min(k_range[1], len(X_pca) - 1)
    if k_min >= k_max:
        best_k = max(2, k_min)
    else:
        best_k = _auto_k(X_pca, range(k_min, k_max + 1))

    # ── K-Means ───────────────────────────────────────────────────────────────
    km     = KMeans(n_clusters=best_k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X_pca)

    sil = float(silhouette_score(X_pca, labels)) if len(set(labels)) > 1 else 0.0

    # ── Per-cluster metadata ──────────────────────────────────────────────────
    X_fail_df = pd.DataFrame(X_fail, columns=feature_names, index=indices)
    cluster_meta = []
    for k in range(best_k):
        mask    = labels == k
        c_idx   = indices[mask]
        c_df    = X_fail_df.loc[c_idx] if len(c_idx) > 0 else pd.DataFrame()
        label   = _cluster_label(k, c_df, feature_names, shap_values, c_idx)
        n_err   = sum(1 for c in fcs if c.sample_idx in set(c_idx.tolist()) and c.is_error)
        mean_conf= np.mean([c.confidence for c in fcs
                             if c.sample_idx in set(c_idx.tolist())]) if len(c_idx) > 0 else 0.0
        cluster_meta.append({
            "cluster"      : k,
            "label"        : label,
            "n_samples"    : int(mask.sum()),
            "n_errors"     : n_err,
            "mean_confidence": round(float(mean_conf), 4),
            "pca_centroid_x": float(km.cluster_centers_[k, 0]),
            "pca_centroid_y": float(km.cluster_centers_[k, 1]),
        })

    # ── Archetype DataFrame ───────────────────────────────────────────────────
    # Map label to each failure case by sample_idx
    idx_to_cluster = dict(zip(indices.tolist(), labels.tolist()))
    idx_to_label   = {k: m["label"] for m in cluster_meta for k in
                      indices[labels == m["cluster"]].tolist()}

    arch_rows = []
    for i, c in enumerate(fcs):
        sidx = c.sample_idx
        clust = idx_to_cluster.get(sidx, -1)
        alabel= idx_to_label.get(sidx, "unassigned")
        arch_rows.append({
            "sample_idx"      : sidx,
            "y_true"          : c.y_true,
            "y_pred"          : c.y_pred,
            "confidence"      : c.confidence,
            "confidence_zone" : c.confidence_zone,
            "is_error"        : c.is_error,
            "cluster"         : clust,
            "archetype"       : alabel,
            "risk_level"      : c.risk_level,
        })
    archetype_df = pd.DataFrame(arch_rows)

    logger.info(
        f"Archetypes discovered: {best_k} clusters  "
        f"silhouette={sil:.4f}  PCA_var={exp_var:.3f}"
    )
    for m in cluster_meta:
        logger.info(f"  [{m['cluster']}] {m['label']}: n={m['n_samples']} errors={m['n_errors']}")

    return ArchetypeResult(
        labels          = labels,
        n_clusters      = best_k,
        silhouette_score= sil,
        cluster_meta    = cluster_meta,
        X_pca           = X_pca,
        sample_indices  = indices,
        archetype_df    = archetype_df,
        explained_var   = exp_var,
    )
