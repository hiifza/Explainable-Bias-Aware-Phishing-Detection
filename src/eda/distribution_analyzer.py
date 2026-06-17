# src/eda/distribution_analyzer.py

import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.eda.visualization_manager import (
    save_figure,
    setup_plot_style,
)

logger = get_logger(__name__)

TARGET = "label"

PHISHING_COLOR = "#E24B4A"
LEGITIMATE_COLOR = "#1D9E75"

OUTLIER_FEATURES = [
    "URLLength",
    "DomainLength",
    "LineOfCode",
    "LargestLineLength",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfExternalRef",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfLettersInURL",
    "NoOfOtherSpecialCharsInURL",
]
def plot_single_distribution(
    df: pd.DataFrame,
    feature: str,
    plots_dir: Path,
    bins: int = 50,
) -> Optional[Path]:

    if feature not in df.columns:
        logger.warning(
            f"Feature '{feature}' not in DataFrame — skipping"
        )
        return None

    setup_plot_style()

    is_binary = df[feature].nunique() <= 2

    is_categorical = (
        df[feature].dtype == "object"
        or str(df[feature].dtype) == "category"
    )

    ph_data = df[df[TARGET] == 0][feature].dropna()
    lg_data = df[df[TARGET] == 1][feature].dropna()
    all_data = df[feature].dropna()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
    )

    fig.suptitle(
        f"Distribution: {feature}",
        fontsize=13,
        fontweight="700",
        y=1.01,
    )

    # --------------------------------------------------
    # PANEL 1 : HISTOGRAM
    # --------------------------------------------------

    ax = axes[0, 0]

    if is_categorical:

        vc = all_data.astype(str).value_counts().head(20)

        ax.bar(
            vc.index,
            vc.values,
            color="#185FA5",
            edgecolor="white",
        )

        ax.tick_params(
            axis="x",
            rotation=90,
        )

        ax.set_title("Top Categories")
        ax.set_ylabel("Count")

    elif is_binary:

        vc = all_data.value_counts().sort_index()

        ax.bar(
            vc.index.astype(str),
            vc.values,
            color="#185FA5",
            edgecolor="white",
        )

        ax.set_title("Value Counts")
        ax.set_ylabel("Count")

    else:

        ax.hist(
            all_data,
            bins=bins,
            color="#185FA5",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )

        ax.set_title("Overall Histogram")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{x:,.0f}"
            )
        )

    sns.despine(ax=ax)
        # --------------------------------------------------
    # PANEL 2 : CLASS COMPARISON
    # --------------------------------------------------

    ax = axes[0, 1]

    if is_categorical:

        ph_counts = (
            ph_data.astype(str)
            .value_counts(normalize=True)
            .head(10)
        )

        lg_counts = (
            lg_data.astype(str)
            .value_counts(normalize=True)
            .head(10)
        )

        cats = sorted(
            set(ph_counts.index).union(set(lg_counts.index))
        )[:10]

        x = np.arange(len(cats))
        width = 0.4

        ax.bar(
            x - width / 2,
            [ph_counts.get(c, 0) * 100 for c in cats],
            width,
            color=PHISHING_COLOR,
            label="Phishing",
        )

        ax.bar(
            x + width / 2,
            [lg_counts.get(c, 0) * 100 for c in cats],
            width,
            color=LEGITIMATE_COLOR,
            label="Legitimate",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            cats,
            rotation=90,
            fontsize=8,
        )

        ax.set_ylabel("%")
        ax.set_title("Top Categories by Class")
        ax.legend()

    elif is_binary:

        ph_pct = (
            len(ph_data[ph_data == 1])
            / max(len(ph_data), 1)
            * 100
        )

        lg_pct = (
            len(lg_data[lg_data == 1])
            / max(len(lg_data), 1)
            * 100
        )

        ax.bar(
            ["Phishing", "Legitimate"],
            [ph_pct, lg_pct],
            color=[
                PHISHING_COLOR,
                LEGITIMATE_COLOR,
            ],
        )

        ax.set_ylabel("% Positive")
        ax.set_title("Positive Rate by Class")

    else:

        try:

            sns.kdeplot(
                ph_data,
                ax=ax,
                color=PHISHING_COLOR,
                label="Phishing (0)",
                fill=True,
                alpha=0.25,
            )

            sns.kdeplot(
                lg_data,
                ax=ax,
                color=LEGITIMATE_COLOR,
                label="Legitimate (1)",
                fill=True,
                alpha=0.25,
            )

        except Exception:

            ax.hist(
                ph_data,
                bins=30,
                alpha=0.5,
                density=True,
                color=PHISHING_COLOR,
                label="Phishing",
            )

            ax.hist(
                lg_data,
                bins=30,
                alpha=0.5,
                density=True,
                color=LEGITIMATE_COLOR,
                label="Legitimate",
            )

        ax.legend()
        ax.set_title("Distribution by Class")

    sns.despine(ax=ax)

    # --------------------------------------------------
    # PANEL 3 : OVERALL BOXPLOT
    # --------------------------------------------------

    ax = axes[1, 0]

    if (
        not is_categorical
        and pd.api.types.is_numeric_dtype(all_data)
    ):

        numeric_data = pd.to_numeric(
            all_data,
            errors="coerce"
        ).dropna()

        ax.boxplot(
            numeric_data,
            vert=False,
            patch_artist=True,
            boxprops={
                "facecolor": "#B5D4F4"
            },
        )

        ax.set_title("Overall Boxplot")
        ax.set_xlabel(feature)

    else:

        ax.text(
            0.5,
            0.5,
            "Categorical Feature",
            ha="center",
            va="center",
            fontsize=12,
        )

        ax.axis("off")

        sns.despine(ax=ax)

    # --------------------------------------------------
    # PANEL 4 : BOXPLOT BY CLASS
    # --------------------------------------------------

    ax = axes[1, 1]

    if (
        not is_binary
        and not is_categorical
        and pd.api.types.is_numeric_dtype(all_data)
    ):

        bp = ax.boxplot(
            [ph_data, lg_data],
            vert=True,
            tick_labels=[
                "Phishing (0)",
                "Legitimate (1)",
            ],
            patch_artist=True,
        )

        for patch, color in zip(
            bp["boxes"],
            [
                PHISHING_COLOR,
                LEGITIMATE_COLOR,
            ],
        ):
            patch.set_facecolor(color)

        ax.set_ylabel(feature)
        ax.set_title("Boxplot by Class")

    else:

        stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Count",
                    "Mean",
                    "Std",
                ],
                "Phishing": [
                    len(ph_data),
                    round(
                        pd.to_numeric(
                            ph_data,
                            errors="coerce",
                        ).mean(),
                        4,
                    )
                    if not is_categorical
                    else "-",
                    round(
                        pd.to_numeric(
                            ph_data,
                            errors="coerce",
                        ).std(),
                        4,
                    )
                    if not is_categorical
                    else "-",
                ],
                "Legitimate": [
                    len(lg_data),
                    round(
                        pd.to_numeric(
                            lg_data,
                            errors="coerce",
                        ).mean(),
                        4,
                    )
                    if not is_categorical
                    else "-",
                    round(
                        pd.to_numeric(
                            lg_data,
                            errors="coerce",
                        ).std(),
                        4,
                    )
                    if not is_categorical
                    else "-",
                ],
            }
        )

        table = ax.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            loc="center",
        )

        table.scale(1, 1.5)

        ax.axis("off")
        ax.set_title("Class Statistics")

    plt.tight_layout()

    out_path = plots_dir / f"{feature}.png"

    return save_figure(
        fig,
        out_path,
    )


def plot_all_distributions(
    df: pd.DataFrame,
    features: list[str],
    plots_dir: Path,
) -> list[Path]:

    dist_dir = plots_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for i, feat in enumerate(features, 1):

        logger.debug(
            f"[{i}/{len(features)}] {feat}"
        )

        path = plot_single_distribution(
            df=df,
            feature=feat,
            plots_dir=dist_dir,
        )

        if path is not None:
            saved.append(path)

    logger.info(
        f"Distribution plots saved: {len(saved)}"
    )

    return saved


def compute_outlier_stats(
    df: pd.DataFrame,
    outlier_features: list[str],
) -> pd.DataFrame:

    rows = []

    for feat in outlier_features:

        if feat not in df.columns:
            continue

        s = pd.to_numeric(
            df[feat],
            errors="coerce"
        ).dropna()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        iqr = q3 - q1

        upper_fence = q3 + 1.5 * iqr

        n_outliers = int(
            (s > upper_fence).sum()
        )

        rows.append(
            {
                "feature": feat,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q1": float(q1),
                "median": float(s.median()),
                "q3": float(q3),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "p999": float(s.quantile(0.999)),
                "max": float(s.max()),
                "iqr": float(iqr),
                "upper_fence": float(upper_fence),
                "n_outliers": n_outliers,
                "outlier_pct": round(
                    n_outliers / len(s) * 100,
                    4,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "outlier_pct",
            ascending=False,
        )

    return result.reset_index(drop=True)
def plot_outlier_boxplots(
    df: pd.DataFrame,
    outlier_features: list[str],
    plots_dir: Path,
) -> None:

    setup_plot_style()

    valid = [
        f for f in outlier_features
        if f in df.columns
    ]

    if not valid:
        return

    rows = int(np.ceil(len(valid) / 2))

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, rows * 3),
    )

    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, valid):

        ph = df[df[TARGET] == 0][feat].dropna()
        lg = df[df[TARGET] == 1][feat].dropna()

        bp = ax.boxplot(
            [ph, lg],
            vert=False,
            tick_labels=[
                "Phishing",
                "Legitimate",
            ],
            patch_artist=True,
        )

        for patch, color in zip(
            bp["boxes"],
            [
                PHISHING_COLOR,
                LEGITIMATE_COLOR,
            ],
        ):
            patch.set_facecolor(color)

        ax.set_title(feat)

    for ax in axes[len(valid):]:
        ax.set_visible(False)

    plt.tight_layout()

    save_figure(
        fig,
        plots_dir / "outlier_boxplots.png",
    )
def plot_skewness_summary(
    df: pd.DataFrame,
    features: list[str],
    plots_dir: Path,
) -> None:

    numeric = [
        c
        for c in features
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    skew = (
        df[numeric]
        .skew()
        .abs()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(10, max(8, len(skew) * 0.25))
    )

    ax.barh(
        skew.index,
        skew.values,
    )

    ax.set_title(
        "Absolute Feature Skewness"
    )

    plt.tight_layout()

    save_figure(
        fig,
        plots_dir / "skewness_summary.png",
    )
def run_distribution_analysis(
    df: pd.DataFrame,
    features: list[str],
    output_dir: str | Path = "outputs/reports",
    plots_dir: str | Path = "outputs/plots/eda",
    plot_all: bool = True,
) -> dict:

    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("=" * 60)
    logger.info(
        "M2.1 — DISTRIBUTION ANALYSIS MODULE"
    )
    logger.info("=" * 60)

    dist_paths = []

    if plot_all:

        logger.info(
            f"Generating distribution plots for {len(features)} features …"
        )

        dist_paths = plot_all_distributions(
            df,
            features,
            plots_dir,
        )

    outlier_stats = compute_outlier_stats(
        df,
        [
            f
            for f in OUTLIER_FEATURES
            if f in df.columns
        ],
    )

    outlier_stats.to_csv(
        output_dir / "outlier_analysis.csv",
        index=False,
    )

    plot_outlier_boxplots(
        df,
        OUTLIER_FEATURES,
        plots_dir,
    )

    plot_skewness_summary(
        df,
        features,
        plots_dir,
    )

    return {
        "outlier_stats": outlier_stats,
        "distribution_paths": dist_paths,
        "n_outlier_features": int(
            (outlier_stats["outlier_pct"] > 5).sum()
        )
        if not outlier_stats.empty
        else 0,
    }

