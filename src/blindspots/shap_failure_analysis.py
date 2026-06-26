from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_shap_failure_analysis(
    shap_values,
    feature_names,
    fcs,
    y_true,
    y_proba,
    plots_dir,
    reports_dir,
):
    plots_dir = Path(plots_dir)
    reports_dir = Path(reports_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)

    ranking_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": mean_abs,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    ranking_df.to_csv(
        reports_dir / "blindspot_shap_failure.csv",
        index=False,
    )

    top_features = ranking_df["feature"].head(10).tolist()

    comparison_plot = plots_dir / "shap_failure_comparison.png"

    plt.figure(figsize=(10, 5))
    plt.bar(
        ranking_df["feature"].head(10),
        ranking_df["importance"].head(10),
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(comparison_plot)
    plt.close()

    masking_plot = plots_dir / "dominant_feature_masking.png"

    cumulative = (
        ranking_df["importance"].cumsum()
        / ranking_df["importance"].sum()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker="o",
    )
    plt.xlabel("Top Features")
    plt.ylabel("Cumulative Importance")
    plt.tight_layout()
    plt.savefig(masking_plot)
    plt.close()

    return {
        "comparison": {
            "top_failure_features": top_features
        },
        "comparison_plot": comparison_plot,
        "masking_plot": masking_plot,
        "ranking_df": ranking_df,
    }