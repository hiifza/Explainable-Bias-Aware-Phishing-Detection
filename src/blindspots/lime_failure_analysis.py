from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def run_lime_failure_analysis(
    agreement_df,
    fcs,
    plots_dir,
    reports_dir,
):
    plots_dir = Path(plots_dir)
    reports_dir = Path(reports_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if agreement_df is None or agreement_df.empty:
        lime_freq_df = pd.DataFrame(
            columns=["feature", "count"]
        )
    else:
        cols = [
            c
            for c in agreement_df.columns
            if "feature" in c.lower()
        ]

        features = []

        for col in cols:
            features.extend(
                agreement_df[col]
                .dropna()
                .astype(str)
                .tolist()
            )

        if len(features) == 0:
            lime_freq_df = pd.DataFrame(
                columns=["feature", "count"]
            )
        else:
            lime_freq_df = (
                pd.Series(features)
                .value_counts()
                .reset_index()
            )

            lime_freq_df.columns = [
                "feature",
                "count",
            ]

    lime_freq_df.to_csv(
        reports_dir / "lime_failure_features.csv",
        index=False,
    )

    plot_path = plots_dir / "lime_failure_features.png"

    if not lime_freq_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(
            lime_freq_df["feature"].head(10),
            lime_freq_df["count"].head(10),
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {
        "lime_freq_df": lime_freq_df,
        "plot_path": plot_path,
    }