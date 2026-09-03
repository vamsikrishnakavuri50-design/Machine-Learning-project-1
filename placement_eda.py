import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data


# --------------------------------------------------
# Charts Directory
# --------------------------------------------------

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(
        _chart_path(filename),
        bbox_inches="tight"
    )
    plt.close("all")


# --------------------------------------------------
# Main EDA Function
# --------------------------------------------------

def run_eda() -> dict:

    # Load dataset
    data = load_data()

    charts = []

    # --------------------------------------------------
    # 1. Missing Values
    # --------------------------------------------------

    missing = data.isnull().sum()

    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    if not missing_df.empty:

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel("Missing %")

        plt.title(
            "Missing Values by Column"
        )

        _save("missing_values.png")

        charts.append(
            "missing_values.png"
        )


    # --------------------------------------------------
    # 2. Duplicate Rows
    # --------------------------------------------------

    duplicate_count = int(
        data.duplicated().sum()
    )


    # --------------------------------------------------
    # 3. Placement Status Distribution
    # --------------------------------------------------

    target_counts = (
        data["PlacementStatus"]
        .value_counts()
        .to_dict()
    )

    plt.figure()

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.xlabel(
        "Placement Status (0 = Not Placed, 1 = Placed)"
    )

    plt.ylabel("Count")

    plt.title(
        "Count of Placement Status"
    )

    _save(
        "target_distribution.png"
    )

    charts.append(
        "target_distribution.png"
    )


    # --------------------------------------------------
    # 4. Individual Feature Distributions
    # --------------------------------------------------

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating"
    ]

    for col in hist_cols:

        if col in data.columns:

            plt.figure()

            sns.histplot(
                data[col],
                kde=True
            )

            plt.title(
                f"Distribution of {col}"
            )

            plt.xlabel(col)

            fname = (
                f"hist_{col.lower()}.png"
            )

            _save(fname)

            charts.append(fname)


    # --------------------------------------------------
    # 5. Numeric Feature Distributions
    # --------------------------------------------------

    all_hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    all_hist_cols = [
        c for c in all_hist_cols
        if (
            c in data.columns
            and pd.api.types.is_numeric_dtype(
                data[c]
            )
        )
    ]

    if all_hist_cols:

        data[all_hist_cols].hist(
            figsize=(14, 10),
            bins=20
        )

        plt.suptitle(
            "Numeric Feature Distributions"
        )

        _save(
            "numeric_distributions.png"
        )

        charts.append(
            "numeric_distributions.png"
        )


    # --------------------------------------------------
    # 6. CGPA Distribution with Mean
    # --------------------------------------------------

    if "CGPA" in data.columns:

        plt.figure()

        sns.histplot(
            data["CGPA"],
            kde=True
        )

        plt.axvline(
            x=np.mean(data["CGPA"]),
            color="green",
            linestyle="--",
            label="Mean"
        )

        plt.legend()

        plt.title(
            "CGPA Distribution with Mean"
        )

        _save(
            "cgpa_with_mean.png"
        )

        charts.append(
            "cgpa_with_mean.png"
        )


    # --------------------------------------------------
    # 7. Outlier Detection - Boxplots
    # --------------------------------------------------

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package"
    ]

    box_cols = [
        c for c in box_cols
        if (
            c in data.columns
            and pd.api.types.is_numeric_dtype(
                data[c]
            )
        )
    ]

    for col in box_cols:

        plt.figure(figsize=(10, 4))

        sns.boxplot(
            x=data[col],
            color="skyblue"
        )

        plt.title(
            f"Boxplot - {col}"
        )

        fname = (
            "boxplot_"
            + col.lower().replace(" ", "_")
            + ".png"
        )

        _save(fname)

        charts.append(fname)


    # --------------------------------------------------
    # 8. Correlation Heatmap
    # --------------------------------------------------

    corr = (
        data
        .select_dtypes(include=[np.number])
        .corr()
    )

    plt.figure(
        figsize=(16, 12)
    )

    sns.heatmap(
        np.round(corr, 2),
        annot=True,
        cmap="YlOrRd"
    )

    plt.title(
        "Correlation Heatmap"
    )

    _save(
        "correlation_heatmap.png"
    )

    charts.append(
        "correlation_heatmap.png"
    )


    # --------------------------------------------------
    # 9. Relationship Plots
    # --------------------------------------------------

    if {
        "CGPA",
        "Salary Package"
    }.issubset(data.columns):

        plt.figure()

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=data,
            scatter_kws={
                "alpha": 0.6
            }
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        _save(
            "cgpa_vs_salary.png"
        )

        charts.append(
            "cgpa_vs_salary.png"
        )


    if {
        "AptitudeTestScore",
        "CodingTestScore"
    }.issubset(data.columns):

        plt.figure()

        sns.regplot(
            x="AptitudeTestScore",
            y="CodingTestScore",
            data=data,
            scatter_kws={
                "alpha": 0.6
            }
        )

        plt.title(
            "Aptitude vs Coding Test Score"
        )

        _save(
            "aptitude_vs_coding.png"
        )

        charts.append(
            "aptitude_vs_coding.png"
        )


    # --------------------------------------------------
    # 10. Categorical Feature Counts
    # --------------------------------------------------

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [
        c for c in cat_cols
        if c in data.columns
    ]

    for col in cat_cols:

        plt.figure()

        sns.countplot(
            x=col,
            data=data
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            f"Count Plot - {col}"
        )

        fname = (
            f"cat_{col.lower()}.png"
        )

        _save(fname)

        charts.append(fname)


    # --------------------------------------------------
    # 11. Gender vs Placement Status
    # --------------------------------------------------

    if {
        "Gender",
        "PlacementStatus"
    }.issubset(data.columns):

        plt.figure()

        sns.countplot(
            x="Gender",
            hue="PlacementStatus",
            data=data,
            palette="pastel"
        )

        plt.title(
            "Placement Outcome by Gender"
        )

        _save(
            "gender_vs_placement.png"
        )

        charts.append(
            "gender_vs_placement.png"
        )


    # --------------------------------------------------
    # 12. College Tier vs Placement
    # --------------------------------------------------

    if {
        "CollegeTier",
        "PlacementStatus"
    }.issubset(data.columns):

        plt.figure()

        sns.countplot(
            x="CollegeTier",
            hue="PlacementStatus",
            data=data,
            palette="muted"
        )

        plt.title(
            "Placement Outcome by College Tier"
        )

        _save(
            "collegetier_vs_placement.png"
        )

        charts.append(
            "collegetier_vs_placement.png"
        )


    # --------------------------------------------------
    # Stream vs Placement
    # --------------------------------------------------

    if {
        "Stream",
        "PlacementStatus"
    }.issubset(data.columns):

        plt.figure()

        sns.countplot(
            x="Stream",
            hue="PlacementStatus",
            data=data,
            palette="muted"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            "Placement Outcome by Stream"
        )

        _save(
            "stream_vs_placement.png"
        )

        charts.append(
            "stream_vs_placement.png"
        )


    # --------------------------------------------------
    # 13. SGPA Trend Across Semesters
    # --------------------------------------------------

    sgpa_cols = [
        c for c in data.columns
        if c.startswith("SGPA_Sem")
    ]

    if sgpa_cols:

        mean_sgpa = (
            data[sgpa_cols]
            .mean()
        )

        plt.figure(
            figsize=(10, 6)
        )

        sns.lineplot(
            x=sgpa_cols,
            y=mean_sgpa.values,
            marker="o"
        )

        plt.title(
            "Average SGPA Trend Across Semesters"
        )

        plt.xlabel("Semester")

        plt.ylabel(
            "Average SGPA"
        )

        _save(
            "sgpa_trend.png"
        )

        charts.append(
            "sgpa_trend.png"
        )


    # --------------------------------------------------
    # 14. Salary Package Analysis
    # --------------------------------------------------

    if {
        "Salary Package",
        "PlacementStatus"
    }.issubset(data.columns):

        placed = data[
            data["PlacementStatus"] == 1
        ]

        plt.figure()

        sns.histplot(
            placed["Salary Package"],
            bins=20,
            kde=True,
            color="skyblue"
        )

        plt.title(
            "Salary Distribution for Placed Students"
        )

        plt.xlabel(
            "Salary Package"
        )

        plt.ylabel("Count")

        _save(
            "salary_distribution.png"
        )

        charts.append(
            "salary_distribution.png"
        )


        if "CollegeTier" in data.columns:

            plt.figure()

            sns.boxplot(
                x="CollegeTier",
                y="Salary Package",
                data=placed,
                hue="CollegeTier",
                legend=False,
                palette="Set3"
            )

            plt.title(
                "Salary Package by College Tier "
                "(Placed Students)"
            )

            _save(
                "salary_by_tier.png"
            )

            charts.append(
                "salary_by_tier.png"
            )


    # --------------------------------------------------
    # 15. Pairplot
    # --------------------------------------------------
    # Use only 1,000 rows so that the Flask page
    # does not become extremely slow.

    pairplot_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    pairplot_cols = [
        c for c in pairplot_cols
        if c in data.columns
    ]

    if (
        pairplot_cols
        and "PlacementStatus" in data.columns
    ):

        pairplot_data = data[
            pairplot_cols + ["PlacementStatus"]
        ].sample(
            n=min(1000, len(data)),
            random_state=42
        )

        sns.pairplot(
            pairplot_data,
            hue="PlacementStatus",
            diag_kind="hist",
            palette="husl"
        )

        plt.suptitle(
            "Pairwise Relationships "
            "Colored by Placement Status",
            y=1.02
        )

        plt.savefig(
            _chart_path("pairplot.png"),
            bbox_inches="tight"
        )

        plt.close("all")

        charts.append(
            "pairplot.png"
        )


    # --------------------------------------------------
    # Return Results to Flask
    # --------------------------------------------------

    missing_dict = {
        col: int(cnt)
        for col, cnt in missing.items()
        if cnt > 0
    }

    return {

        "n_rows": len(data),

        "n_cols": len(data.columns),

        "duplicate_count": duplicate_count,

        "missing": missing_dict,

        "target_counts": {
            str(k): int(v)
            for k, v in target_counts.items()
        },

        "charts": charts

    }


# --------------------------------------------------
# Test Run
# --------------------------------------------------

if __name__ == "__main__":

    results = run_eda()

    print("\nEDA completed successfully!")

    print("\nResults:")

    print(results)