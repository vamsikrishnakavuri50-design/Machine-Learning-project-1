import matplotlib

from load_data import load_data

matplotlib.use("Agg")      # MUST be before importing pyplot

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

# Folder where charts will be stored
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, filename), dpi=100)
    plt.close()


def run_eda():
    """
    Performs Exploratory Data Analysis and
    saves charts into static/charts folder.
    """

    data = load_data()

    charts = []

    # =============================
    # Dataset Information
    # =============================
    shape = data.shape
    columns = list(data.columns)
    dtypes = data.dtypes.astype(str).to_dict()

    # =============================
    # Missing Values
    # =============================
    missing = data.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    missing_df = missing_df[missing_df["Missing Values"] > 0]

    if not missing_df.empty:

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x="Column",
            y="Missing Values",
            data=missing_df
        )

        plt.xticks(rotation=45)
        plt.title("Missing Values")

        save_chart("missing_values.png")

        charts.append("missing_values.png")

    # =============================
    # Duplicate Rows
    # =============================
    duplicate_count = data.duplicated().sum()

    # =============================
    # Placement Distribution
    # =============================
    if "PlacementStatus" in data.columns:

        plt.figure(figsize=(8, 5))

        sns.countplot(
            x="PlacementStatus",
            data=data
        )

        plt.title("Placement Status Distribution")

        save_chart("placement_distribution.png")

        charts.append("placement_distribution.png")

    # =============================
    # Numeric Feature Histograms
    # =============================
    numeric_columns = data.select_dtypes(include="number").columns

    for column in numeric_columns:

        plt.figure(figsize=(7, 4))

        sns.histplot(
            data[column],
            bins=30,
            kde=True
        )

        plt.title(column)

        filename = f"{column}.png"

        save_chart(filename)

        charts.append(filename)

    # =============================
    # Correlation Heatmap
    # =============================
    numeric = data.select_dtypes(include="number")

    if len(numeric.columns) > 1:

        plt.figure(figsize=(12, 8))

        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        save_chart("correlation_heatmap.png")

        charts.append("correlation_heatmap.png")

    return {

        "shape": shape,

        "columns": columns,

        "dtypes": dtypes,

        "duplicates": int(duplicate_count),

        "missing": missing_df.to_dict(orient="records"),

        "charts": charts
    }


if __name__ == "__main__":

    results = run_eda()

    print(results)