import os
import pandas as pd


# ============================================================
# 1. DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "placement_dataset.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_data(path=DATA_PATH):

    # Check whether dataset exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    # Load CSV dataset
    df = pd.read_csv(path)

    # Remove unnecessary spaces from column names
    df.columns = df.columns.str.strip()

    return df


# ============================================================
# 3. DATA SUMMARY
# ============================================================

def get_data_summary():

    df = load_data()


    # --------------------------------------------------------
    # Identify categorical columns
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns


    # --------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------

    summary = {

        # Dataset dimensions
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],


        # ----------------------------------------------------
        # Column Information
        # ----------------------------------------------------

        "columns": list(df.columns),


        # ----------------------------------------------------
        # Data Types
        # ----------------------------------------------------

        "dtypes": {
            col: str(df[col].dtype)
            for col in df.columns
        },


        # ----------------------------------------------------
        # Missing Values
        # ----------------------------------------------------

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },


        # ----------------------------------------------------
        # Missing Value Percentage
        # ----------------------------------------------------

        "missing_percent": {
            col: round(
                df[col].isnull().mean() * 100,
                2
            )
            for col in df.columns
        },


        # ----------------------------------------------------
        # Duplicate Rows
        # ----------------------------------------------------

        "duplicates": int(
            df.duplicated().sum()
        ),


        # ----------------------------------------------------
        # Memory Usage in KB
        # ----------------------------------------------------

        "memory_usage": round(
            df.memory_usage(
                deep=True
            ).sum() / 1024,
            2
        ),


        # ----------------------------------------------------
        # Numerical Statistics
        # ----------------------------------------------------

        "numeric_summary": (
            df.describe(
                include=["number"]
            )
            .round(2)
            .to_html(
                classes="data-table",
                border=0
            )
        ),


        # ----------------------------------------------------
        # Categorical Statistics
        # ----------------------------------------------------

        "categorical_summary": (
            df[categorical_columns]
            .describe()
            .to_html(
                classes="data-table",
                border=0
            )
            if len(categorical_columns) > 0
            else "<p>No categorical columns found.</p>"
        ),


        # ----------------------------------------------------
        # Dataset Preview
        # ----------------------------------------------------

        "preview": (
            df.head(10)
            .to_dict("records")
        )
    }


    return summary


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    summary = get_data_summary()

    print("==========================================")
    print("          DATASET INFORMATION")
    print("==========================================")

    print(
        "\nNumber of Rows:",
        summary["n_rows"]
    )

    print(
        "Number of Columns:",
        summary["n_cols"]
    )

    print("\nDataset loaded successfully!")

    print("==========================================")