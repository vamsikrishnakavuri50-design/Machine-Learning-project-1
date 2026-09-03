import traceback

from flask import Flask, render_template

from load_data import get_data_summary, load_data
from placement_eda import run_eda


app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = str(e)

    return render_template(
        "EDA.html",
        active="eda",
        results=results,
        error=error
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():

    error = None
    preprocessing = None

    try:

        data = load_data()

        # ----------------------------------------------------
        # Target variable
        # ----------------------------------------------------

        target = "PlacementStatus"


        # ----------------------------------------------------
        # Columns removed before model training
        # ----------------------------------------------------

        removed_columns = [
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]


        # Keep only columns that exist
        removed_columns = [
            column
            for column in removed_columns
            if column in data.columns
        ]


        # ----------------------------------------------------
        # Features used by the models
        # ----------------------------------------------------

        feature_data = data.drop(
            columns=removed_columns + [target],
            errors="ignore"
        )


        # ----------------------------------------------------
        # Numerical features
        # ----------------------------------------------------

        numerical_features = feature_data.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()


        # ----------------------------------------------------
        # Categorical features
        # ----------------------------------------------------

        categorical_features = feature_data.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()


        # ----------------------------------------------------
        # Preprocessing information
        # ----------------------------------------------------

        preprocessing = {

            "original_rows": data.shape[0],

            "original_columns": data.shape[1],

            "target": target,

            "feature_count": feature_data.shape[1],

            "removed_columns": removed_columns,

            "numerical_features": numerical_features,

            "categorical_features": categorical_features,

            "num_numerical": len(
                numerical_features
            ),

            "num_categorical": len(
                categorical_features
            ),

            "missing_numerical":
                "Median Imputation",

            "numerical_scaling":
                "StandardScaler",

            "missing_categorical":
                "Most Frequent Imputation",

            "categorical_encoding":
                "OneHotEncoder",

            "unknown_categories":
                "handle_unknown='ignore'",

            "train_samples":
                int(data.shape[0] * 0.80),

            "test_samples":
                int(data.shape[0] * 0.20),

            "split":
                "80% Training / 20% Testing",

            "random_state":
                42
        }


    except FileNotFoundError as e:

        error = str(e)


    except Exception as e:

        traceback.print_exc()

        error = str(e)


    return render_template(
        "preprocessing.html",
        active="preprocessing",
        preprocessing=preprocessing,
        error=error
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)