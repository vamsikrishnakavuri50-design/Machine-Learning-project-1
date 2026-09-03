import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = r"C:\Users\vamsi\PycharmProjects\PythonProject1\placement_dataset.csv"

data = pd.read_csv(FILE_PATH)

print("==========================================")
print("       ADABOOST - PLACEMENT STATUS")
print("==========================================")

print("\nOriginal Dataset Shape:")
print(data.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("\nColumns Available in Dataset:")

for column in data.columns:
    print("-", column)


# ============================================================
# 3. FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "Placement Status",
    "PlacementStatus",
    "placement status",
    "placement_status",
    "Placement_Status",
    "Placement"
]

TARGET = None

for column in data.columns:

    cleaned_column = (
        column.strip()
        .lower()
        .replace("_", " ")
    )

    for candidate in target_candidates:

        cleaned_candidate = (
            candidate.lower()
            .replace("_", " ")
        )

        if cleaned_column == cleaned_candidate:
            TARGET = column
            break

    if TARGET is not None:
        break


# If target is not found
if TARGET is None:

    print("\nERROR: Could not find PlacementStatus column.")

    print("\nAvailable columns:")
    print(data.columns.tolist())

    raise SystemExit


print("\nTarget Column Found:")
print(TARGET)


# ============================================================
# 4. REMOVE UNWANTED COLUMNS
# ============================================================

# These columns should NOT be used as input features.
#
# StudentID     -> Identifier, not useful for prediction
# TARGET        -> Target variable
# Salary Package -> Information available after placement
# IsAnomaly     -> Remove to avoid possible data leakage


REMOVE_COLUMNS = [
    "StudentID",
    TARGET,
    "Salary Package",
    "IsAnomaly"
]


# Only remove columns that actually exist
columns_to_remove = [
    column
    for column in REMOVE_COLUMNS
    if column in data.columns
]


X = data.drop(columns=columns_to_remove)

y = data[TARGET]


# ============================================================
# 5. TARGET VARIABLE INFORMATION
# ============================================================

print("\n==========================================")
print("TARGET VARIABLE")
print("==========================================")

print("Target:", TARGET)

print("\nTarget Values:")
print(y.value_counts())

print("\nNumber of Classes:")
print(y.nunique())


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\n==========================================")
print("FEATURE INFORMATION")
print("==========================================")

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 7. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 9. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("TRAIN-TEST SPLIT")
print("==========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 11. CREATE ADABOOST CLASSIFIER
# ============================================================

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)


# ============================================================
# 12. CREATE COMPLETE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            adaboost
        )
    ]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n==========================================")
print("TRAINING ADABOOST MODEL")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 15. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==========================================")
print("MODEL ACCURACY")
print("==========================================")

print("Accuracy:", accuracy)

print(
    "Accuracy Percentage: {:.2f}%".format(
        accuracy * 100
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 18. SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")


results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})


print(results)


# ============================================================
# 19. MODEL EXECUTION COMPLETED
# ============================================================

print("\n==========================================")
print("ADABOOST MODEL EXECUTION COMPLETED")
print("==========================================")