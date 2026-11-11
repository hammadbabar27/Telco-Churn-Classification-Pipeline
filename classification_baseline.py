"""
Task 4 — Baseline Classification Pipeline (Telco Customer Churn)

"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# 1. Configuration and dataset loading
# ---------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

LOCAL_DATA_CANDIDATES = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Telco-Customer-Churn.csv",
    "customer_churn.csv",
    "telco.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"


def load_dataset():
    """Load a local Task 2 CSV if present; otherwise use the IBM public mirror."""
    for filename in LOCAL_DATA_CANDIDATES:
        path = Path(filename)
        if path.exists():
            print(f"Loading local dataset: {path.resolve()}")
            return pd.read_csv(path, keep_default_na=False), str(path)

    print("No local CSV found. Loading the public IBM Telco sample...")
    try:
        data = pd.read_csv(DATA_URL, keep_default_na=False)
        return data, DATA_URL
    except Exception as exc:
        raise RuntimeError(
            "Could not load the dataset from the public URL. "
            "Download WA_Fn-UseC_-Telco-Customer-Churn.csv from the Task 2 "
            "Kaggle dataset and place it beside this script, then run again."
        ) from exc


# ---------------------------------------------------------------------
# 2. Prepare target and fix TotalCharges
# ---------------------------------------------------------------------

def prepare_data(df):
    required_columns = {"Churn", "TotalCharges", "customerID"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df.copy()

    # In the original CSV, blank TotalCharges entries are stored as strings.
    blank_total_charges = (
        df["TotalCharges"].astype(str).str.strip().eq("").sum()
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    missing_total_charges = int(df["TotalCharges"].isna().sum())

    # Remove rows with a missing target only; do not drop rows for missing predictors.
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = df["Churn"].astype(str).str.strip()

    if not set(df["Churn"].unique()).issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
        raise ValueError(
            "Expected Churn labels 'No' and 'Yes'. Found: "
            f"{sorted(df['Churn'].unique().tolist())}"
        )

    # customerID is an identifier, not a meaningful predictive feature.
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]

    return X, y, blank_total_charges, missing_total_charges, df


# ---------------------------------------------------------------------
# 3. Build preprocessing inside a Pipeline
# ---------------------------------------------------------------------

def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


# ---------------------------------------------------------------------
# 4. Evaluation helper
# ---------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    # Find the probability column for Churn = Yes safely from model.classes_.
    class_labels = list(model.classes_)
    positive_index = class_labels.index(POSITIVE_LABEL)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    y_test_binary = (y_test == POSITIVE_LABEL).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test_binary, probabilities),
    }

    matrix = confusion_matrix(
        y_test, predictions, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
    )

    print(f"\n{'=' * 68}")
    print(name)
    print("=" * 68)
    print(classification_report(
        y_test,
        predictions,
        labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        target_names=["No (Stayed)", "Yes (Churned)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual; columns = predicted):")
    print(pd.DataFrame(
        matrix,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"],
    ))

    return metrics, matrix, predictions, probabilities


# ---------------------------------------------------------------------
# 5. Markdown report generation
# ---------------------------------------------------------------------

def pct(value):
    return f"{value * 100:.2f}%"


def matrix_to_markdown(matrix):
    return (
        "| Actual / Predicted | Predicted No | Predicted Yes |\n"
        "|---|---:|---:|\n"
        f"| Actual No | {matrix[0, 0]:,} | {matrix[0, 1]:,} |\n"
        f"| Actual Yes | {matrix[1, 0]:,} | {matrix[1, 1]:,} |"
    )


def create_markdown_report(
    report_path,
    source,
    row_count,
    blank_total_charges,
    missing_total_charges,
    train_counts,
    test_counts,
    numeric_features,
    categorical_features,
    results,
    matrices,
):
    historical = {
        "Accuracy": 0.7346,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "ROC-AUC": None,
    }

    rows = [
        (
            "Task 2 historical majority baseline (full dataset)",
            historical,
        ),
        ("Task 4 majority baseline (stratified test set)", results["majority"]),
        ("Logistic Regression (default)", results["default"]),
        ("Logistic Regression (balanced)", results["balanced"]),
    ]

    table_lines = [
        "| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, values in rows:
        auc_text = "Not calculated in Task 2" if values["ROC-AUC"] is None else f"{values['ROC-AUC']:.4f}"
        table_lines.append(
            f"| {name} | {pct(values['Accuracy'])} | "
            f"{pct(values['Precision'])} | {pct(values['Recall'])} | "
            f"{pct(values['F1'])} | {auc_text} |"
        )

    default = results["default"]
    balanced = results["balanced"]

    default_cm = matrices["default"]
    balanced_cm = matrices["balanced"]
    majority_cm = matrices["majority"]

    if balanced["Recall"] > default["Recall"]:
        recommendation = (
            f"**Recommendation: Logistic Regression with "
            f"`class_weight='balanced'`.** Its churn recall is "
            f"{pct(balanced['Recall'])}, compared with {pct(default['Recall'])} "
            f"for the default model. This means it identifies a larger share of "
            f"customers who actually churn. The trade-off is precision "
            f"({pct(balanced['Precision'])} balanced vs. "
            f"{pct(default['Precision'])} default), so the retention team may "
            f"receive more false alarms. Given that missing a churner is likely "
            f"more costly than contacting a customer who would have stayed, "
            f"the higher-recall model is the stronger starting point."
        )
    elif balanced["Recall"] == default["Recall"]:
        recommendation = (
            f"**Recommendation: compare the false-positive counts and precision "
            f"before choosing.** Both models have the same churn recall "
            f"({pct(default['Recall'])}). The balanced model therefore has not "
            f"demonstrated a recall advantage on this test split. Prefer the "
            f"model with the more useful precision/false-alarm trade-off, then "
            f"validate the decision with business costs."
        )
    else:
        recommendation = (
            f"**Recommendation: Logistic Regression (default) on this test split.** "
            f"The default model has higher churn recall "
            f"({pct(default['Recall'])}) than the balanced model "
            f"({pct(balanced['Recall'])}). Although class weighting is intended "
            f"to give the minority class more influence during training, it does "
            f"not guarantee higher recall on every dataset or split. Review the "
            f"confusion matrices and consider threshold tuning on a validation "
            f"set before deployment."
        )

    text = f"""# Task 4 — Baseline Classification Pipeline (Telco Customer Churn)

## 1. Objective

Build and evaluate a Logistic Regression classification pipeline that predicts whether a telecom customer will churn (`Churn = Yes`). The task continues the Task 2 Telco Customer Churn EDA and majority-class baseline.

## 2. Dataset and data preparation

- **Dataset rows used:** {row_count:,}
- **Dataset source:** {source}
- **Target:** `Churn` (`No` / `Yes`)
- **Rows with blank `TotalCharges` strings before conversion:** {blank_total_charges}
- **Rows with missing `TotalCharges` after numeric conversion:** {missing_total_charges:,} (imputed inside the numeric pipeline)
- **Identifier removed:** `customerID`
- **Numeric features:** {len(numeric_features)}
- **Categorical features:** {len(categorical_features)}

`TotalCharges` was converted to numeric using `pd.to_numeric(..., errors="coerce")`. The blank strings become missing values and are handled by the numeric median imputer inside the pipeline. The special category values such as `No internet service` and `No phone service` were retained as valid categorical values; they were not treated as missing data.

## 3. Validation strategy

The data was divided into training and test sets using an **80/20 split**, `random_state={RANDOM_STATE}`, and `stratify=y`.

| Split | No (Stayed) | Yes (Churned) | Total |
|---|---:|---:|---:|
| Training | {train_counts[NEGATIVE_LABEL]:,} | {train_counts[POSITIVE_LABEL]:,} | {sum(train_counts.values()):,} |
| Testing | {test_counts[NEGATIVE_LABEL]:,} | {test_counts[POSITIVE_LABEL]:,} | {sum(test_counts.values()):,} |

Stratification preserves approximately the same churn/non-churn proportions in both sets.

All learned preprocessing was placed inside a scikit-learn `Pipeline` and fitted only on the training data:
- Numeric columns: median imputation followed by `StandardScaler`.
- Categorical columns: most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
- `customerID` was excluded because it is only an identifier.

## 4. Model comparison

The positive class for precision, recall and F1 is **`Churn = Yes`**.

{chr(10).join(table_lines)}

**Baseline comparison note:** Task 2 calculated its majority-class baseline on the full dataset before introducing train/test validation. Its original results are retained above for continuity. The Task 4 majority baseline is also evaluated on the same stratified test set as the Logistic Regression models, making that row the fair like-for-like comparison. Task 2 did not calculate ROC-AUC, so it is marked as not calculated rather than inventing a historical value.

## 5. Confusion matrices

Rows represent actual classes and columns represent predicted classes. The positive class is `Yes`.

### Task 2 historical majority-class baseline (full dataset)

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 5,174 | 0 |
| Actual Yes | 1,869 | 0 |

### Task 4 majority-class baseline (same test set)

{matrix_to_markdown(majority_cm)}

### Logistic Regression — default class weighting

{matrix_to_markdown(default_cm)}

### Logistic Regression — `class_weight="balanced"`

{matrix_to_markdown(balanced_cm)}

### Business meaning of the four cells

- **True Negative (TN):** A customer stayed, and the model correctly predicted `No`.
- **False Positive (FP):** A customer stayed, but the model predicted `Yes`. This is a false alarm and may cause an unnecessary retention offer.
- **False Negative (FN):** A customer churned, but the model predicted `No`. This is a missed churner and may mean the business loses a customer without attempting retention.
- **True Positive (TP):** A customer churned, and the model correctly predicted `Yes`.

## 6. Business interpretation and recommendation

{recommendation}

The default Logistic Regression model uses the standard class-weighting behavior. With `class_weight="balanced"`, scikit-learn automatically assigns greater training weight to the less frequent class. It does not add new churn examples or change the test-set distribution.

Accuracy alone is not enough for this problem. Recall measures the share of actual churners found by the model, while precision measures how many customers flagged as churn risks really churned. ROC-AUC measures how well the model ranks positive cases above negative cases across possible thresholds; it does not tell us the precision/recall trade-off at the default 0.5 threshold. Therefore, ROC-AUC should be interpreted alongside the confusion matrix, precision, recall and F1.

## 7. Threshold awareness and limitations

Both Logistic Regression models use the default decision threshold of **0.50**. A lower threshold may identify more potential churners, but it can also increase false alarms. Any threshold adjustment should be selected using validation data and a clear business cost trade-off—not by repeatedly tuning against the final test set.

This is a baseline model, not a production-ready churn system. Before deployment, the business should validate performance on newer customer data, decide how many retention contacts it can handle, and monitor model performance over time.

## 8. Conclusion

This task demonstrates a leakage-aware classification workflow: stratified splitting, training-only preprocessing, Logistic Regression, explicit class-imbalance handling, and evaluation using multiple metrics. The final model choice should prioritize the business cost of missed churners while keeping false alarms at a manageable level.
"""

    report_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# 6. Main workflow
# ---------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df, source = load_dataset()
    print("\nOriginal dataset shape:", df.shape)

    X, y, blank_total_charges, missing_total_charges, cleaned_df = prepare_data(df)

    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nTarget percentages:")
    print((y.value_counts(normalize=True) * 100).round(2))

    # Mandatory stratified 80/20 split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_counts = y_train.value_counts().reindex(
        [NEGATIVE_LABEL, POSITIVE_LABEL], fill_value=0
    ).to_dict()
    test_counts = y_test.value_counts().reindex(
        [NEGATIVE_LABEL, POSITIVE_LABEL], fill_value=0
    ).to_dict()

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print("Training class counts:", train_counts)
    print("Testing class counts:", test_counts)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    # Same-test-set majority baseline for a fair comparison.
    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)

    # Two Logistic Regression pipelines. Each pipeline learns preprocessing
    # from X_train only because preprocessing is inside the Pipeline.
    default_model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    balanced_model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )),
        ]
    )

    print("\nTraining majority-class baseline...")
    majority_metrics, majority_cm, _, _ = evaluate_model(
        "Task 4 majority-class baseline", majority_model, X_test, y_test
    )

    print("\nTraining Logistic Regression (default)...")
    default_model.fit(X_train, y_train)
    default_metrics, default_cm, _, _ = evaluate_model(
        "Logistic Regression (default)", default_model, X_test, y_test
    )

    print("\nTraining Logistic Regression (class_weight='balanced')...")
    balanced_model.fit(X_train, y_train)
    balanced_metrics, balanced_cm, _, _ = evaluate_model(
        "Logistic Regression (balanced)", balanced_model, X_test, y_test
    )

    results = {
        "majority": majority_metrics,
        "default": default_metrics,
        "balanced": balanced_metrics,
    }
    matrices = {
        "majority": majority_cm,
        "default": default_cm,
        "balanced": balanced_cm,
    }

    # Save comparison table.
    metrics_df = pd.DataFrame([
        {
            "Model": "Task 4 majority baseline (test set)",
            **{k: v for k, v in majority_metrics.items() if k != "Model"},
        },
        {
            "Model": "Logistic Regression (default)",
            **{k: v for k, v in default_metrics.items() if k != "Model"},
        },
        {
            "Model": "Logistic Regression (balanced)",
            **{k: v for k, v in balanced_metrics.items() if k != "Model"},
        },
    ])
    metrics_df.to_csv(output_dir / "classification_metrics.csv", index=False)

    # Save confusion matrices as a CSV for transparent reporting.
    matrix_rows = []
    for model_name, matrix in matrices.items():
        matrix_rows.append({
            "Model": model_name,
            "TN": int(matrix[0, 0]),
            "FP": int(matrix[0, 1]),
            "FN": int(matrix[1, 0]),
            "TP": int(matrix[1, 1]),
        })
    pd.DataFrame(matrix_rows).to_csv(
        output_dir / "confusion_matrices.csv", index=False
    )

    # Save a confusion-matrix image for each Task 4 model.
    for model_name, matrix in matrices.items():
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["No (Stayed)", "Yes (Churned)"],
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(model_name.replace("_", " ").title())
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").replace("'", "")
        fig.savefig(
            output_dir / f"confusion_matrix_{safe_name}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Generate the full Markdown report with the actual results from this run.
    report_path = Path("MODEL_COMPARISON.md")
    create_markdown_report(
        report_path=report_path,
        source=source,
        row_count=len(cleaned_df),
        blank_total_charges=blank_total_charges,
        missing_total_charges=missing_total_charges,
        train_counts=train_counts,
        test_counts=test_counts,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        results=results,
        matrices=matrices,
    )

    print("\nFinal metrics:")
    print(metrics_df.round(4).to_string(index=False))
    print("\nSaved files:")
    print(f"- {report_path.resolve()}")
    print(f"- {(output_dir / 'classification_metrics.csv').resolve()}")
    print(f"- {(output_dir / 'confusion_matrices.csv').resolve()}")
    print(f"- Confusion matrix images in: {output_dir.resolve()}")
    print("\nTask 4 pipeline completed successfully.")


if __name__ == "__main__":
    main()
