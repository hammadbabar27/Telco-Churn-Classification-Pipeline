# Task 4 — Baseline Classification Pipeline (Telco Customer Churn)

## 1. Objective

Build and evaluate a Logistic Regression classification pipeline that predicts whether a telecom customer will churn (`Churn = Yes`). The task continues the Task 2 Telco Customer Churn EDA and majority-class baseline.

## 2. Dataset and data preparation

- **Dataset rows used:** 7,043
- **Dataset source:** Telco-Customer-Churn.csv
- **Target:** `Churn` (`No` / `Yes`)
- **Rows with blank `TotalCharges` strings before conversion:** 11
- **Rows with missing `TotalCharges` after numeric conversion:** 11 (imputed inside the numeric pipeline)
- **Identifier removed:** `customerID`
- **Numeric features:** 4
- **Categorical features:** 15

`TotalCharges` was converted to numeric using `pd.to_numeric(..., errors="coerce")`. The blank strings become missing values and are handled by the numeric median imputer inside the pipeline. The special category values such as `No internet service` and `No phone service` were retained as valid categorical values; they were not treated as missing data.

## 3. Validation strategy

The data was divided into training and test sets using an **80/20 split**, `random_state=42`, and `stratify=y`.

| Split | No (Stayed) | Yes (Churned) | Total |
|---|---:|---:|---:|
| Training | 4,139 | 1,495 | 5,634 |
| Testing | 1,035 | 374 | 1,409 |

Stratification preserves approximately the same churn/non-churn proportions in both sets.

All learned preprocessing was placed inside a scikit-learn `Pipeline` and fitted only on the training data:
- Numeric columns: median imputation followed by `StandardScaler`.
- Categorical columns: most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
- `customerID` was excluded because it is only an identifier.

## 4. Model comparison

The positive class for precision, recall and F1 is **`Churn = Yes`**.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Task 2 historical majority baseline (full dataset) | 73.46% | 0.00% | 0.00% | 0.00% | Not calculated in Task 2 |
| Task 4 majority baseline (stratified test set) | 73.46% | 0.00% | 0.00% | 0.00% | 0.5000 |
| Logistic Regression (default) | 80.55% | 65.72% | 55.88% | 60.40% | 0.8419 |
| Logistic Regression (balanced) | 73.81% | 50.43% | 78.34% | 61.36% | 0.8413 |

**Baseline comparison note:** Task 2 calculated its majority-class baseline on the full dataset before introducing train/test validation. Its original results are retained above for continuity. The Task 4 majority baseline is also evaluated on the same stratified test set as the Logistic Regression models, making that row the fair like-for-like comparison. Task 2 did not calculate ROC-AUC, so it is marked as not calculated rather than inventing a historical value.

## 5. Confusion matrices

Rows represent actual classes and columns represent predicted classes. The positive class is `Yes`.

### Task 2 historical majority-class baseline (full dataset)

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 5,174 | 0 |
| Actual Yes | 1,869 | 0 |

### Task 4 majority-class baseline (same test set)

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 1,035 | 0 |
| Actual Yes | 374 | 0 |

### Logistic Regression — default class weighting

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 926 | 109 |
| Actual Yes | 165 | 209 |

### Logistic Regression — `class_weight="balanced"`

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 747 | 288 |
| Actual Yes | 81 | 293 |

### Business meaning of the four cells

- **True Negative (TN):** A customer stayed, and the model correctly predicted `No`.
- **False Positive (FP):** A customer stayed, but the model predicted `Yes`. This is a false alarm and may cause an unnecessary retention offer.
- **False Negative (FN):** A customer churned, but the model predicted `No`. This is a missed churner and may mean the business loses a customer without attempting retention.
- **True Positive (TP):** A customer churned, and the model correctly predicted `Yes`.

## 6. Business interpretation and recommendation

**Recommendation: Logistic Regression with `class_weight='balanced'`.** Its churn recall is 78.34%, compared with 55.88% for the default model. This means it identifies a larger share of customers who actually churn. The trade-off is precision (50.43% balanced vs. 65.72% default), so the retention team may receive more false alarms. Given that missing a churner is likely more costly than contacting a customer who would have stayed, the higher-recall model is the stronger starting point.

The default Logistic Regression model uses the standard class-weighting behavior. With `class_weight="balanced"`, scikit-learn automatically assigns greater training weight to the less frequent class. It does not add new churn examples or change the test-set distribution.

Accuracy alone is not enough for this problem. Recall measures the share of actual churners found by the model, while precision measures how many customers flagged as churn risks really churned. ROC-AUC measures how well the model ranks positive cases above negative cases across possible thresholds; it does not tell us the precision/recall trade-off at the default 0.5 threshold. Therefore, ROC-AUC should be interpreted alongside the confusion matrix, precision, recall and F1.

## 7. Threshold awareness and limitations

Both Logistic Regression models use the default decision threshold of **0.50**. A lower threshold may identify more potential churners, but it can also increase false alarms. Any threshold adjustment should be selected using validation data and a clear business cost trade-off—not by repeatedly tuning against the final test set.

This is a baseline model, not a production-ready churn system. Before deployment, the business should validate performance on newer customer data, decide how many retention contacts it can handle, and monitor model performance over time.

## 8. Conclusion

This task demonstrates a leakage-aware classification workflow: stratified splitting, training-only preprocessing, Logistic Regression, explicit class-imbalance handling, and evaluation using multiple metrics. The final model choice should prioritize the business cost of missed churners while keeping false alarms at a manageable level.
