# Task 4 — Baseline Classification Pipeline (Telco Churn)

This project continues Task 2 using the same IBM Telco Customer Churn dataset.

## Files

- `classification_baseline.py` — complete classification workflow.
- `MODEL_COMPARISON.md` — generated automatically after running the script; includes the actual metrics, confusion matrices and business recommendation.
- `requirements.txt` — required Python libraries.
- `outputs/` — created automatically; contains metric CSVs, confusion-matrix CSV and PNG images.

## Dataset

Original Kaggle source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The script first looks for one of these local filenames:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `Telco-Customer-Churn.csv`
- `customer_churn.csv`
- `telco.csv`

If no local file is found, it tries the same public IBM CSV mirror used in Task 2.

## Run in Google Colab

1. Upload `classification_baseline.py` and the Task 2 CSV into the same Colab session.
2. Install the libraries:

   ```python
   !pip install -r requirements.txt
   ```

3. Run the script:

   ```python
   !python classification_baseline.py
   ```

4. After it finishes, open `MODEL_COMPARISON.md` and inspect the `outputs/` folder.
5. Download the generated report, CSV files and confusion-matrix images if you want to include them in GitHub.

If you use a Colab notebook instead of uploading a requirements file, install directly:

```python
!pip install pandas numpy scikit-learn matplotlib
```

## Run locally in VS Code

```bash
pip install -r requirements.txt
python classification_baseline.py
```

## What the script does

- Converts `TotalCharges` from text to numeric. Blank strings become missing values.
- Removes `customerID` from model features because it is an identifier.
- Uses an 80/20 split with `stratify=y` and `random_state=42`.
- Fits numeric imputation and scaling, and categorical imputation and one-hot encoding, inside each model pipeline.
- Compares a same-test-set `DummyClassifier`, default Logistic Regression, and Logistic Regression with `class_weight="balanced"`.
- Reports accuracy, precision, recall, F1 and ROC-AUC for `Churn = Yes`.
- Keeps the original Task 2 full-dataset majority baseline in the report for historical comparison. It also reports a new majority baseline on the Task 4 test set so the model comparison is fair.

## Important

The script generates the actual `MODEL_COMPARISON.md` from the metrics produced when you run it. Do not submit estimated or copied metrics: run the script on the same Telco CSV first and use the generated report.
