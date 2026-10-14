# Diabeta AI

Diabeta AI is an educational Streamlit application for exploring diabetes classification with Random Forest and XGBoost. It trains both models from a local CSV dataset, compares their performance, displays model feature importances, and lets a user submit clinical measurements for side-by-side predictions.

> **Medical disclaimer:** This is an academic demonstration, not a diagnostic or clinical decision-support tool. Do not use its predictions to make medical decisions.

## Features

- Dashboard with dataset counts and held-out model ROC-AUC scores.
- Interactive prediction form that displays each model's predicted class and positive-class probability.
- Model analytics: accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and ROC curves.
- Side-by-side feature-importance visualizations for both models.
- Dataset preview, descriptive statistics, and exploratory charts.
- Responsive Streamlit interface with Plotly charts.

## Models and training

At application startup, `app.py` loads the data and trains both classifiers. The displayed results and predictions are computed from the fitted models; prediction values are not hardcoded.

The application selects these seven input features:

1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `Insulin`
5. `BMI`
6. `DiabetesPedigreeFunction`
7. `Age`

`SkinThickness` is present in the CSV but is not used by the models. `Outcome` is the binary target (`0` or `1`).

Preprocessing and training implemented in `app.py`:

1. Zero measurements in `Glucose`, `BloodPressure`, `Insulin`, and `BMI` are treated as missing values.
2. Missing values in those columns are replaced with their respective medians.
3. The data is split into 80% training and 20% test partitions, stratified by `Outcome`, with `random_state=42`.
4. A Random Forest is fitted with 300 trees and `class_weight="balanced"`.
5. XGBoost is fitted with 300 estimators, depth 4, learning rate 0.05, subsample 0.85, and column subsample 0.85.
6. Both models are evaluated on the same held-out test partition.

The current implementation calculates medians before splitting the data. This means the test partition contributes to the imputation values; this is a small preprocessing leakage limitation to address before treating evaluation scores as rigorous research results. The current evaluation is a single split, not cross-validation.

## Dataset

The application reads **`data/diabetes.csv`**, using a path relative to the working directory. Launch the app from the project root so this path resolves correctly. The supplied dataset has 768 rows and 9 columns: eight predictor columns and the `Outcome` target. The model uses the seven features listed above.

The root-level `diabetes.csv` is also included as supplied, but the application does not read that copy. Keep `data/diabetes.csv` in place when running the app. The CSV's source and redistribution terms should be verified independently before redistributing the dataset.

## Project layout

```text
Project/
├── app.py
├── README.md
├── requirements.txt
├── run_project.bat
├── diabetes.csv              # Additional supplied copy; not loaded by the app
├── data/
│   └── diabetes.csv          # Dataset loaded by app.py
└── models/
    └── README.txt            # Notes that models train on application startup
```

## Requirements

- Python 3.10 or later is recommended.
- Packages and minimum versions are listed in `requirements.txt`:
  - Streamlit
  - pandas
  - NumPy
  - scikit-learn
  - XGBoost
  - Plotly

## Run locally

Open a terminal in the `Project` directory, then create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell prevents script activation, use Command Prompt instead:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, normally <http://localhost:8501>. Open that address in a browser. The included `run_project.bat` provides an alternate startup path, but the commands above make each setup step explicit and report failures more clearly.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Using the application

Choose a page from the sidebar:

- **Dashboard** — dataset size, selected-feature count, model ROC-AUC, class distribution, and metric comparison.
- **Prediction** — enter the seven model inputs and submit the form. The app shows each model's predicted class and probability for `Outcome = 1`.
- **Model Analytics** — test-set metrics, confusion matrices, and ROC curves.
- **Feature Importance** — compare the fitted estimators' built-in feature-importance values.
- **Dataset Explorer** — inspect a data preview, missing-cell count, summary statistics, and charts.
- **About** — project objective and technology summary.

The prediction form's initial values are demo defaults, not a representative patient profile. A displayed probability is the classifier's output and is not necessarily a calibrated estimate of an individual's real-world risk.

## Reproducing the current evaluation

The training/test split is deterministic for a fixed dataset and compatible dependency versions because `random_state=42` is set. On the supplied dataset, one run produced approximately:

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.753 | 0.760 |
| Precision | 0.633 | 0.660 |
| Recall | 0.704 | 0.648 |
| F1 score | 0.667 | 0.654 |
| ROC-AUC | 0.825 | 0.829 |

These are observed results for the current implementation, not guaranteed benchmarks. They can vary with dependency versions, dataset contents, and execution environment. The app calculates the metrics on startup rather than loading these table values.

## Limitations and responsible use

- The dataset is small and the evaluation uses one stratified hold-out split.
- Imputation currently happens before the train/test split; a production-quality evaluation should fit preprocessing only on training data, ideally through a scikit-learn pipeline.
- The project uses a fixed set of selected variables and does not validate input values as a clinical workflow.
- Built-in feature importance describes how the trained model uses its inputs. It does not establish causation or explain an individual prediction.
- The classifiers are not shown to be calibrated or externally validated for real-world clinical use.
- Do not submit identifiable or sensitive patient data to this demonstration.

## Possible extensions

- Put imputation and model fitting in a leakage-safe scikit-learn pipeline.
- Add cross-validation and external validation.
- Add calibration and threshold analysis.
- Add local explanation methods such as SHAP, with appropriate caveats.
- Add input validation, automated tests, and pinned/reproducible dependency versions.

