# Intelligent Data Quality & Anomaly Platform

A Streamlit application for exploring uploaded tabular data, identifying data quality issues, reviewing potential anomalies, applying selected cleaning operations, and retaining analysis history in PostgreSQL when configured.

## Implemented features

- Upload CSV, Excel (`.xlsx` and `.xls`), and JSON datasets.
- Profile row and column counts, data types, missing values, duplicates, and numerical statistics.
- Detect missing values, duplicates, invalid values, data type problems, format problems, inconsistent values, and statistical outliers.
- Select useful features, preprocess numeric and categorical columns, and run Isolation Forest and Local Outlier Factor (LOF).
- Review combined anomaly results. Anomalies are review suggestions and are not automatically removed by cleaning.
- Apply supported cleaning operations and download the cleaned dataset.
- Review a data quality score and download quality and anomaly reports for uploaded datasets.
- Optionally persist analyses, results, and applied cleaning actions in PostgreSQL and browse analysis history.
- Run a separate synthetic-data evaluation comparing IQR, Z-score, Isolation Forest, and LOF, with a downloadable CSV report.

## Technology

- Python, Streamlit, pandas, and NumPy
- scikit-learn for machine learning; NumPy and pandas for data operations (SciPy is also included in `requirements.txt`)
- PostgreSQL through Psycopg; `python-dotenv` reads local environment configuration
- `openpyxl` and `xlrd` for Excel formats
- pytest for the test suite (installed separately; it is not listed in `requirements.txt`)

The dependencies are listed in `requirements.txt`; most are version-pinned. PostgreSQL access in `modules/database.py` uses Psycopg directly.

## Repository structure

```text
.
├── app.py                         # Streamlit application and workflow orchestration
├── main.py                        # PyCharm starter stub; not the application entry point
├── requirements.txt
├── .env.example                   # Safe PostgreSQL configuration template
├── models/
│   ├── isolation_forest.py        # Isolation Forest model
│   └── lof.py                     # Local Outlier Factor model
├── modules/
│   ├── data_loader.py             # CSV, Excel, and JSON loading
│   ├── profiling.py               # Dataset and column profiling
│   ├── missing_values.py          # Missing-value checks
│   ├── duplicates.py              # Duplicate checks
│   ├── invalid_values.py          # Rule-based value checks
│   ├── data_type_validation.py    # Data type checks
│   ├── format_validation.py       # Common format checks
│   ├── consistency.py             # Inconsistent category checks
│   ├── statistical_outliers.py    # IQR detection
│   ├── feature_selection.py       # ML feature selection
│   ├── ml_preprocessing.py        # Upload-workflow preprocessing helpers
│   ├── preprocessing.py           # ML imputation, encoding, and scaling
│   ├── anomaly_pipeline.py        # Feature preparation and IF/LOF pipeline
│   ├── anomaly_analysis.py        # Combined anomaly interpretation
│   ├── anomaly_dashboard.py       # Anomaly summaries and tables
│   ├── cleaning.py                # Selected dataset cleaning operations
│   ├── quality_score.py           # Quality dimensions and overall score
│   ├── report_builder.py           # Uploaded-dataset quality/anomaly reports
│   ├── ml_evaluation.py            # Synthetic evaluation, metrics, and CSV report
│   ├── analysis_persistence.py     # Analysis identity and cleaning persistence
│   ├── database.py                 # PostgreSQL connection, schema, and queries
│   └── unified_dashboard.py        # Dataset findings and unified result helpers
└── tests/                          # Unit and integration-style pytest tests
```

The Phase 22 architecture in the project plan is approximate. This repository splits checks and workflow responsibilities across focused modules and currently has no `utils/`, `data/`, `reports/`, or `docs/` directory.

## Windows setup (PowerShell)

Run these commands from the project root. Python must be installed and available as `python` on `PATH`. The repository does not pin a Python version, so use a version compatible with the listed package versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

If PowerShell blocks virtual environment activation, allow scripts for the current PowerShell process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## PostgreSQL configuration (optional for analysis, required for saved history)

The database helper loads `.env` from the project root. Copy the safe template and edit `.env` locally:

```powershell
Copy-Item .env.example .env
notepad .env
```

Supported variables and code defaults:

| Variable | Default / requirement |
| --- | --- |
| `DB_HOST` | `localhost` |
| `DB_PORT` | `5432` |
| `DB_NAME` | `intelligent_data_quality` |
| `DB_USER` | `postgres` |
| `DB_PASSWORD` | Required; no default |

Never commit a real `.env` file or put credentials in this README. `.env` is ignored by Git. The database server must be running, the configured database must already exist, and the configured role must have permission to create tables and indexes. Initialize the project tables from the project root after configuring `.env`:

```powershell
python -m modules.database
```

That module tests the connection and then runs its `CREATE TABLE/INDEX IF NOT EXISTS` initialization. It does not create the PostgreSQL database itself. Database initialization and connectivity have not been verified as part of this documentation; configure them for your local PostgreSQL installation and check the command output. Analysis and downloads can be used without saving history, but PostgreSQL-backed history and persistence require a working database configuration.

## Run the Streamlit application

From the project root, with `.venv` active and requirements installed:

```powershell
python -m streamlit run app.py
```

`app.py` is the actual Streamlit entry point. `main.py` remains a PyCharm starter stub and does not launch the platform.

## Normal uploaded-dataset workflow

1. Upload a CSV, `.xlsx`, `.xls`, or JSON file.
2. Review the dataset profile and detected quality issues.
3. Review anomaly results from Isolation Forest and LOF, along with other findings.
4. Choose cleaning operations and apply them; potential anomalies are not automatically deleted.
5. Review the quality score and download the cleaned dataset or uploaded-dataset reports.
6. If PostgreSQL is configured, review saved analysis history and cleaning actions.

## Synthetic ML Evaluation

ML Evaluation generates its own reproducible synthetic dataset with known anomaly labels. It does **not** evaluate or modify the dataset uploaded in the normal workflow.

It compares four approaches:

- **IQR:** flags records with at least one numeric value outside that feature's 1.5×IQR bounds.
- **Z-score:** flags records with at least one numeric value beyond the configured standard-deviation threshold.
- **Isolation Forest:** multivariate tree-based detector.
- **LOF:** local density detector; it uses the shared robust-scaled, imputed feature matrix because it is distance-based.

The comparison displays detected counts, TP/FP/FN/TN, precision, recall, F1-score, and per-method execution time. The random seed, record count, known anomaly percentage, and Isolation Forest/LOF contamination setting are shown with the evaluation. IQR and Z-score use fixed rules; contamination applies only to Isolation Forest and LOF. Selecting the true anomaly rate gives those two detectors ground-truth prevalence information and should be interpreted as an oracle-assisted setting.

Per-method time is the mean detection-step time over repeated runs. Shared preprocessing time is shown separately and is excluded from those per-method timings. These timings are not full end-to-end times and can be affected by configuration and environment; they should be interpreted separately from detection quality. Results are measured on the generated batch and do not establish generalization to other datasets.

After running an evaluation, use **Download ML Evaluation Report** in that results section to download `ml_evaluation_report.csv`. The report is built from the already-computed result and includes one row per method with its metrics, settings, and shared preprocessing time. This synthetic evaluation export is separate from quality and anomaly reports for uploaded datasets.

## Tests

With the virtual environment active, run focused tests, for example:

```powershell
python -m pytest -q tests/test_ml_evaluation.py tests/test_isolation_forest.py tests/test_lof.py tests/test_preprocessing.py
```

Run the full test suite with:

```powershell
python -m pytest -q
```

PostgreSQL-related tests use mocks for database operations; running pytest does not require changing the live database. This documentation task did not run the tests.
