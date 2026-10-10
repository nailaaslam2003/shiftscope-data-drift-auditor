# ShiftScope 🔎 — Explainable Data Drift Auditor

**A practical data-quality tool that compares a baseline dataset with a newer dataset and explains which features may have changed.**

ShiftScope is an open-source portfolio project built with Python. It supports CSV uploads, statistical comparisons, distribution visualizations, missing-value checks, and downloadable audit reports.

## Why it matters
Data collected over time can change: customer behaviour shifts, sensors drift, data pipelines break, and machine-learning models encounter inputs unlike their training data. ShiftScope helps analysts identify changes worth investigating before those changes cause downstream problems.

## Features
- Compare two CSV datasets using shared columns
- Check schema differences and missing-value changes
- Compare numeric distributions with the two-sample Kolmogorov–Smirnov test
- Compare categorical distributions with a chi-square test when feasible
- Calculate Population Stability Index (PSI) as a practical shift indicator
- Visualize numeric and categorical distributions interactively
- Explain flagged changes in plain language
- Export a feature-by-feature audit report
- Built-in fictional demo data; no upload required to try it

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Choose **Use built-in demo** or select **Upload two CSV files** in the sidebar.

## CSV requirements
Upload a baseline/reference CSV and a current CSV. Columns with matching names are compared. A numeric column should be numeric in both files; otherwise it is treated as categorical.

## Methods and interpretation
- **KS test:** checks whether two numeric samples may come from different distributions.
- **Chi-square test:** checks whether categorical frequency distributions differ, where the test is feasible.
- **PSI:** summarizes differences between binned numeric distributions or category proportions. Values around 0.10 and 0.25 are commonly used as moderate/high-shift heuristics, but these are not universal thresholds.
- **Missingness change:** shows the percentage-point difference in missing values between datasets.

A small p-value is not proof of practical importance, and a large p-value does not prove that there is no change. Sample size, feature dependence, multiple testing, and domain context matter. The app does not apply a multiple-testing correction and should be used as an investigation aid, not as an automated decision-maker.

## Project structure

```text
shiftscope-data-drift-auditor/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── reference_sample.csv
    └── current_sample.csv
```

## Technologies
Python · Pandas · NumPy · SciPy · Plotly · Streamlit

## Privacy
The included sample data is fictional. If you deploy this app publicly, do not upload private, sensitive, or identifying datasets. Review the hosting platform's data-handling policies before using real data.

## Roadmap
- Add bootstrap confidence intervals for distribution differences
- Add false-discovery-rate correction for many simultaneous feature tests
- Add configurable business-relevant thresholds
- Add command-line mode and automated unit tests
- Add time-window monitoring for streaming data

## Author
**Naila Aslam** — Mathematics, computational modelling, scientific programming, and data analysis.

GitHub: https://github.com/nailaaslam2003
