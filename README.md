# Loan Approval Prediction System

Capstone project for the AIML Summer Internship 2026, IIHMF, MNNIT Allahabad. The system predicts whether a loan application is likely to be approved, based on applicant and loan attributes, and is deployed as an interactive Streamlit app.

## Problem Statement

Manual loan approval decisions are slow and inconsistent across applicants. This project builds a classification model that predicts loan approval status from applicant demographics, income, and loan details, and deploys it through a guided web form.

## Dataset

Kaggle Loan Prediction Dataset (`Dataset/train.csv`) — 614 records with 13 attributes including gender, marital status, dependents, education, employment type, income, loan amount, loan term, credit history, and property area.

## Project Structure

```
LoanApprovalPrediction/
│
├── Dataset/
│   └── train.csv
├── Notebook/
│   └── loan_prediction.ipynb
├── Model/
│   ├── loan_model.pkl
│   └── scaler.pkl
├── Streamlit_App/
│   └── app.py
├── Documentation/
│   ├── Project_Report.pdf
│   └── Presentation.pptx
├── requirements.txt
└── README.md
```

## Methodology

1. **Data Preprocessing** — duplicate removal, missing value imputation (mode for categorical, median/mode for numeric), IQR-based outlier capping on income and loan amount, label encoding of categorical features.
2. **Exploratory Data Analysis** — univariate (distribution plots, boxplots), bivariate (credit history vs. approval, income vs. approval), and correlation heatmap.
3. **Feature Engineering** — `TotalIncome` (applicant + coapplicant income) and `Loan_Income_Ratio` (loan amount relative to income) were derived; the redundant raw income columns and `Loan_ID` were dropped.
4. **Model Building** — Logistic Regression, Decision Tree, and Random Forest classifiers were trained on an 80/20 stratified split with standardized features.
5. **Model Evaluation** — Accuracy, Precision, Recall, F1 Score, and ROC-AUC were compared across models; Random Forest was selected as the final model.
6. **Deployment** — The trained model and scaler are served through a multi-step Streamlit form that returns an approval decision, an approval score, an estimated EMI, and the key factors behind the prediction.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.854 | 0.832 | 0.988 | 0.903 | 0.854 |
| Decision Tree | 0.650 | 0.784 | 0.682 | 0.730 | 0.631 |
| Random Forest | 0.862 | 0.878 | 0.929 | 0.903 | 0.853 |

Random Forest was chosen for deployment as it gives the best accuracy and precision while keeping recall and F1 competitive with Logistic Regression.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Reproduce training (optional — the trained model is already included under `Model/`):

```bash
jupyter notebook Notebook/loan_prediction.ipynb
```

Run the app:

```bash
streamlit run Streamlit_App/app.py
```

## Submission

Team Lead: _add name here_
Team Members: _add name(s) here_
