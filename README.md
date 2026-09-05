# Customer Churn Prediction with Explainable AI

## 1. Project overview
This project builds a production-quality customer churn prediction system that estimates whether a customer is likely to leave and explains why the model reached that conclusion. It is designed for portfolio use, academic projects, and demonstration in a real business context.

## 2. Problem statement
Customer retention teams need early, explainable insight into which customers are at risk of churn. Traditional reporting often shows aggregated churn rates but does not explain which customer behaviors or contract conditions are driving risk. This project combines machine learning and explainable AI to help answer three questions clearly:

1. Who is likely to churn?
2. How likely are they to churn?
3. Why does the model think they will churn?

## 3. Business motivation
Customer acquisition is usually more expensive than retention. Identifying churn risk in advance lets a business intervene with targeted offers, proactive support, or contract incentives. The system is designed to support retention decisions without making causal claims.

## 4. Objectives
- Load and validate customer churn datasets
- Clean and preprocess data without leakage
- Engineer business-oriented features
- Compare multiple classification models
- Evaluate results using churn-focused metrics
- Tune the decision threshold for business outcomes
- Explain predictions using SHAP-based and human-readable outputs
- Provide a Streamlit dashboard for prediction and exploration

## 5. Dataset description
The repository is built to work with common customer churn CSV files. A standard telecom-style churn dataset may contain fields such as:

- Customer ID
- Gender
- Senior citizen flag
- Partner / dependents
- Tenure
- Phone service
- Internet service
- Contract type
- Monthly charges
- Total charges
- Tech support
- Payment method
- Churn target

The target column is configurable and inferred automatically when possible.

## 6. Features
Key features commonly used in customer churn prediction include customer tenure, contract type, billing patterns, support usage, and digital service engagement. In this project, the pipeline handles both numerical and categorical data and prepares a robust preprocessing pipeline that is fitted only on training data.

## 7. Methodology
The project uses a typical machine learning lifecycle:

1. Inspect raw data and schema
2. Validate target and feature columns
3. Preprocess train/test split data
4. Engineer business features
5. Train multiple models
6. Compare performance with churn-focused metrics
7. Optimize threshold and select the best model
8. Explain predictions with SHAP
9. Package the model and pipeline for deployment

## 8. Data preprocessing
The preprocessing module manages:

- duplicate removal
- missing value handling
- categorical and numerical detection
- constant-column checks
- target encoding
- train/test splitting with stratification
- preprocessing pipeline construction with scikit-learn

A ColumnTransformer is used to apply median imputation and scaling to numeric columns and one-hot encoding to categorical columns.

## 9. Exploratory analysis
The project is designed to generate business insights using churn distributions, tenure analysis, contract comparisons, payment-method comparisons, and feature correlation checks. These reports can be expanded in the notebook workflow or the dashboard.

## 10. Feature engineering
Example business-oriented features include:

- tenure_group
- avg_monthly_spend
- total_services_used
- support_dependency
- contract_risk
- payment_risk
- service_engagement_score

These are features created for modeling and are not intended to imply causation. They help expose patterns linked to churn risk.

## 11. Machine learning models
The project compares several models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoostingClassifier
- XGBoost (when available)

Each model is evaluated using stratified cross-validation and then compared on a holdout test set.

## 12. Evaluation metrics
The model is evaluated using more than accuracy alone, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

For churn prediction, recall and F1 are important because missing a likely churner can be costly to the business.

## 13. Explainable AI methodology
The explainability module uses SHAP to generate:

- global feature importance rankings
- summary plots
- top churn-driving factors
- protective factors that reduce risk

Predictions are translated into business language such as:

- Short customer tenure is increasing churn risk.
- Customers on month-to-month contracts show a higher estimated likelihood of leaving.

This is explanatory, not causal proof.

## 14. System architecture
The repository follows a modular layout:

- src/data_preprocessing.py
- src/feature_engineering.py
- src/train_model.py
- src/evaluate_model.py
- src/explainability.py
- app/app.py

This allows clean reuse of data logic, model logic, and UI logic.

## 15. Project structure
```text
customer-churn-explainable-ai/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
├── src/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

## 16. Installation instructions
Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 17. How to run the application
From the project root:

```bash
streamlit run app/app.py
```

## 18. Example prediction
A user can enter a customer record in the dashboard and receive:

- customer churn probability
- risk category
- top risk drivers
- protective factors
- retention suggestions

## 19. Screenshots
Add screenshots here once the dashboard is running locally.

## 20. Limitations
- Model explanations are not causal evidence.
- Performance depends heavily on dataset quality and feature design.
- Without a real churn dataset, the project can only demonstrate a representative workflow.

## 21. Future improvements
- Add offline model retraining pipeline
- Integrate a proper event-driven dashboard backend
- Add more model types and calibration methods
- Build advanced retention recommendation rules
- Support file upload validation and schema-aware preprocessing

## 22. Ethical considerations
This project should be used responsibly. Predictions must not be used to discriminate against protected groups or to deny services without human review. The system is a decision-support tool, not an autonomous decision-maker.

## 23. License
This project is provided for educational and portfolio use. Add an appropriate open-source license such as MIT if needed.
