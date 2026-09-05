from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def add_business_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"].fillna(0),
            bins=[-1, 6, 12, 24, 48, 1000],
            labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "48+ months"],
            right=False,
        )

    if {"MonthlyCharges", "TotalCharges"}.issubset(df.columns):
        df["avg_monthly_spend"] = np.where(
            df["tenure"].fillna(0) > 0,
            df["TotalCharges"].fillna(0) / df["tenure"].fillna(1),
            df["MonthlyCharges"].fillna(0),
        )

    service_columns = [
        "PhoneService",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    active_service_columns = []
    for column in service_columns:
        if column in df.columns:
            active_service_columns.append(column)
    if active_service_columns:
        df["total_services_used"] = df[active_service_columns].apply(
            lambda row: sum(
                1
                for value in row
                if isinstance(value, str) and value.lower() in {"yes", "true", "online", "fiber optic", "dsl"}
                or (pd.api.types.is_numeric_dtype(type(value)) and value == 1)
                or (isinstance(value, (int, float)) and value > 0)
            ),
            axis=1,
        )

    if "TechSupport" in df.columns:
        df["support_dependency"] = np.where(df["TechSupport"].astype(str).str.lower().isin(["no", "not available"]), 1, 0)

    if "Contract" in df.columns:
        df["contract_risk"] = df["Contract"].astype(str).str.lower().map(
            {
                "month-to-month": 1,
                "one year": 0.5,
                "two year": 0.1,
            }
        ).fillna(0.5)

    if "PaymentMethod" in df.columns:
        df["payment_risk"] = df["PaymentMethod"].astype(str).str.lower().map(
            {
                "electronic check": 1.0,
                "mailed check": 0.5,
                "bank transfer (automatic)": 0.3,
                "credit card (automatic)": 0.2,
            }
        ).fillna(0.5)

    if active_service_columns:
        df["service_engagement_score"] = df[active_service_columns].apply(
            lambda row: (
                sum(1 for value in row if isinstance(value, str) and value.lower() in {"yes", "fiber optic", "dsl"})
                + sum(1 for value in row if isinstance(value, (int, float)) and value > 0)
            )
            / max(len(active_service_columns), 1),
            axis=1,
        )

    if "MonthlyCharges" in df.columns:
        df["monthly_charge_risk"] = np.where(df["MonthlyCharges"].fillna(0) > df["MonthlyCharges"].median(), 1, 0)

    return df


def explain_feature_meanings() -> Dict[str, str]:
    return {
        "avg_monthly_spend": "Average monthly spend adjusted for tenure, helping identify customers whose spend is unusually high relative to time on service.",
        "tenure_group": "Customer tenure bucket; short tenures often signal higher churn risk and stronger need for early engagement.",
        "total_services_used": "Number of active service products a customer uses; lower engagement can signal lower product stickiness.",
        "support_dependency": "Indicator for whether the customer lacks technical support, which may correlate with frustration or service dissatisfaction.",
        "contract_risk": "Risk score based on contract type, where month-to-month arrangements are often more churn-prone than longer commitments.",
        "payment_risk": "Risk score based on payment method, where electronic checks may be associated with weaker retention patterns.",
        "service_engagement_score": "Composite measure of how broadly a customer is using the offered digital services and support features.",
        "monthly_charge_risk": "Indicator that monthly charges exceed the median, which may be relevant in pricing-sensitive churn scenarios.",
    }
