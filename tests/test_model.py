import os

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import preprocess_data
from src.train_model import train_and_compare_models


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004", "C005", "C006"],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "tenure": [1, 5, 12, 18, 30, 42],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "DSL", "Fiber optic", "No", "DSL"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "One year"],
            "MonthlyCharges": [75.0, 55.5, 45.2, 85.0, 40.0, 50.0],
            "TotalCharges": [80.0, 260.0, 520.0, 1500.0, 1200.0, 2000.0],
            "Churn": ["Yes", "No", "No", "Yes", "No", "No"],
        }
    )


def test_preprocess_data_handles_missing_and_schema(sample_frame):
    processed = preprocess_data(sample_frame.copy(), target_col="Churn")
    assert "X_train" in processed
    assert "X_test" in processed
    assert "y_train" in processed
    assert "y_test" in processed
    assert processed["X_train"].shape[0] > 0
    assert processed["X_test"].shape[0] > 0
    assert set(processed["y_train"]) <= {0, 1}


def test_training_returns_model_results(sample_frame):
    results = train_and_compare_models(sample_frame.copy(), target_col="Churn", include_xgboost=False)
    assert "models" in results
    assert isinstance(results["comparison_table"], pd.DataFrame)
    assert not results["comparison_table"].empty


def test_prediction_probability_range(sample_frame):
    processed = preprocess_data(sample_frame.copy(), target_col="Churn")
    model = train_and_compare_models(sample_frame.copy(), target_col="Churn", include_xgboost=False)["best_model"]
    prob = model.predict_proba(processed["X_test"])[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))


def test_invalid_target_raises(sample_frame):
    with pytest.raises(ValueError):
        preprocess_data(sample_frame.copy().drop(columns=["Churn"]), target_col="Churn")