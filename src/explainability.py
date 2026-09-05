from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import shap


def risk_category(probability: float, low_threshold: float = 0.3, high_threshold: float = 0.7) -> str:
    if probability < low_threshold:
        return "LOW"
    if probability < high_threshold:
        return "MEDIUM"
    return "HIGH"


def explain_global_features(model: Any, X_transformed: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    mean_abs = np.abs(values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "feature_importance": feature_importance,
        "shap_values": shap_values,
    }


def explain_individual_customer(
    model: Any,
    preprocessor: Any,
    customer_record: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    transformed = preprocessor.transform(customer_record)
    probability = float(model.predict_proba(transformed)[0, 1])
    category = risk_category(probability)

    feature_names = preprocessor.get_feature_names_out().tolist()
    explainer = shap.Explainer(model, transformed)
    shap_values = explainer(transformed)

    values = np.array(shap_values.values)[0]
    sorted_idx = np.argsort(np.abs(values))[::-1]
    top_positive = []
    top_negative = []

    for idx in sorted_idx[:5]:
        if values[idx] > 0:
            top_positive.append((feature_names[idx], float(values[idx])))
        else:
            top_negative.append((feature_names[idx], float(values[idx])))

    positive_desc = [
        _humanize_feature(name, value)
        for name, value in top_positive[:3]
    ]
    negative_desc = [
        _humanize_feature(name, value, positive=False)
        for name, value in top_negative[:3]
    ]

    return {
        "probability": probability,
        "category": category,
        "risk_level": category,
        "prediction": "Churn" if probability >= threshold else "Retain",
        "top_risk_factors": positive_desc,
        "protective_factors": negative_desc,
        "shap_values": shap_values,
    }


def _humanize_feature(name: str, value: float, positive: bool = True) -> str:
    label = name.replace("num__", "").replace("cat__", "")
    if "Contract_" in label:
        contract_name = label.split("Contract_")[-1].replace("_", " ")
        if positive:
            return f"The {contract_name} contract is increasing the customer's churn risk."
        return f"The {contract_name} contract is reducing churn risk for this customer."
    if "InternetService_" in label:
        service = label.split("InternetService_")[-1].replace("_", " ")
        if positive:
            return f"Internet service type '{service}' is associated with elevated churn risk."
        return f"Internet service type '{service}' is helping reduce churn risk."
    if "PaymentMethod_" in label:
        payment = label.split("PaymentMethod_")[-1].replace("_", " ")
        if positive:
            return f"Payment method '{payment}' is contributing to a higher churn estimate."
        return f"Payment method '{payment}' is acting as a protective factor."
    if "TechSupport_" in label:
        tech = label.split("TechSupport_")[-1].replace("_", " ")
        if positive:
            return f"Lack of technical support ({tech}) is increasing churn risk."
        return f"Technical support access ({tech}) is helping reduce churn risk."
    if "tenure" in label.lower():
        if positive:
            return "Short customer tenure is increasing the estimated churn risk."
        return "Longer tenure is lowering the estimated churn risk."
    if "MonthlyCharges" in label:
        if positive:
            return "Higher monthly charges are increasing churn risk."
        return "Lower monthly charges are reducing churn risk."
    if positive:
        return f"Feature '{label}' is contributing to a higher churn estimate."
    return f"Feature '{label}' is acting as a protective factor."


def summarize_feature_importance(feature_importance: Iterable[Tuple[str, float]]) -> List[Dict[str, Any]]:
    output = []
    for feature, value in feature_importance:
        output.append({"feature": feature, "importance": float(value)})
    return sorted(output, key=lambda item: item["importance"], reverse=True)
