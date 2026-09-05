from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_business_features

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


DEFAULT_MODEL_CONFIG = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
        },
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    ),
    (
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42),
        {
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [1, 3, 5],
            "n_estimators": [200, 300],
        },
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3],
        },
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=42, max_depth=6),
        {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 6],
            "max_leaf_nodes": [15, 31],
        },
    ),
]


def _score_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    min_class_count = min(np.bincount(y_train).min(), 3)
    if min_class_count < 2:
        raise ValueError("Each class must have at least 2 examples for cross-validation scoring.")
    cv = StratifiedKFold(n_splits=min_class_count, shuffle=True, random_state=42)
    metrics = {
        "roc_auc": np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")),
        "f1": np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")),
        "recall": np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")),
    }
    return metrics


def _estimate_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No trained models were produced.")
    ranked = sorted(results, key=lambda item: (item["roc_auc"], item["f1"], item["recall"]), reverse=True)
    return ranked[0]


def train_and_compare_models(
    data: pd.DataFrame,
    target_col: str = "Churn",
    include_xgboost: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    if data is None or data.empty:
        raise ValueError("The dataset is empty.")

    processed = preprocess_data(data.copy(), target_col=target_col, random_state=random_state)

    X_train = processed["X_train"]
    X_test = processed["X_test"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]

    candidate_models = list(DEFAULT_MODEL_CONFIG)
    if include_xgboost and XGBClassifier is not None:
        candidate_models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
                {
                    "n_estimators": [200, 300],
                    "max_depth": [3, 6],
                    "learning_rate": [0.03, 0.05],
                },
            )
        )

    rows = []
    label_counts = np.bincount(y_train)
    cv_folds = min(3, int(np.min(label_counts))) if np.min(label_counts) >= 2 else None

    for model_name, estimator, param_grid in candidate_models:
        if cv_folds is not None:
            search = GridSearchCV(
                estimator,
                param_grid,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
        else:
            best_estimator = estimator
            best_params = {}
            best_estimator.fit(X_train, y_train)

        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "PR_AUC": average_precision_score(y_test, y_prob),
            "Best_Params": best_params,
            "Estimator": best_estimator,
        }
        rows.append(metrics)

    comparison_table = pd.DataFrame(rows).sort_values(["ROC_AUC", "F1", "Recall"], ascending=False).reset_index(drop=True)
    best_entry = _estimate_best_model(
        [{
            "Model": row["Model"],
            "roc_auc": row["ROC_AUC"],
            "f1": row["F1"],
            "recall": row["Recall"],
            "estimator": row["Estimator"],
        } for _, row in comparison_table.iterrows()]
    )

    return {
        "models": {row["Model"]: row["Estimator"] for _, row in comparison_table.iterrows()},
        "comparison_table": comparison_table,
        "best_model": best_entry["estimator"],
        "best_model_name": best_entry["Model"],
        "preprocessor": processed["preprocessor"],
        "feature_columns": processed["feature_columns"],
        "target_col": target_col,
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
    }


def save_model_bundle(model_bundle: Dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "best_model.pkl"
    preprocessor_path = output_path / "preprocessing_pipeline.pkl"
    metadata_path = output_path / "model_metadata.json"

    joblib.dump(model_bundle["best_model"], model_path)
    joblib.dump(model_bundle["preprocessor"], preprocessor_path)

    metadata = {
        "best_model_name": model_bundle["best_model_name"],
        "target_col": model_bundle["target_col"],
        "feature_columns": model_bundle["feature_columns"],
        "models_trained": list(model_bundle["models"].keys()),
        "comparison_table": model_bundle["comparison_table"].to_dict(orient="records"),
    }
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, default=str)


def load_model_bundle(model_dir: str | Path) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    return {
        "best_model": joblib.load(model_dir / "best_model.pkl"),
        "preprocessor": joblib.load(model_dir / "preprocessing_pipeline.pkl"),
        "metadata": json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8")),
    }


def train_and_save_model(data: pd.DataFrame, target_col: str = "Churn", output_dir: str | Path = "models") -> Dict[str, Any]:
    bundle = train_and_compare_models(data, target_col=target_col)
    save_model_bundle(bundle, output_dir)
    return bundle
