from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


def calculate_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.8) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        metrics = calculate_classification_metrics(y_true, y_prob, threshold=threshold)
        score = metrics["recall"] - abs(metrics["recall"] - target_recall) * 0.5 + metrics["f1"] * 0.2
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def threshold_tradeoff_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = calculate_classification_metrics(y_true, y_prob, threshold=threshold)
        rows.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
        })
    return pd.DataFrame(rows)


def summarize_model_results(model_results: pd.DataFrame) -> pd.DataFrame:
    return model_results.sort_values(["ROC_AUC", "F1", "Recall"], ascending=False).reset_index(drop=True)
