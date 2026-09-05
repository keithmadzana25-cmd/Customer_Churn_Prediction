from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGET_CHOICES = {
    "churn": "Churn",
    "churn_flag": "Churn",
    "attrition": "Churn",
    "exited": "Churn",
    "is_churn": "Churn",
    "target": "Churn",
    "customer_churn": "Churn",
}


def _normalize_column_name(column_name: Any) -> str:
    return str(column_name).strip().lower().replace(" ", "_").replace("-", "_")


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.replace({"": np.nan, " ": np.nan, "NA": np.nan, "N/A": np.nan, "nan": np.nan})
    try:
        return pd.to_numeric(cleaned, errors="coerce")
    except Exception:
        return cleaned


def infer_target_column(data: pd.DataFrame, target_col: Optional[str] = None) -> str:
    if target_col and target_col in data.columns:
        return target_col

    normalized_map = {_normalize_column_name(col): col for col in data.columns}
    for candidate in DEFAULT_TARGET_CHOICES:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for column in data.columns:
        normalized = _normalize_column_name(column)
        if "churn" in normalized or "target" in normalized or "attrition" in normalized or "exit" in normalized:
            return column

    raise ValueError(
        "The target column could not be inferred automatically. Please provide target_col explicitly."
    )


def identify_feature_columns(
    data: pd.DataFrame,
    target_col: str,
    feature_mapping: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    if feature_mapping:
        mapped = feature_mapping.get("features") or feature_mapping.get("feature_columns") or []
        if mapped:
            available = [col for col in mapped if col in data.columns]
            if available:
                return [col for col in available if col != target_col]

    excluded = {target_col}
    return [column for column in data.columns if column not in excluded]


def encode_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(series.dropna().unique().tolist())
        if set(unique_values).issubset({0, 1}):
            return series.astype(int)
        return series

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "active": 1,
        "churn": 1,
        "churned": 1,
        "left": 1,
        "left_company": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "inactive": 0,
        "stay": 0,
        "retained": 0,
        "not_churn": 0,
    }
    encoded = cleaned.map(mapping)
    if encoded.isna().any():
        remaining = cleaned[encoded.isna()].unique()
        raise ValueError(f"Unsupported target labels found: {list(remaining)}")
    return encoded.astype(int)


def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    constant_columns = []
    for column in df.columns:
        if df[column].nunique(dropna=True) <= 1:
            constant_columns.append(column)
    return constant_columns


def sanitize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.replace({"": np.nan, " ": np.nan, "?": np.nan})

    for column in df.columns:
        if column.lower() in {"customerid", "customer_id", "customer"}:
            continue
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].replace({"nan": np.nan, "None": np.nan, "null": np.nan})

    numeric_candidates = []
    for column in df.columns:
        if column.lower() in {"customerid", "customer_id", "customer"}:
            continue
        if df[column].dropna().astype(str).str.contains(r"^\d+(\.\d+)?$", regex=True).all():
            numeric_candidates.append(column)
    for column in numeric_candidates:
        df[column] = _safe_to_numeric(df[column])

    return df


def identify_column_types(data: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    feature_columns = [column for column in data.columns if column != target_col]
    numeric_columns = []
    categorical_columns = []
    for column in feature_columns:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return feature_columns, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def preprocess_data(
    data: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_mapping: Optional[Dict[str, List[str]]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_constant_columns: bool = True,
) -> Dict[str, Any]:
    if data is None or data.empty:
        raise ValueError("Input data is empty or missing.")

    df = sanitize_dataframe(data.copy())
    inferred_target = infer_target_column(df, target_col)

    if inferred_target not in df.columns:
        raise ValueError(f"Target column '{inferred_target}' not found in the provided dataset.")

    if df.duplicated().any():
        df = df.drop_duplicates().reset_index(drop=True)

    constant_columns = detect_constant_columns(df.drop(columns=[inferred_target])) if drop_constant_columns else []
    if constant_columns:
        df = df.drop(columns=constant_columns)

    feature_columns = identify_feature_columns(df, inferred_target, feature_mapping)
    if not feature_columns:
        raise ValueError("No feature columns were found after preprocessing.")

    if inferred_target not in df.columns:
        raise ValueError(f"Target column '{inferred_target}' not present after the preprocessing step.")

    X = df[feature_columns].copy()
    y = encode_target(df[inferred_target]).copy()

    if y.nunique() < 2:
        raise ValueError("The target variable has fewer than two unique values after encoding.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_columns_train = list(X_train.columns)
    _, numeric_columns, categorical_columns = identify_column_types(X_train, target_col="__unused__")
    numeric_columns = [column for column in feature_columns_train if pd.api.types.is_numeric_dtype(X_train[column])]
    categorical_columns = [column for column in feature_columns_train if column not in numeric_columns]

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "df": df,
        "feature_columns": feature_columns_train,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
        "target_col": inferred_target,
        "y_original": y,
        "X_raw_train": X_train,
        "X_raw_test": X_test,
    }


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(preprocessor, output_path)


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    import joblib

    return joblib.load(path)


def summarize_dataset(data: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    df = sanitize_dataframe(data.copy())
    target = infer_target_column(df, target_col)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[target].value_counts(dropna=False).to_dict(),
    }


def export_training_metadata(metadata: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, default=str)
