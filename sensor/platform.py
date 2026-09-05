"""Leakage-safe local training and inference primitives."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(frame: pd.DataFrame, target: str = "class", random_state: int = RANDOM_STATE) -> SplitData:
    """Make deterministic, stratified train/validation/final-test splits."""
    if target not in frame or frame[target].nunique() != 2:
        raise ValueError("Target must be a binary column present in the dataset.")
    train_val, test = train_test_split(frame, test_size=0.20, stratify=frame[target], random_state=random_state)
    train, validation = train_test_split(train_val, test_size=0.25, stratify=train_val[target], random_state=random_state)
    return SplitData(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True))


def validate_frame(frame: pd.DataFrame, target: str = "class", required_features: list[str] | None = None) -> dict[str, Any]:
    required = set(required_features or [])
    missing = sorted(required - set(frame.columns))
    unexpected = sorted(set(frame.columns) - required - {target}) if required_features else []
    target_ok = target in frame and frame[target].dropna().nunique() == 2
    features = frame.drop(columns=[target], errors="ignore")
    report = {
        "valid": not missing and target_ok and not features.empty,
        "missing_required": missing, "unexpected_columns": unexpected,
        "target_present_and_binary": target_ok,
        "all_null_columns": features.columns[features.isna().all()].tolist(),
        "zero_variance_columns": features.columns[features.nunique(dropna=True).le(1)].tolist(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_fraction": {k: float(v) for k, v in frame.isna().mean().items()},
        "non_numeric_features": features.select_dtypes(exclude=[np.number]).columns.tolist(),
    }
    report["valid"] = report["valid"] and not report["all_null_columns"] and not report["non_numeric_features"]
    return report


def drop_zero_std_columns(frame: pd.DataFrame, target: str = "class") -> tuple[pd.DataFrame, list[str]]:
    columns = [c for c in frame.columns if c != target and frame[c].nunique(dropna=True) <= 1]
    return frame.drop(columns=columns), columns


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> list[dict[str, Any]]:
    """KS drift for numeric features only, with Benjamini-Hochberg FDR correction."""
    rows: list[dict[str, Any]] = []
    for col in reference.select_dtypes(include=[np.number]).columns.intersection(current.columns):
        a, b = reference[col].dropna(), current[col].dropna()
        if len(a) < 2 or len(b) < 2 or a.nunique() <= 1 or b.nunique() <= 1:
            rows.append({"feature": col, "test": "ks_2samp", "statistic": None, "raw_p_value": None, "adjusted_p_value": None, "drift_detected": False, "reason": "insufficient_or_constant"})
            continue
        result = ks_2samp(a, b)
        rows.append({"feature": col, "test": "ks_2samp", "statistic": float(result.statistic), "raw_p_value": float(result.pvalue)})
    usable = sorted((r for r in rows if r.get("raw_p_value") is not None), key=lambda r: r["raw_p_value"])
    m = len(usable)
    adjusted = [min(1.0, r["raw_p_value"] * m / (i + 1)) for i, r in enumerate(usable)]
    for i in range(m - 2, -1, -1): adjusted[i] = min(adjusted[i], adjusted[i + 1])
    for row, p in zip(usable, adjusted): row.update(adjusted_p_value=p, drift_detected=p < alpha)
    return rows


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    return {"f1": float(f1_score(y, prediction, zero_division=0)), "precision": float(precision_score(y, prediction, zero_division=0)), "recall": float(recall_score(y, prediction, zero_division=0)), "roc_auc": float(roc_auc_score(y, probability)), "pr_auc": float(average_precision_score(y, probability)), "confusion_matrix": confusion_matrix(y, prediction).tolist()}


def choose_threshold(y: np.ndarray, probability: np.ndarray, false_negative_cost: float | None = None, false_positive_cost: float | None = None) -> float:
    if false_negative_cost is None or false_positive_cost is None: return 0.5
    thresholds = np.linspace(0.05, 0.95, 91)
    return float(min(thresholds, key=lambda t: false_negative_cost * ((y == 1) & (probability < t)).sum() + false_positive_cost * ((y == 0) & (probability >= t)).sum()))


def train_csv(path: str | Path, output_dir: str | Path = "artifacts/production", target: str = "class", random_state: int = RANDOM_STATE, false_negative_cost: float | None = None, false_positive_cost: float | None = None) -> dict[str, Any]:
    frame = pd.read_csv(path).replace("na", np.nan)
    validation = validate_frame(frame, target)
    if not validation["valid"]: raise ValueError(f"Invalid data: {json.dumps(validation)}")
    frame, dropped = drop_zero_std_columns(frame, target)
    splits = split_data(frame, target, random_state)
    features = [c for c in frame.columns if c != target]
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    x_train = preprocessor.fit_transform(splits.train[features])
    # Resampling belongs exclusively to training data after fitting the preprocessor.
    x_train, y_train = SMOTETomek(random_state=random_state).fit_resample(x_train, splits.train[target])
    x_val = preprocessor.transform(splits.validation[features]); y_val = splits.validation[target].to_numpy()
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1),
    }
    # Selection uses validation PR-AUC only; the test set is untouched until below.
    fitted = {name: model.fit(x_train, y_train) for name, model in candidates.items()}
    selected_name, selected = max(fitted.items(), key=lambda item: average_precision_score(y_val, item[1].predict_proba(x_val)[:, 1]))
    threshold = choose_threshold(y_val, selected.predict_proba(x_val)[:, 1], false_negative_cost, false_positive_cost)
    test_probability = selected.predict_proba(preprocessor.transform(splits.test[features]))[:, 1]
    test_metrics = metrics(splits.test[target].to_numpy(), test_probability, threshold)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{selected_name}-{timestamp}"
    metadata = {"model_version": model_version, "training_timestamp": timestamp, "algorithm": selected_name, "hyperparameters": selected.get_params(), "features": features, "dropped_zero_variance_columns": dropped, "threshold": threshold, "test_metrics": test_metrics, "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "split_sizes": {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)}}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": selected, "metadata": metadata}, output / "model.joblib")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
