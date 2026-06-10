"""
Data loading, cleaning, and preprocessing.
Handles the "dirty" Dry Bean dataset: missing values, type errors,
unit suffixes, and mislabeled classes.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import dump, load

from .config import SEED, TARGET_COL, EXPECTED_CLASSES

np.random.seed(SEED)


# ── Class name normalization ──────────────────────────────────────────────

def _normalize_class(cls: str) -> str:
    """Fix typos, case, and whitespace in a single class label."""
    s = str(cls).strip().upper()
    # Digit-for-letter substitutions (common OCR/entry errors)
    s = s.replace("0", "O").replace("3", "E").replace("1", "I").replace("5", "S")
    # Strip trailing/leading non-alpha
    s = re.sub(r"[^A-Z]", "", s)
    # Map known remnants to correct classes
    known = {
        "DERMASON": "DERMASON", "DERMASN": "DERMASON",
        "SIRA": "SIRA",
        "SEKER": "SEKER",
        "HOROZ": "HOROZ",
        "CALI": "CALI",
        "BARBUNYA": "BARBUNYA",
        "BOMBAY": "BOMBAY",
    }
    return known.get(s, s)


# ── Main loading & cleaning ───────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load a raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)

    # --- Clean Solidity (string -> float, "?" -> NaN) ---
    df["Solidity"] = pd.to_numeric(df["Solidity"], errors="coerce")

    # --- Clean Compactness (remove " cm" suffix, convert to float) ---
    def _fix_compactness(val):
        if isinstance(val, str):
            val = val.replace(" cm", "").replace("cm", "").strip()
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    df["Compactness"] = df["Compactness"].apply(_fix_compactness)

    # --- Clean class labels ---
    df[TARGET_COL] = df[TARGET_COL].apply(_normalize_class)

    # Drop rows with unrecognized classes (should not happen after normalization)
    df = df[df[TARGET_COL].isin(EXPECTED_CLASSES)]

    return df


# ── Preprocessing pipeline ────────────────────────────────────────────────

def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame,
               df_val: pd.DataFrame | None = None):
    """
    Fit scaler and imputer on training data, transform all sets.

    Returns:
        X_train, X_test, X_val, y_train, y_test, y_val, scaler, label_encoder
    """
    feature_cols = [c for c in df_train.columns if c != TARGET_COL]

    X_train_raw = df_train[feature_cols].copy()
    y_train_raw = df_train[TARGET_COL].copy()

    X_test_raw = df_test[feature_cols].copy()
    y_test_raw = df_test[TARGET_COL].copy()

    if df_val is not None:
        X_val_raw = df_val[feature_cols].copy()
        y_val_raw = df_val[TARGET_COL].copy()
    else:
        X_val_raw, y_val_raw = None, None

    # Impute missing values (median for all numeric features)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    X_val = imputer.transform(X_val_raw) if X_val_raw is not None else None

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_val is not None:
        X_val = scaler.transform(X_val)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    y_val = label_encoder.transform(y_val_raw) if y_val_raw is not None else None

    # Save transformers
    dump(scaler, "models/scaler.joblib")
    dump(imputer, "models/imputer.joblib")
    dump(label_encoder, "models/label_encoder.joblib")

    return (X_train, X_test, X_val,
            y_train, y_test, y_val,
            scaler, label_encoder,
            feature_cols)
