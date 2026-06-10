"""
Unified CLI entry point for the Dry Bean Classification project.

Usage:
    python main.py --mode train     # Train all models and save to disk
    python main.py --mode test      # Load models and evaluate
    python main.py --mode full      # Train + Evaluate (one-shot)
    python main.py --mode eda       # Exploratory data analysis only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_PATH, TEST_PATH, VAL_PATH, RESULTS_DIR, SEED, TARGET_COL
)
from src.data_loader import load_and_clean, preprocess
from src.train import train_all
from src.evaluate import evaluate_all
from src.utils import (
    plot_class_distribution, plot_feature_boxplots,
    plot_missing_values,
)


def run_eda():
    """Exploratory Data Analysis — generate diagnostic plots."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df_train = load_and_clean(TRAIN_PATH)
    feature_cols = [c for c in df_train.columns if c != TARGET_COL]

    print(f"\n  Shape: {df_train.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes: {df_train[TARGET_COL].nunique()}")
    print(f"\n  Missing values:\n{df_train.isnull().sum()[df_train.isnull().sum() > 0]}")
    print(f"\n  Class distribution:\n{df_train[TARGET_COL].value_counts()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n  Generating plots...")
    plot_missing_values(df_train)
    plot_feature_boxplots(df_train, feature_cols, title_prefix="Before Cleaning — ")

    # Class distribution needs encoded labels for the plot function
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_train[TARGET_COL])
    plot_class_distribution(y_encoded, le)

    print(f"\n  EDA plots saved to {RESULTS_DIR}/")
    print("Done.")


def run_train():
    """Train all models."""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    print("\nLoading and cleaning data...")
    df_train = load_and_clean(TRAIN_PATH)
    df_test = load_and_clean(TEST_PATH)
    df_val = load_and_clean(VAL_PATH)

    print(f"  Train: {df_train.shape}, Test: {df_test.shape}, Val: {df_val.shape}")

    print("\nPreprocessing...")
    (X_train, X_test, X_val,
     y_train, y_test, y_val,
     scaler, label_encoder, feature_cols) = preprocess(df_train, df_test, df_val)

    print(f"\nTraining models (seed={SEED})...")
    models, train_times, loss_curves = train_all(X_train, y_train, X_val, y_val)

    print("\nAll models trained and saved.")
    return X_train, X_test, X_val, y_train, y_test, y_val, models, train_times, loss_curves, label_encoder, feature_cols


def run_test(X_train, X_test, y_train, y_test,
             models, train_times, loss_curves,
             label_encoder, feature_cols):
    """Evaluate saved/trained models."""
    print("=" * 60)
    print("EVALUATION MODE")
    print("=" * 60)

    evaluate_all(
        models, train_times, loss_curves,
        X_train, y_train, X_test, y_test,
        feature_cols, label_encoder,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dry Bean Classification — ML Project"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["eda", "train", "test", "full"],
        help="Operation mode: eda (analysis only), train, test, or full (train+test)"
    )
    args = parser.parse_args()

    if args.mode == "eda":
        run_eda()

    elif args.mode == "train":
        run_train()

    elif args.mode == "test":
        # Load data and saved models for evaluation
        print("Loading data for evaluation...")
        df_train = load_and_clean(TRAIN_PATH)
        df_test = load_and_clean(TEST_PATH)
        df_val = load_and_clean(VAL_PATH)
        (X_train, X_test, X_val,
         y_train, y_test, y_val,
         scaler, label_encoder, feature_cols) = preprocess(df_train, df_test, df_val)

        from joblib import load
        models = {}
        train_times = {}
        loss_curves = {}
        for name in ["RandomForest", "KNN", "MLP", "GradientBoosting", "SVC"]:
            try:
                models[name] = load(f"models/{name}.joblib")
                print(f"  Loaded {name}.joblib")
                train_times[name] = 0  # Not available from saved model
                loss_curves[name] = None
            except FileNotFoundError:
                print(f"  WARNING: models/{name}.joblib not found, skipping")

        if not models:
            print("No models found. Run 'train' mode first.")
            sys.exit(1)

        run_test(X_train, X_test, y_train, y_test,
                 models, train_times, loss_curves,
                 label_encoder, feature_cols)

    elif args.mode == "full":
        (X_train, X_test, X_val,
         y_train, y_test, y_val,
         models, train_times, loss_curves,
         label_encoder, feature_cols) = run_train()
        run_test(X_train, X_test, y_train, y_test,
                 models, train_times, loss_curves,
                 label_encoder, feature_cols)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
