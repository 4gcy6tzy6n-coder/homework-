"""
Utility functions: noise injection, class distribution plots, EDA helpers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RESULTS_DIR


def inject_gaussian_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise with std 'sigma' to the data."""
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise


def plot_class_distribution(y, label_encoder, title="Class Distribution"):
    """Bar plot of class frequencies."""
    counts = pd.Series(y).value_counts().sort_index()
    labels = label_encoder.inverse_transform(counts.index)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts.values, color="#3498db", edgecolor="white")
    plt.xlabel("Bean Class")
    plt.ylabel("Count")
    plt.title(title)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=150)
    plt.close()


def plot_feature_boxplots(df: pd.DataFrame, feature_cols, title_prefix=""):
    """Generate boxplots to visualize feature distributions and outliers."""
    n = len(feature_cols)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{title_prefix}Feature Distributions (Boxplot)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_boxplots.png", dpi=150)
    plt.close()


def plot_missing_values(df: pd.DataFrame):
    """Heatmap-style bar chart of missing value counts."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.bar(missing.index, missing.values, color="#e74c3c", edgecolor="white")
    plt.ylabel("Missing Count")
    plt.title("Missing Values per Column")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/missing_values.png", dpi=150)
    plt.close()
