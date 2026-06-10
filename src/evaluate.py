"""
Evaluation module: accuracy, loss curves, inference speed,
robustness to noise, overfitting analysis, confusion matrices,
feature importance, correlation, PCA, learning curves, F1 scores.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, log_loss,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve

from .config import RESULTS_DIR, NOISE_LEVELS, SEED
from .utils import inject_gaussian_noise

np.random.seed(SEED)


def evaluate_all(models, train_times, loss_curves,
                 X_train, y_train, X_test, y_test,
                 feature_cols, label_encoder):
    """
    Run full evaluation suite and save all figures/tables.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    class_names = label_encoder.classes_.tolist()

    # 1) Test accuracy
    print("\n[1/10] Computing test accuracy...")
    acc_results = _compute_accuracy(models, X_test, y_test, label_encoder)

    # 2) Loss curves
    print("[2/10] Generating loss curves...")
    _plot_loss_curves(loss_curves)

    # 3) Inference speed
    print("[3/10] Measuring inference speed...")
    speed_results = _measure_speed(models, X_test)

    # 4) Confusion matrices
    print("[4/10] Plotting confusion matrices...")
    _plot_confusion_matrices(models, X_test, y_test, class_names)

    # 5) Robustness to noise
    print("[5/10] Evaluating robustness...")
    robustness_results = _evaluate_robustness(models, X_test, y_test)

    # 6) Overfitting analysis
    print("[6/10] Analyzing overfitting...")
    overfit_results = _analyze_overfitting(models, X_train, y_train, X_test, y_test)

    # 7) Feature importance (RF + GBC)
    print("[7/10] Plotting feature importance...")
    _plot_feature_importance(models, feature_cols)

    # 8) Correlation heatmap
    print("[8/10] Plotting correlation heatmap...")
    _plot_correlation_heatmap(X_train, feature_cols)

    # 9) PCA 2D visualization
    print("[9/10] Plotting PCA projection...")
    _plot_pca(X_train, y_train, label_encoder)

    # 10) Learning curves
    print("[10/10] Plotting learning curves...")
    _plot_learning_curves(models, X_train, y_train)

    # ── Summary table ──
    summary = _build_summary_table(
        models, train_times, acc_results, overfit_results,
        speed_results, robustness_results, X_test, y_test, label_encoder,
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Sub-functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_accuracy(models, X_test, y_test, label_encoder):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  {name}: {acc:.4f}")

        cls_names = label_encoder.inverse_transform(range(len(set(y_test))))
        per_class = {}
        for i, cls in enumerate(cls_names):
            mask = y_test == i
            if mask.sum() > 0:
                per_class[cls] = accuracy_score(y_test[mask], y_pred[mask])
        pd.Series(per_class, name=name).to_csv(
            f"{RESULTS_DIR}/per_class_accuracy_{name}.csv"
        )

    return results


def _plot_loss_curves(loss_curves):
    """Plot training/validation loss for MLP."""
    for name, curves in loss_curves.items():
        if curves is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Loss subplot
        axes[0].plot(curves["train_loss"], color="#e74c3c", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{name} — Training Loss")
        axes[0].grid(True, alpha=0.3)
        # Validation score subplot
        if curves["val_loss"] is not None:
            axes[1].plot(curves["val_loss"], color="#2ecc71", linewidth=1.5)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation Score")
            axes[1].set_title(f"{name} — Validation Accuracy")
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/loss_curve_{name}.png", dpi=150)
        plt.close()
        print(f"  Saved loss_curve_{name}.png")


def _measure_speed(models, X_test, n_runs=20):
    """Average inference time over n_runs."""
    results = {}
    for name, model in models.items():
        _ = model.predict(X_test[:10])  # warm-up
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            _ = model.predict(X_test)
            times.append(time.time() - t0)
        avg = np.mean(times)
        results[name] = avg
        print(f"  {name}: {avg*1000:.2f} ms (avg over {n_runs} runs)")

    # Bar chart
    plt.figure(figsize=(10, 5))
    names = list(results.keys())
    vals = [results[n] * 1000 for n in names]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    bars = plt.bar(names, vals, color=colors[:len(names)], edgecolor="white")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Speed Comparison (lower = faster)")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/inference_speed.png", dpi=150)
    plt.close()
    return results


def _plot_confusion_matrices(models, X_test, y_test, class_names):
    """Individual confusion matrix for each model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_title(name, fontsize=10)
        # Annotate each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j] if cm[i, j] > 0 else "",
                        ha="center", va="center", fontsize=6)

    plt.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # Combined F1/Precision/Recall bar chart
    _plot_f1_comparison(models, X_test, y_test, class_names)
    print("  Saved confusion_matrices.png")


def _plot_f1_comparison(models, X_test, y_test, class_names):
    """Per-class F1, precision, recall comparison across models."""
    metrics_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average=None)
        prec = precision_score(y_test, y_pred, average=None)
        rec = recall_score(y_test, y_pred, average=None)
        for i, cls in enumerate(class_names):
            metrics_data.append({
                "Model": name, "Class": cls,
                "F1": f1[i], "Precision": prec[i], "Recall": rec[i],
            })

    df = pd.DataFrame(metrics_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ["F1", "Precision", "Recall"]):
        pivot = df.pivot(index="Class", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Per-Class Metrics Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/f1_precision_recall.png", dpi=150)
    plt.close()
    print("  Saved f1_precision_recall.png")


def _evaluate_robustness(models, X_test, y_test):
    """Test accuracy degradation under increasing Gaussian noise."""
    results = {name: {} for name in models}
    all_data = []
    for level in NOISE_LEVELS:
        X_noisy = inject_gaussian_noise(X_test, sigma=level)
        for name, model in models.items():
            y_pred = model.predict(X_noisy)
            acc = accuracy_score(y_test, y_pred)
            results[name][level] = acc
            all_data.append({"Model": name, "Noise σ": level, "Accuracy": acc})

    # Plot
    df = pd.DataFrame(all_data)
    plt.figure(figsize=(10, 5))
    markers = {"RandomForest": "o", "KNN": "s", "MLP": "^",
               "GradientBoosting": "D", "SVC": "v"}
    colors = {"RandomForest": "#2ecc71", "KNN": "#3498db", "MLP": "#e74c3c",
              "GradientBoosting": "#f39c12", "SVC": "#9b59b6"}
    for name in models:
        sub = df[df["Model"] == name]
        plt.plot(sub["Noise σ"], sub["Accuracy"],
                 marker=markers.get(name, "o"), color=colors.get(name),
                 label=name, linewidth=2, markersize=8)
    plt.xlabel("Gaussian Noise σ", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Robustness to Input Noise", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/robustness.png", dpi=150)
    plt.close()
    print("  Saved robustness.png")
    return results


def _analyze_overfitting(models, X_train, y_train, X_test, y_test):
    """Compare train vs test accuracy to measure overfitting."""
    results = {}
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        gap = train_acc - test_acc
        results[name] = {"train_acc": train_acc, "test_acc": test_acc, "gap": gap}
        flag = " !! OVERFITTING" if gap > 0.05 else ""
        print(f"  {name}: train={train_acc:.4f}  test={test_acc:.4f}  gap={gap:.4f}{flag}")

    # Plot
    plt.figure(figsize=(12, 5))
    names = list(models.keys())
    x = np.arange(len(names))
    width = 0.3
    train_vals = [results[n]["train_acc"] for n in names]
    test_vals = [results[n]["test_acc"] for n in names]
    colors_train = ["#3498db"] * len(names)
    colors_test = ["#e74c3c"] * len(names)
    plt.bar(x - width / 2, train_vals, width, label="Train Acc",
            color=colors_train, edgecolor="white")
    plt.bar(x + width / 2, test_vals, width, label="Test Acc",
            color=colors_test, edgecolor="white")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    # Add gap annotations
    for i, name in enumerate(names):
        gap = results[name]["gap"]
        color = "#e74c3c" if gap > 0.05 else "#27ae60"
        plt.annotate(f"gap={gap:.3f}", (i, max(train_vals[i], test_vals[i])),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color=color, fontweight="bold")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy (Overfitting Check)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/overfitting.png", dpi=150)
    plt.close()
    print("  Saved overfitting.png")
    return results


def _plot_feature_importance(models, feature_cols):
    """Feature importance from tree-based models (RF, GBC)."""
    for name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        n_top = min(16, len(feature_cols))

        plt.figure(figsize=(10, 6))
        plt.barh(range(n_top), importances[indices][:n_top][::-1],
                 color="#3498db", edgecolor="white")
        plt.yticks(range(n_top), [feature_cols[i] for i in indices[:n_top]][::-1])
        plt.xlabel("Importance")
        plt.title(f"Feature Importance — {name}")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/feature_importance_{name}.png", dpi=150)
        plt.close()
    print("  Saved feature importance plots")


def _plot_correlation_heatmap(X, feature_cols):
    """Pearson correlation heatmap of features."""
    df = pd.DataFrame(X, columns=feature_cols)
    corr = df.corr()

    plt.figure(figsize=(14, 11))
    im = plt.imshow(corr, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(feature_cols)), feature_cols, fontsize=8)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            if i != j:
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                         ha="center", va="center", fontsize=6,
                         color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    plt.colorbar(im, shrink=0.8)
    plt.title("Feature Correlation Matrix (Pearson)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  Saved correlation_heatmap.png")


def _plot_pca(X, y, label_encoder):
    """PCA 2D projection of the training data."""
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    class_names = label_encoder.classes_
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for i, name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[i], label=name, alpha=0.5, s=10, edgecolors="none")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA 2D Projection of Training Data")
    plt.legend(markerscale=3, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pca_projection.png", dpi=150)
    plt.close()
    print("  Saved pca_projection.png")


def _plot_learning_curves(models, X_train, y_train):
    """Learning curves for selected models (subset to avoid slow SVC)."""
    fast_models = {k: v for k, v in models.items() if k != "SVC"}
    train_sizes = np.linspace(0.2, 1.0, 6)

    for name, model in fast_models.items():
        print(f"    Computing learning curve for {name}...")
        train_sz, train_sc, test_sc = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes, cv=3,
            scoring="accuracy", n_jobs=-1, random_state=SEED,
        )
        train_mean = train_sc.mean(axis=1)
        test_mean = test_sc.mean(axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_sz, train_mean, "o-", color="#3498db",
                 label="Training Score", linewidth=2)
        plt.plot(train_sz, test_mean, "o-", color="#e74c3c",
                 label="Cross-Validation Score", linewidth=2)
        plt.fill_between(train_sz,
                         train_sc.mean(axis=1) - train_sc.std(axis=1),
                         train_sc.mean(axis=1) + train_sc.std(axis=1),
                         alpha=0.15, color="#3498db")
        plt.fill_between(train_sz,
                         test_sc.mean(axis=1) - test_sc.std(axis=1),
                         test_sc.mean(axis=1) + test_sc.std(axis=1),
                         alpha=0.15, color="#e74c3c")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curve — {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/learning_curve_{name}.png", dpi=150)
        plt.close()
    print("  Saved learning curve plots")


def _build_summary_table(models, train_times, acc_results, overfit_results,
                         speed_results, robustness_results,
                         X_test, y_test, label_encoder):
    """Construct and save the summary CSV + formatted text."""
    rows = []
    for name in models:
        y_pred = models[name].predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        row = {
            "Model": name,
            "Train Time (s)": f"{train_times.get(name, 0):.3f}",
            "Test Accuracy": f"{acc_results[name]:.4f}",
            "Train Accuracy": f"{overfit_results[name]['train_acc']:.4f}",
            "Accuracy Gap": f"{overfit_results[name]['gap']:.4f}",
            "Macro F1": f"{f1_macro:.4f}",
            "Inference (ms)": f"{speed_results[name] * 1000:.2f}",
        }
        for level in NOISE_LEVELS:
            row[f"Acc σ={level}"] = f"{robustness_results[name][level]:.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{RESULTS_DIR}/summary_table.csv", index=False)
    print(f"\n  Summary table saved to {RESULTS_DIR}/summary_table.csv")
    print(summary.to_string(index=False))
    return summary
