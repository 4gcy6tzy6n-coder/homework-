"""
Train multiple multi-class classifiers and save models to disk.
5 algorithms: RandomForest, KNN, MLP, GradientBoosting, SVC.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from .config import SEED, MODELS_DIR

np.random.seed(SEED)


def train_all(X_train, y_train, X_val=None, y_val=None):
    """
    Train all models, record training time, and save to disk.

    Returns:
        models: dict of {name: trained_model}
        train_times: dict of {name: train_seconds}
        loss_curves: dict of {name: loss_curve or None}
    """
    n_classes = len(np.unique(y_train))

    models_cfg = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            verbose=False,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=SEED,
        ),
        "SVC": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
            ensemble=False,
        ),
    }

    models = {}
    train_times = {}
    loss_curves = {}

    for name, model in models_cfg.items():
        print(f"  Training {name}...")

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_times[name] = elapsed
        models[name] = model

        # Extract loss curve for MLP
        if name == "MLP":
            loss_curves[name] = {
                "train_loss": model.loss_curve_,
                "val_loss": model.validation_scores_,
            }
        else:
            loss_curves[name] = None

        dump(model, f"{MODELS_DIR}/{name}.joblib")
        print(f"    Done in {elapsed:.2f}s")

    return models, train_times, loss_curves
