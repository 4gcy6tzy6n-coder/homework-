"""
Project configuration — paths, random seed, and constants.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data files
TRAIN_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_test.csv")
VAL_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_val.csv")

# Random seed for reproducibility
SEED = 42

# Noise levels for robustness evaluation
NOISE_LEVELS = [0.01, 0.05, 0.1]

# Train/val split ratio (from training set, for XGBoost early stopping)
VAL_RATIO = 0.15

# Target column
TARGET_COL = "Class"

# Expected bean classes (after cleaning)
EXPECTED_CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
