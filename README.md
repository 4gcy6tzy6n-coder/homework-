# Dry Bean Classification — Machine Learning Project

Multi-class classification of dry bean varieties using 5 machine learning algorithms.
Course: 机器学习与项目实践 (Machine Learning & Project Practice), 2026.

## Dataset

**Dry Bean Dataset** — 13,611 samples, 16 morphological features, 7 bean classes.

| Class | Train Samples | Description |
|-------|--------------|-------------|
| DERMASON | 2,503 | Dermason bean |
| SIRA | 1,837 | Sira bean |
| SEKER | 1,408 | Seker bean |
| HOROZ | 1,340 | Horoz bean |
| CALI | 1,151 | Cali bean |
| BARBUNYA | 927 | Barbunya bean |
| BOMBAY | 361 | Bombay bean |

**16 Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness,
ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

## Data Cleaning

The original dataset contains deliberate "dirty" data:

| Issue | Column | Treatment |
|-------|--------|------------|
| "?" placeholders | Solidity | → NaN → median imputation |
| " cm" suffix | Compactness | Stripped, converted to float |
| 5% missing | Perimeter | Median imputation |
| Typos / OCR errors | Class | D3RMAS0N→DERMASON, etc. normalized to 7 classes |
| Different scales | All features | StandardScaler normalization |

## Algorithms (5 models)

| # | Algorithm | Type | Classroom | Hyperparameters |
|---|-----------|------|-----------|-----------------|
| 1 | Random Forest | Ensemble (bagging) | Covered | 200 trees, max_depth=12 |
| 2 | K-Nearest Neighbors | Instance-based | Covered | k=7, distance-weighted |
| 3 | SVM (RBF Kernel) | Maximum margin | Covered | C=1.0, gamma=scale |
| 4 | **MLP Neural Network** | Deep learning | **NOT covered** | 2 hidden layers (128,64) |
| 5 | **Gradient Boosting** | Ensemble (boosting) | **NOT covered** | 150 estimators, lr=0.1 |

## Results Summary

| Model | Test Acc | Train Time | Infer (ms) | Gap | Macro F1 |
|-------|----------|------------|------------|-----|----------|
| **SVC** | **92.95%** | 0.79s | 157.4ms | 0.06% | 0.9376 |
| MLP | 92.80% | 0.38s | **0.78ms** | **0.25%** | 0.9368 |
| GradientBoosting | 92.80% | 33.19s | 30.9ms | 7.19% | 0.9399 |
| KNN | 92.33% | 0.001s | 15.9ms | 7.67% | 0.9321 |
| Random Forest | 92.18% | 0.40s | 22.4ms | 4.53% | 0.9317 |

**Key findings:**
- **SVC** achieves the highest accuracy (92.95%) with essentially zero overfitting
- **MLP** has the best generalization (gap=0.25%) and fastest inference (0.78ms)
- **GradientBoosting** has the best Macro F1 score (0.9399) but overfits significantly
- **KNN** has ~0 training time but suffers from memorization (theoretical gap)
- All models degrade gracefully under Gaussian noise; SVC is the most robust

## Analysis Plots Generated

The `results/` directory contains 20+ analysis figures:

| Category | Files |
|----------|-------|
| **EDA** | `class_distribution.png`, `missing_values.png`, `feature_boxplots.png` |
| **Correlation** | `correlation_heatmap.png` |
| **Dimensionality** | `pca_projection.png` |
| **Loss Curves** | `loss_curve_MLP.png` |
| **Confusion Matrices** | `confusion_matrices.png` |
| **Per-Class Metrics** | `f1_precision_recall.png` |
| **Feature Importance** | `feature_importance_RandomForest.png`, `feature_importance_GradientBoosting.png` |
| **Overfitting** | `overfitting.png` |
| **Inference Speed** | `inference_speed.png` |
| **Robustness** | `robustness.png` |
| **Learning Curves** | `learning_curve_*.png` (4 models) |

## Project Structure

```
DryBean_Project/
├── data/                          # Dirty CSV files (train/test/val)
├── src/
│   ├── config.py                  # Paths, seed, constants
│   ├── data_loader.py             # Load, clean, normalize, preprocess
│   ├── train.py                   # 5 classifiers
│   ├── evaluate.py                # 10 evaluation dimensions
│   └── utils.py                   # Noise injection, EDA plots
├── models/                        # 5 .joblib + scaler/imputer/encoder
├── results/                       # 20+ PNG + CSV outputs
├── main.py                        # CLI: --mode {eda,train,test,full}
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl

# One-shot: train all 5 models + full evaluation
python main.py --mode full

# Exploratory data analysis only
python main.py --mode eda

# Train only (saves models to models/)
python main.py --mode train

# Evaluate only (loads saved models)
python main.py --mode test
```

## Dependencies

- Python ≥ 3.9
- pandas, numpy, matplotlib, seaborn
- scikit-learn ≥ 1.0
- joblib
