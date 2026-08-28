"""
Model optimization module for the Breast Cancer Diagnostic Classification project.

This module performs:
1. Grid Search for Logistic Regression.
2. Randomized Search for Random Forest.

ROC-AUC is used as the primary optimization metric.

The optimization results are automatically saved into:
results/optimization_results.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset
from preprocessing import split_data


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results"
)


# ---------------------------------------------------------
# Logistic Regression Optimization
# ---------------------------------------------------------

def optimize_logistic_regression(X_train, y_train):
    """
    Optimize Logistic Regression using GridSearchCV.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ])

    parameter_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
        "model__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION OPTIMIZATION")
    print("=" * 60)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest Cross-Validation ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    return search


# ---------------------------------------------------------
# Random Forest Optimization
# ---------------------------------------------------------

def optimize_random_forest(X_train, y_train):
    """
    Optimize Random Forest using RandomizedSearchCV.
    """

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    parameter_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("RANDOM FOREST OPTIMIZATION")
    print("=" * 60)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest Cross-Validation ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    return search


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load dataset
    X, y = load_dataset()

    # Train/Test Split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # -----------------------------------------------------
    # Optimize Logistic Regression
    # -----------------------------------------------------

    logistic_search = optimize_logistic_regression(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Optimize Random Forest
    # -----------------------------------------------------

    random_forest_search = optimize_random_forest(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Optimization completed
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION COMPLETED")
    print("=" * 60)

    print(
        "\nOptimized models are ready for final test evaluation."
    )

    # -----------------------------------------------------
    # Create results directory
    # -----------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Prepare optimization results
    # -----------------------------------------------------

    optimization_results = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "optimization_method": "Grid Search",
            "best_cv_roc_auc": logistic_search.best_score_,
            "best_parameters": str(
                logistic_search.best_params_
            )
        },
        {
            "model": "Random Forest",
            "optimization_method": "Randomized Search",
            "best_cv_roc_auc": random_forest_search.best_score_,
            "best_parameters": str(
                random_forest_search.best_params_
            )
        }
    ])

    # -----------------------------------------------------
    # Save optimization results
    # -----------------------------------------------------

    optimization_file = os.path.join(
        RESULTS_DIR,
        "optimization_results.csv"
    )

    optimization_results.to_csv(
        optimization_file,
        index=False
    )

    # -----------------------------------------------------
    # Display saved results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS SAVED")
    print("=" * 60)

    print("\nResults saved to:")
    print(optimization_file)

    print("\nOptimization Results:")
    print(
        optimization_results.to_string(
            index=False
        )
    )