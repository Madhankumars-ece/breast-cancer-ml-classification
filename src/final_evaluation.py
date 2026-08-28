"""
Final evaluation module for the Breast Cancer Diagnostic Classification project.

This module:
1. Loads the Breast Cancer Wisconsin dataset.
2. Creates the same train/test split used throughout the project.
3. Optimizes Logistic Regression and Random Forest.
4. Evaluates the optimized models on the untouched test set.
5. Saves the final comparison results to CSV.
"""

import os

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
# Model Optimization
# ---------------------------------------------------------

def optimize_logistic_regression(X_train, y_train):
    """Optimize Logistic Regression using GridSearchCV."""

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
        pipeline,
        parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest using RandomizedSearchCV."""

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
        model,
        parameter_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate an already-trained model on the untouched test set."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    matrix = confusion_matrix(y_test, predictions)

    print(f"\n{model_name}")
    print("-" * len(model_name))

    for metric, value in results.items():
        if metric != "model":
            print(f"{metric.replace('_', ' ').title():10}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    # Load dataset
    X, y = load_dataset()

    # Create fixed train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Optimize models using training data only
    print("\nOptimizing Logistic Regression...")
    logistic_search = optimize_logistic_regression(
        X_train,
        y_train
    )

    print("Logistic Regression optimization completed.")

    print("\nOptimizing Random Forest...")
    random_forest_search = optimize_random_forest(
        X_train,
        y_train
    )

    print("Random Forest optimization completed.")

    # Print selected parameters
    print("\n" + "=" * 60)
    print("SELECTED HYPERPARAMETERS")
    print("=" * 60)

    print("\nLogistic Regression:")
    print(logistic_search.best_params_)

    print("\nRandom Forest:")
    print(random_forest_search.best_params_)

    # Final test evaluation
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    results = []

    logistic_result = evaluate_model(
        "Tuned Logistic Regression",
        logistic_search.best_estimator_,
        X_test,
        y_test
    )

    results.append(logistic_result)

    random_forest_result = evaluate_model(
        "Tuned Random Forest",
        random_forest_search.best_estimator_,
        X_test,
        y_test
    )

    results.append(random_forest_result)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = pd.DataFrame(results)

    output_file = os.path.join(
        RESULTS_DIR,
        "optimized_test_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 60)

    print(f"\nResults saved to:")
    print(output_file)

    print("\nResults:")
    print(results_df.to_string(index=False))