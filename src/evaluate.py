"""
Model evaluation module for the Breast Cancer Diagnostic Classification project.

This module evaluates the baseline classification models using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- 5-fold Stratified Cross-Validation

The results are automatically saved into the results/ directory.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_loader import load_dataset
from preprocessing import split_data
from train import (
    create_logistic_regression_pipeline,
    create_decision_tree,
    create_random_forest
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results"
)


# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a single classification model.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negative": matrix[0][0],
        "false_positive": matrix[0][1],
        "false_negative": matrix[1][0],
        "true_positive": matrix[1][1]
    }


# ---------------------------------------------------------
# Cross Validation
# ---------------------------------------------------------

def cross_validate_model(name, model, X_train, y_train):
    """
    Perform 5-fold stratified cross-validation using ROC-AUC.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    print(f"\n{name} - 5-Fold Cross-Validation")
    print("-" * 40)
    print("ROC-AUC scores:", np.round(scores, 4))
    print(f"Mean ROC-AUC: {scores.mean():.4f}")
    print(f"Std ROC-AUC : {scores.std():.4f}")

    return {
        "model": name,
        "fold_1_roc_auc": scores[0],
        "fold_2_roc_auc": scores[1],
        "fold_3_roc_auc": scores[2],
        "fold_4_roc_auc": scores[3],
        "fold_5_roc_auc": scores[4],
        "mean_roc_auc": scores.mean(),
        "std_roc_auc": scores.std()
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    X, y = load_dataset()

    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # -----------------------------------------------------
    # Create baseline models
    # -----------------------------------------------------

    models = {
        "Logistic Regression": create_logistic_regression_pipeline(),
        "Decision Tree": create_decision_tree(),
        "Random Forest": create_random_forest()
    }

    baseline_results = []
    cross_validation_results = []

    # -----------------------------------------------------
    # Evaluate models
    # -----------------------------------------------------

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        baseline_results.append(result)

        cv_result = cross_validate_model(
            name,
            model,
            X_train,
            y_train
        )

        cross_validation_results.append(cv_result)

    # -----------------------------------------------------
    # Create results directory
    # -----------------------------------------------------

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Save baseline results
    # -----------------------------------------------------

    baseline_df = pd.DataFrame(baseline_results)

    baseline_file = os.path.join(
        RESULTS_DIR,
        "baseline_results.csv"
    )

    baseline_df.to_csv(
        baseline_file,
        index=False
    )

    # -----------------------------------------------------
    # Save cross-validation results
    # -----------------------------------------------------

    cv_df = pd.DataFrame(cross_validation_results)

    cv_file = os.path.join(
        RESULTS_DIR,
        "cross_validation_results.csv"
    )

    cv_df.to_csv(
        cv_file,
        index=False
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULT FILES CREATED")
    print("=" * 60)

    print("\nBaseline results:")
    print(baseline_file)

    print("\nCross-validation results:")
    print(cv_file)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)