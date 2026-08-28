"""
Model training module for the Breast Cancer Diagnostic Classification project.

This module trains three baseline classification models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_dataset
from preprocessing import split_data, create_logistic_regression_pipeline


def create_decision_tree():
    """
    Create the baseline Decision Tree classifier.
    """
    return DecisionTreeClassifier(
        random_state=42
    )


def create_random_forest():
    """
    Create the baseline Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


def train_models():
    """
    Load the dataset, split the data and train all baseline models.

    Returns
    -------
    dict
        Dictionary containing trained models and test data.
    """

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Logistic Regression": create_logistic_regression_pipeline(),
        "Decision Tree": create_decision_tree(),
        "Random Forest": create_random_forest()
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained successfully.")

    return trained_models, X_test, y_test


if __name__ == "__main__":
    trained_models, X_test, y_test = train_models()

    print("\n" + "=" * 50)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 50)

    print(f"\nNumber of trained models: {len(trained_models)}")
    print(f"Test samples available: {len(X_test)}")

    print("\nModels:")
    for model_name in trained_models:
        print(f"- {model_name}")