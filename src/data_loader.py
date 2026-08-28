"""
Data loading module for the Breast Cancer Diagnostic Classification project.

This module loads the Breast Cancer Wisconsin (Diagnostic) dataset
provided by scikit-learn.
"""

from sklearn.datasets import load_breast_cancer


def load_dataset():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns
    -------
    X : pandas.DataFrame
        30 predictive numerical features.
    y : pandas.Series
        Binary target labels.
    """

    data = load_breast_cancer(as_frame=True)

    X = data.data
    y = data.target

    return X, y


if __name__ == "__main__":
    X, y = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print("\nTarget distribution:")
    print(y.value_counts())