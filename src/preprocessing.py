"""
Preprocessing module for the Breast Cancer Diagnostic Classification project.

This module:
1. Loads the dataset.
2. Checks data quality.
3. Splits the data into training and test sets.
4. Creates a preprocessing and Logistic Regression pipeline.
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_loader import load_dataset


def check_data_quality(X):
    """
    Check the dataset for missing values and duplicate rows.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    dict
        Data quality information.
    """

    quality_report = {
        "rows": X.shape[0],
        "features": X.shape[1],
        "missing_values": X.isna().sum().sum(),
        "duplicate_rows": X.duplicated().sum(),
    }

    return quality_report


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into stratified training and test sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    test_size : float
        Proportion of data used for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def create_logistic_regression_pipeline():
    """
    Create a preprocessing pipeline using StandardScaler
    followed by Logistic Regression.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Configured ML pipeline.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ])

    return pipeline


if __name__ == "__main__":
    X, y = load_dataset()

    quality = check_data_quality(X)

    print("Data Quality Report")
    print("-------------------")
    print(f"Rows: {quality['rows']}")
    print(f"Features: {quality['features']}")
    print(f"Missing values: {quality['missing_values']}")
    print(f"Duplicate rows: {quality['duplicate_rows']}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nData Split")
    print("----------")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    pipeline = create_logistic_regression_pipeline()

    pipeline.fit(X_train, y_train)

    print("\nPreprocessing and baseline pipeline created successfully.")