from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/lending_data.csv")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_PATH = Path("artifacts/scaler.pkl")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Args:
        path (Path): Path to raw dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features and remove multicollinearity.

    Keeps only one variable among highly correlated ones.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    cols_to_keep = [
        "borrower_income",
        "debt_to_income",
        "num_of_accounts",
        "derogatory_marks",
        "loan_status",
    ]
    return df[cols_to_keep]


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.
    """
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets using stratification.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float, optional): Proportion of test set. Defaults to 0.2.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using StandardScaler.

    Fits scaler on training data and applies to both train and test.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        scaler_path (Path): Path to save scaler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled train and test features.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return (
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
    )


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """
    Save processed train and test datasets to disk.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        X_test (pd.DataFrame): Scaled test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        output_dir (Path): Directory to save processed data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["loan_status"] = y_train.values

    test_df = X_test.copy()
    test_df["loan_status"] = y_test.values

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def run_pipeline() -> None:
    """
    Execute full feature engineering pipeline:
    - Load data
    - Select features
    - Split train/test
    - Scale features
    - Save processed datasets
    """
    df = load_data(RAW_PATH)
    df = select_features(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, ARTIFACTS_PATH)

    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, PROCESSED_DIR)


if __name__ == "__main__":
    run_pipeline()
