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
    scaler_path: Path,
) -> pd.DataFrame:
    """
    Fit a StandardScaler on training data and persist it to disk.

    Only the training set is used to fit the scaler. The test set must
    NOT be passed here — it should be transformed independently at
    evaluation time via transform_features(), using the saved artifact.

    Args:
        X_train (pd.DataFrame): Training features (used to fit the scaler).
        scaler_path (Path): Path to persist the fitted scaler artifact.

    Returns:
        pd.DataFrame: Scaled training features.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns)


def transform_features(
    X: pd.DataFrame,
    scaler_path: Path,
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to new data at evaluation or inference time.

    This function should be used instead of scale_features whenever the
    scaler has already been fitted (e.g. during model evaluation or serving).
    It guarantees the scaler is never re-fitted on unseen data.

    Args:
        X (pd.DataFrame): Features to transform.
        scaler_path (Path): Path to the persisted fitted scaler artifact.

    Returns:
        pd.DataFrame: Scaled features using training statistics.
    """
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """
    Save processed datasets to disk.

    The training set is saved already scaled (ready for model training).
    The test set is saved in its original, unscaled form so that it can
    simulate real incoming data at evaluation time — transformation is
    applied on demand via transform_features() using the saved scaler.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        X_test (pd.DataFrame): Raw (unscaled) test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        output_dir (Path): Directory to save processed data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["loan_status"] = y_train.values

    # test set is intentionally saved unscaled — apply transform_features()
    # at evaluation time using the persisted scaler artifact
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
    - Scale training features (fit + transform); persist scaler artifact
    - Save train (scaled) and test (raw) datasets to disk
    """
    df = load_data(RAW_PATH)
    df = select_features(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # scaler is fitted only on train; X_test is kept raw
    X_train_scaled = scale_features(X_train, ARTIFACTS_PATH)

    # test set saved unscaled — transform at evaluation time via transform_features()
    save_processed_data(X_train_scaled, X_test, y_train, y_test, PROCESSED_DIR)


if __name__ == "__main__":
    run_pipeline()
