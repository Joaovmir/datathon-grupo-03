from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from features import feature_engineering as fe
from features.feature_engineering import (
    load_data,
    save_processed_data,
    scale_features,
    select_features,
    split_features_target,
    split_train_test,
    transform_features,
)


@pytest.fixture
def temp_scaler_path(tmp_path: Path) -> Path:
    """
    Provide a temporary file path for saving the scaler artifact.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        Path: Path to scaler file.
    """
    return tmp_path / "scaler.pkl"


# ------------------------
# Unit Tests
# ------------------------


def test_load_data(tmp_path: Path):
    """
    Test if CSV data is correctly loaded into a DataFrame.
    """
    file_path = tmp_path / "sample.csv"

    df_expected = pd.DataFrame(
        {
            "borrower_income": [1000, 2000],
            "debt_to_income": [0.3, 0.4],
            "num_of_accounts": [2, 3],
            "derogatory_marks": [0, 1],
            "loan_status": [0, 1],
        }
    )

    df_expected.to_csv(file_path, index=False)

    df_loaded = load_data(file_path)

    assert not df_loaded.empty
    assert df_loaded.shape == df_expected.shape
    assert list(df_loaded.columns) == list(df_expected.columns)


def test_select_features(sample_data: pd.DataFrame):
    """
    Ensure only the expected feature columns are selected.
    """
    df = select_features(sample_data)

    expected_columns = {
        "borrower_income",
        "debt_to_income",
        "num_of_accounts",
        "derogatory_marks",
        "loan_status",
    }

    assert set(df.columns) == expected_columns


def test_split_features_target(sample_data: pd.DataFrame):
    """
    Validate separation between features (X) and target (y).
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    assert "loan_status" not in X.columns
    assert len(X) == len(y)


def test_split_train_test(sample_data: pd.DataFrame):
    """
    Test train/test split and approximate stratification preservation.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) + len(X_test) == len(X)

    original_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(original_ratio - train_ratio) < 0.05
    assert abs(original_ratio - test_ratio) < 0.05


def test_save_processed_data(tmp_path: Path):
    """
    Verify that processed datasets are saved correctly and contain target column.
    """
    X_train = pd.DataFrame({"f1": [0.1, 0.2]})
    X_test = pd.DataFrame({"f1": [0.3]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1])

    save_processed_data(X_train, X_test, y_train, y_test, tmp_path)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert "loan_status" in train_df.columns
    assert "loan_status" in test_df.columns
    assert len(train_df) == len(X_train)
    assert len(test_df) == len(X_test)


def test_scale_features(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Ensure scaling preserves shape and creates scaler artifact.
    Only X_train is passed — X_test is not involved in fitting or transforming here.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, _, _, _ = split_train_test(X, y)

    X_train_scaled = scale_features(X_train, temp_scaler_path)

    assert X_train.shape == X_train_scaled.shape
    assert temp_scaler_path.exists()


def test_run_pipeline(sample_data: pd.DataFrame, tmp_path: Path, monkeypatch):
    """
    Test full pipeline execution including file outputs.
    Verifies that train.csv is scaled and test.csv is saved raw.
    """
    raw_file = tmp_path / "raw.csv"
    sample_data.to_csv(raw_file, index=False)

    monkeypatch.setattr(fe, "RAW_PATH", raw_file)
    monkeypatch.setattr(fe, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(fe, "ARTIFACTS_PATH", tmp_path / "scaler.pkl")

    fe.run_pipeline()

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "scaler.pkl").exists()

    train_df = pd.read_csv(tmp_path / "train.csv")
    test_df = pd.read_csv(tmp_path / "test.csv")

    assert not train_df.empty
    assert not test_df.empty
    assert "loan_status" in train_df.columns
    assert "loan_status" in test_df.columns

    # test.csv must be unscaled: values should match original feature ranges
    feature_cols = [
        "borrower_income",
        "debt_to_income",
        "num_of_accounts",
        "derogatory_marks",
    ]
    assert (test_df[feature_cols].abs() > 1).any().any()


# ------------------------
# Feature Engineering Tests
# ------------------------


def test_no_nulls_after_processing(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Ensure no null values exist in scaled training data.
    Test set is transformed separately via transform_features().
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, _, _ = split_train_test(X, y)
    X_train_scaled = scale_features(X_train, temp_scaler_path)
    X_test_scaled = transform_features(X_test, temp_scaler_path)

    assert not X_train_scaled.isnull().any().any()
    assert not X_test_scaled.isnull().any().any()


def test_train_scaled_mean_std(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Validate that TRAINING scaled features have mean ~0 and std ~1.

    This guarantee applies only to the training set, since the scaler is
    fitted exclusively on it. The test set is transformed using training
    statistics and will generally not have mean=0 or std=1.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, _, _, _ = split_train_test(X, y)
    X_train_scaled = scale_features(X_train, temp_scaler_path)

    assert np.allclose(X_train_scaled.mean(), 0, atol=1e-1)
    assert np.allclose(X_train_scaled.std(), 1, atol=1e-1)


def test_test_set_not_normalized(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Confirm that the test set does NOT necessarily have mean=0 or std=1.

    The scaler is fitted only on training data. Applying it to the test set
    uses training statistics, so test features will generally deviate from
    standard normal — which is the correct and expected behavior.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, _, _ = split_train_test(X, y)
    scale_features(X_train, temp_scaler_path)
    X_test_scaled = transform_features(X_test, temp_scaler_path)

    # At least one feature should deviate meaningfully from mean=0
    assert not np.allclose(X_test_scaled.mean(), 0, atol=1e-10)


def test_scaler_fitted_only_on_train(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Verify the persisted scaler statistics match the training set, not the test set.

    The scaler's mean_ and scale_ must reflect training data exclusively.
    Using test statistics here would indicate data leakage.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, _, _ = split_train_test(X, y)
    scale_features(X_train, temp_scaler_path)

    scaler = joblib.load(temp_scaler_path)

    # Scaler parameters must match training set statistics
    assert np.allclose(scaler.mean_, X_train.mean().values, atol=1e-6)
    assert np.allclose(scaler.scale_, X_train.std(ddof=0).values, atol=1e-6)

    # Scaler parameters must NOT match test set statistics (different distribution)
    assert not np.allclose(scaler.mean_, X_test.mean().values, atol=1e-1)


def test_transform_features_uses_train_statistics(
    sample_data: pd.DataFrame, temp_scaler_path: Path
):
    """
    Ensure transform_features applies the pre-fitted scaler correctly.

    Scales X_test via transform_features() and verifies the result matches
    what a manual scaler.transform() call would produce — confirming the
    saved artifact is used without re-fitting.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, _, _ = split_train_test(X, y)
    scale_features(X_train, temp_scaler_path)

    # Simulate evaluation: load scaler and transform test set independently
    X_test_scaled = transform_features(X_test, temp_scaler_path)

    # Manually verify using the saved scaler directly
    scaler = joblib.load(temp_scaler_path)
    expected = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    pd.testing.assert_frame_equal(X_test_scaled, expected)


def test_feature_ranges(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Ensure scaled training values remain within reasonable bounds.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, _, _, _ = split_train_test(X, y)
    X_train_scaled = scale_features(X_train, temp_scaler_path)

    assert (X_train_scaled.abs() < 10).all().all()


def test_output_shapes(sample_data: pd.DataFrame, temp_scaler_path: Path):
    """
    Ensure consistency between features and target sizes after processing.
    """
    df = select_features(sample_data)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_scaled = scale_features(X_train, temp_scaler_path)
    X_test_scaled = transform_features(X_test, temp_scaler_path)

    assert X_train_scaled.shape[0] == y_train.shape[0]
    assert X_test_scaled.shape[0] == y_test.shape[0]


def test_select_features_missing_column():
    """
    Ensure function raises KeyError when required columns are missing.
    """
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})

    with pytest.raises(KeyError):
        select_features(df)


def test_empty_dataframe():
    """
    Validate behavior when input DataFrame is empty.
    """
    df = pd.DataFrame(
        columns=[
            "borrower_income",
            "debt_to_income",
            "num_of_accounts",
            "derogatory_marks",
            "loan_status",
        ]
    )

    X, y = split_features_target(df)

    assert X.empty
    assert y.empty


def test_scaler_constant_values(temp_scaler_path: Path):
    """
    Ensure scaler handles constant-value features without producing NaNs.
    """
    df = pd.DataFrame(
        {
            "borrower_income": [1000] * 10,
            "debt_to_income": [0.5] * 10,
            "num_of_accounts": [2] * 10,
            "derogatory_marks": [1] * 10,
            "loan_status": [0, 1] * 5,
        }
    )

    X, y = split_features_target(df)
    X_train, _, _, _ = split_train_test(X, y)

    X_train_scaled = scale_features(X_train, temp_scaler_path)

    assert not X_train_scaled.isnull().any().any()
