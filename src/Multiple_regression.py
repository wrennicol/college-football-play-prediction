import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_df.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multiple_regression"


TARGET = "ppa"
SPLIT_COLUMN = "split"

# Numeric predictors
NUMERIC_FEATURES = [
    "distance",
    "yardsToGoal",
    "sec_left_game",
    "score_diff",
    "drive_play_num",
    "prev_yards",
    "pass_rate_last5",
    "pass_rate_game_so_far",
    "team_prior_pass_rate",
    "team_prior_pass_rate_down",
]

# Binary predictors
BINARY_FEATURES = [
    "is_home",
    "red_zone",
    "goal_to_go",
    "short_yardage",
    "two_minute",
    "trailing_late",
    "first_play_of_game",
    "prev_same_drive",
    "prev_call_1",
    "prev_call_2",
    "prev_call_3",
]

# Categorical predictors
CATEGORICAL_FEATURES = [
    "down",
    "period",
]

FEATURES = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run multiple linear regression to predict PPA."
    )

    parser.add_argument(
        "data",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Path to model_df.csv (defaults to data/processed/model_df.csv).",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where regression results will be saved.",
    )

    return parser.parse_args()


def resolve_path(path):
    """Resolve relative paths from the project root."""
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{data_path}\n\n"
            "Make sure model_df.csv exists in data/processed/."
        )

    df = pd.read_csv(data_path)

    print("\n" + "=" * 65)
    print("COLLEGE FOOTBALL PPA REGRESSION")
    print("=" * 65)

    print(f"\nDataset: {data_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required_columns = FEATURES + [TARGET, SPLIT_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nThe dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    # Ensure the target is numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Remove observations without a target.
    df = df.dropna(subset=[TARGET]).copy()

    # Normalize split labels.
    df[SPLIT_COLUMN] = (
        df[SPLIT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Could not find both 'train' and 'test' observations "
            "in the split column."
        )

    print(f"\nTraining observations: {len(train_df):,}")
    print(f"Testing observations:  {len(test_df):,}")
    print(f"Target: {TARGET}")

    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Impute missing values using training data only.
    One-hot encode categorical predictors.
    Align the test matrix with the training matrix.
    """

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    # Convert numeric and binary features to numeric values.
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        X_train[column] = pd.to_numeric(
            X_train[column], errors="coerce"
        )
        X_test[column] = pd.to_numeric(
            X_test[column], errors="coerce"
        )

        # Use training median for missing values.
        median_value = X_train[column].median()

        if pd.isna(median_value):
            median_value = 0

        X_train[column] = X_train[column].fillna(median_value)
        X_test[column] = X_test[column].fillna(median_value)

    # Convert categorical features to strings.
    for column in CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].astype("string")
        X_test[column] = X_test[column].astype("string")

        # Use the training mode for missing values.
        modes = X_train[column].mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else "Unknown"

        X_train[column] = X_train[column].fillna(fill_value).astype(str)
        X_test[column] = X_test[column].fillna(fill_value).astype(str)

    # One-hot encode categorical features.
    X_train = pd.get_dummies(
        X_train,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )

    X_test = pd.get_dummies(
        X_test,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )

    # Make sure test columns match training columns.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # Statsmodels requires numeric values.
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Remove predictors that have no variation in training data.
    constant_columns = [
        column for column in X_train.columns
        if X_train[column].nunique() <= 1
    ]

    if constant_columns:
        print("\nRemoving constant predictors:")
        for column in constant_columns:
            print(f"  - {column}")

        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)

    print(f"\nRegression predictors: {len(X_train.columns)}")

    return X_train, X_test


def train_model(X_train, y_train):
    # Add the intercept.
    X_train = sm.add_constant(X_train, has_constant="add")

    # Check for exact linear dependencies.
    matrix_rank = np.linalg.matrix_rank(X_train.to_numpy())

    if matrix_rank < X_train.shape[1]:
        warnings.warn(
            "The regression design matrix is rank-deficient. "
            "Some predictors are linearly dependent.",
            RuntimeWarning,
        )
        print(
            f"\nWARNING: Matrix rank is {matrix_rank}, "
            f"but there are {X_train.shape[1]} columns."
        )

    # Fit OLS with heteroskedasticity-robust standard errors.
    model = sm.OLS(y_train, X_train).fit(cov_type="HC3")

    return model, X_train


def evaluate_model(model, X_train, X_test, y_train, y_test):
    X_test = sm.add_constant(X_test, has_constant="add")

    # Match the exact training column order.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE")
    print("=" * 65)

    print("\nTraining Set")
    print(f"R-squared:          {train_r2:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"RMSE:               {train_rmse:.4f}")
    print(f"MAE:                {train_mae:.4f}")

    print("\nTesting Set")
    print(f"R-squared:          {test_r2:.4f}")
    print(f"RMSE:               {test_rmse:.4f}")
    print(f"MAE:                {test_mae:.4f}")

    print("\n" + "=" * 65)

    return test_predictions

def save_results(
    model,
    test_df,
    test_predictions,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save coefficients, p-values, and confidence intervals.
    confidence_intervals = model.conf_int()

    coefficients = pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "standard_error": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_lower": confidence_intervals.iloc[:, 0].values,
        "ci_upper": confidence_intervals.iloc[:, 1].values,
    })

    coefficients["significant_at_0.05"] = (
        coefficients["p_value"] < 0.05
    )

    coefficients = coefficients.sort_values(
        "p_value",
        ascending=True,
    )

    # Save test predictions.
    prediction_columns = [
        column for column in ["id", "gameId", "offense", "defense"]
        if column in test_df.columns
    ]

    predictions = test_df[prediction_columns].copy()
    predictions["actual_ppa"] = test_df[TARGET].values
    predictions["predicted_ppa"] = test_predictions
    predictions["residual"] = (
        predictions["actual_ppa"] - predictions["predicted_ppa"]
    )

    # Save the complete statistical summary.
    with open(
        output_dir / "ppa_regression_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(model.summary().as_text())

    print("\nRESULTS SAVED")
    print(f"Output directory: {output_dir}")
    print("  - ppa_regression_summary.txt")


def main():
    args = parse_arguments()

    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)

    # Load data.
    train_df, test_df = load_data(data_path)

    # Separate target.
    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    # Prepare predictors.
    X_train, X_test = prepare_features(train_df, test_df)

    # Train model.
    model, X_train = train_model(X_train, y_train)

    # Print statistical summary.
    print("\n" + "=" * 65)
    print("OLS REGRESSION SUMMARY")
    print("=" * 65)
    print(model.summary())

    # Evaluate model.
    test_predictions = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Save outputs.
    save_results(
        model,
        test_df,
        test_predictions,
        output_dir,
    )

    print("\nRegression completed successfully.")


if __name__ == "__main__":
    main()
