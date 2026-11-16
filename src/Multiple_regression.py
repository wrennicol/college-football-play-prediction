
"""
Multiple Linear Regression: Predicting College Football PPA

Target:
    ppa

Data:
    College football play-by-play dataset with existing train/test splits.

Method:
    Ordinary Least Squares (OLS), using statsmodels.

Evaluation:
    Training R-squared and adjusted R-squared
    Test R-squared, RMSE, and MAE
    Coefficients, p-values, and confidence intervals
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

TARGET = "ppa"
SPLIT_COLUMN = "split"

# Predictors from the regression configuration.
NUMERIC_FEATURES = [
    "distance",
    "yardsToGoal",
    "sec_left_half",
    "sec_left_game",
    "score_diff",
    "drive_play_num",
    "prev_yards",
    "pass_rate_last5",
    "pass_rate_game_so_far",
    "team_prior_pass_rate",
    "team_prior_pass_rate_down",
]

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

CATEGORICAL_FEATURES = [
    "down",
    "period",
]

# Optional: include team identities as additional predictors.
# Set to True to include offense and defense.
INCLUDE_TEAM_FEATURES = False

OPTIONAL_TEAM_FEATURES = [
    "offense",
    "defense",
]


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

def load_data(file_path):
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required_columns = (
        [TARGET, SPLIT_COLUMN]
        + NUMERIC_FEATURES
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )

    if INCLUDE_TEAM_FEATURES:
        required_columns += OPTIONAL_TEAM_FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# ---------------------------------------------------------
# 3. PREPARE TRAINING AND TEST DATA
# ---------------------------------------------------------

def prepare_data(df):

    features = (
        NUMERIC_FEATURES
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )

    if INCLUDE_TEAM_FEATURES:
        features += OPTIONAL_TEAM_FEATURES

    # Remove rows where the target is missing.
    df = df.dropna(subset=[TARGET]).copy()

    # Separate using the existing game-level split.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Could not find both train and test rows. "
            "Check the values in the split column."
        )

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    y_train = pd.to_numeric(
        train_df[TARGET], errors="coerce"
    )

    y_test = pd.to_numeric(
        test_df[TARGET], errors="coerce"
    )

    # Remove any rows with invalid target values.
    valid_train = y_train.notna()
    valid_test = y_test.notna()

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    X_test = X_test.loc[valid_test]
    y_test = y_test.loc[valid_test]

    numeric_features = NUMERIC_FEATURES + BINARY_FEATURES
    categorical_features = CATEGORICAL_FEATURES.copy()

    if INCLUDE_TEAM_FEATURES:
        categorical_features += OPTIONAL_TEAM_FEATURES

    print("\nData split:")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations:  {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_features,
        categorical_features,
    )


# ---------------------------------------------------------
# 4. PREPROCESS FEATURES
# ---------------------------------------------------------

def preprocess_features(
    X_train,
    X_test,
    numeric_features,
    categorical_features,
):

    # Numeric missing values are replaced with training medians.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical missing values are replaced with the
    # most frequent training category.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Combine numeric and categorical preprocessing.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # Fit preprocessing ONLY on training data.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same transformations to test data.
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    # Ensure all values are numeric and finite.
    X_train_processed = X_train_processed.astype(float)
    X_test_processed = X_test_processed.astype(float)

    if not np.isfinite(X_train_processed.to_numpy()).all():
        raise ValueError("Non-finite values found in training predictors.")

    if not np.isfinite(X_test_processed.to_numpy()).all():
        raise ValueError("Non-finite values found in test predictors.")

    return X_train_processed, X_test_processed


# ---------------------------------------------------------
# 5. FIT MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------

def fit_regression(X_train, y_train):

    # Add intercept.
    X_train_constant = sm.add_constant(
        X_train,
        has_constant="add",
    )

    # Fit Ordinary Least Squares regression.
    model = sm.OLS(
        y_train.astype(float),
        X_train_constant,
    )

    # HC3 robust standard errors help account for
    # heteroskedasticity in play-level observations.
    results = model.fit(cov_type="HC3")

    return results


# ---------------------------------------------------------
# 6. EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(results, X_train, X_test, y_train, y_test):

    # Training predictions.
    X_train_constant = sm.add_constant(
        X_train,
        has_constant="add",
    )

    train_predictions = results.predict(X_train_constant)

    # Test predictions.
    X_test_constant = sm.add_constant(
        X_test,
        has_constant="add",
    )

    test_predictions = results.predict(X_test_constant)

    # Training metrics.
    train_r2 = r2_score(y_train, train_predictions)

    n = len(y_train)
    p = len(X_train.columns)

    if n > p + 1:
        adjusted_r2 = (
            1 - (1 - train_r2) * (n - 1) / (n - p - 1)
        )
    else:
        adjusted_r2 = np.nan

    # Test metrics.
    test_r2 = r2_score(y_test, test_predictions)

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print("\nTraining Performance:")
    print(f"R-squared:           {train_r2:.4f}")
    print(f"Adjusted R-squared:  {adjusted_r2:.4f}")

    print("\nTest Performance:")
    print(f"R-squared:           {test_r2:.4f}")
    print(f"RMSE:                {test_rmse:.4f}")
    print(f"MAE:                 {test_mae:.4f}")

    return train_predictions, test_predictions


# ---------------------------------------------------------
# 7. DISPLAY REGRESSION RESULTS
# ---------------------------------------------------------

def display_results(results):

    print("\n" + "=" * 60)
    print("OLS REGRESSION SUMMARY")
    print("=" * 60)

    print(results.summary())

    # Create a coefficient table.
    coefficient_table = pd.DataFrame({
        "Coefficient": results.params,
        "Std Error (HC3)": results.bse,
        "P-value": results.pvalues,
        "CI Lower": results.conf_int()[0],
        "CI Upper": results.conf_int()[1],
    })

    coefficient_table["Significant (0.05)"] = (
        coefficient_table["P-value"] < 0.05
    )

    coefficient_table = coefficient_table.sort_values(
        by="P-value"
    )

    print("\n" + "=" * 60)
    print("COEFFICIENTS AND STATISTICAL SIGNIFICANCE")
    print("=" * 60)

    print(coefficient_table.to_string())

    return coefficient_table


# ---------------------------------------------------------
# 8. SAVE OUTPUTS
# ---------------------------------------------------------

def save_results(
    results,
    coefficient_table,
    y_test,
    test_predictions,
):

    coefficient_table.to_csv(
        "ppa_regression_coefficients.csv"
    )

    predictions = pd.DataFrame({
        "Actual_PPA": y_test,
        "Predicted_PPA": test_predictions,
        "Residual": y_test - test_predictions,
    })

    predictions.to_csv(
        "ppa_regression_predictions.csv",
        index=True,
    )

    with open("ppa_regression_summary.txt", "w") as file:
        file.write(results.summary().as_text())

    print("\nSaved files:")
    print("- ppa_regression_coefficients.csv")
    print("- ppa_regression_predictions.csv")
    print("- ppa_regression_summary.txt")


# ---------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Multiple linear regression predicting PPA."
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="college_football_plays.csv",
        help="Path to the play-by-play CSV file.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("COLLEGE FOOTBALL PPA MULTIPLE REGRESSION")
    print("=" * 60)

    # Load data.
    df = load_data(args.csv_file)

    # Prepare train/test data.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_features,
        categorical_features,
    ) = prepare_data(df)

    # Preprocess predictors.
    X_train_processed, X_test_processed = preprocess_features(
        X_train,
        X_test,
        numeric_features,
        categorical_features,
    )

    print("\nNumber of regression predictors after encoding:")
    print(len(X_train_processed.columns))

    # Fit OLS regression.
    results = fit_regression(
        X_train_processed,
        y_train,
    )

    # Display coefficients and p-values.
    coefficient_table = display_results(results)

    # Evaluate predictions.
    train_predictions, test_predictions = evaluate_model(
        results,
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
    )

    # Save outputs.
    save_results(
        results,
        coefficient_table,
        y_test,
        test_predictions,
    )

    print("\nRegression completed successfully.")


if __name__ == "__main__":
    main()
