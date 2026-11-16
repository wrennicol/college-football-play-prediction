
"""
Ridge Regression: College Football Play Prediction

Run from the project root:
    python src/Ridge_regression.py

Optional:
    python src/Ridge_regression.py data/processed/model_df.csv

Target:
    ppa

Uses:
    - Existing train/test split
    - StandardScaler
    - OneHotEncoder
    - Ridge regression
    - Cross-validation to select the best alpha
    - GroupKFold by gameId to prevent game-level leakage
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_df.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "ridge_regression"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

TARGET = "ppa"
SPLIT_COLUMN = "split"
GROUP_COLUMN = "gameId"

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

FEATURES = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)

# Values tested during cross-validation.
ALPHA_VALUES = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000,
]


# ---------------------------------------------------------
# ARGUMENTS AND PATHS
# ---------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Ridge regression to predict PPA."
    )

    parser.add_argument(
        "data",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Path to model_df.csv.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved regression results.",
    )

    return parser.parse_args()


def resolve_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(data_path):

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = (
        FEATURES
        + [TARGET, SPLIT_COLUMN, GROUP_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df.dropna(subset=[TARGET]).copy()

    df[SPLIT_COLUMN] = (
        df[SPLIT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_df = df[
        df[SPLIT_COLUMN] == "train"
    ].copy()

    test_df = df[
        df[SPLIT_COLUMN] == "test"
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Both train and test splits must exist."
        )

    print("\n" + "=" * 65)
    print("COLLEGE FOOTBALL RIDGE REGRESSION")
    print("=" * 65)

    print(f"\nDataset: {data_path}")
    print(f"Total observations: {len(df):,}")
    print(f"Training observations: {len(train_df):,}")
    print(f"Testing observations: {len(test_df):,}")
    print(f"Target: {TARGET}")

    return train_df, test_df


# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------

def build_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("ridge", Ridge()),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# CROSS-VALIDATION AND MODEL TRAINING
# ---------------------------------------------------------

def train_model(X_train, y_train, groups):

    unique_games = groups.nunique()

    if unique_games < 2:
        raise ValueError(
            "At least two unique games are required "
            "for group-based cross-validation."
        )

    n_splits = min(5, unique_games)

    cv = GroupKFold(n_splits=n_splits)

    pipeline = build_pipeline()

    parameter_grid = {
        "ridge__alpha": ALPHA_VALUES,
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    print("\n" + "=" * 65)
    print("RIDGE REGRESSION CROSS-VALIDATION")
    print("=" * 65)

    print(f"\nCross-validation folds: {n_splits}")
    print(f"Unique training games: {unique_games}")
    print(f"Alpha values: {ALPHA_VALUES}")

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    best_model = search.best_estimator_

    print("\nBest alpha:", search.best_params_["ridge__alpha"])

    print(
        "Best cross-validation RMSE:",
        f"{np.sqrt(-search.best_score_):.4f}",
    )

    return search, best_model


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "R_squared": r2_score(y_true, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "MAE": mean_absolute_error(
            y_true,
            predictions,
        ),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE")
    print("=" * 65)

    print("\nTraining Set")

    for metric, value in train_metrics.items():
        print(f"{metric:<15}: {value:.4f}")

    print("\nTesting Set")

    for metric, value in test_metrics.items():
        print(f"{metric:<15}: {value:.4f}")

    return (
        train_predictions,
        test_predictions,
        train_metrics,
        test_metrics,
    )


# ---------------------------------------------------------
# COEFFICIENT EXTRACTION
# ---------------------------------------------------------

def get_coefficients(model):

    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["ridge"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": ridge.coef_,
    })

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    return coefficients


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    model,
    search,
    test_df,
    test_predictions,
    train_metrics,
    test_metrics,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save coefficients.
    coefficients = get_coefficients(model)

    coefficients.to_csv(
        output_dir / "ridge_regression_coefficients.csv",
        index=False,
    )

    # Save test predictions.
    id_columns = [
        column
        for column in [
            "id",
            "gameId",
            "offense",
            "defense",
        ]
        if column in test_df.columns
    ]

    predictions = test_df[id_columns].copy()

    predictions["actual_ppa"] = test_df[TARGET].values
    predictions["predicted_ppa"] = test_predictions

    predictions["residual"] = (
        predictions["actual_ppa"]
        - predictions["predicted_ppa"]
    )

    predictions.to_csv(
        output_dir / "ridge_regression_predictions.csv",
        index=False,
    )

    # Save cross-validation results.
    cv_results = pd.DataFrame(search.cv_results_)

    cv_results.to_csv(
        output_dir / "ridge_cross_validation.csv",
        index=False,
    )

    # Save model summary.
    summary_path = (
        output_dir / "ridge_regression_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("COLLEGE FOOTBALL RIDGE REGRESSION\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Target: {TARGET}\n")
        file.write(
            f"Best alpha: {search.best_params_['ridge__alpha']}\n"
        )

        file.write(
            "Best CV RMSE: "
            f"{np.sqrt(-search.best_score_):.4f}\n\n"
        )

        file.write("TRAINING PERFORMANCE\n")
        file.write("-" * 60 + "\n")

        for metric, value in train_metrics.items():
            file.write(f"{metric}: {value:.4f}\n")

        file.write("\nTESTING PERFORMANCE\n")
        file.write("-" * 60 + "\n")

        for metric, value in test_metrics.items():
            file.write(f"{metric}: {value:.4f}\n")

        file.write("\nMODEL COEFFICIENTS\n")
        file.write("-" * 60 + "\n")

        file.write(coefficients.to_string(index=False))

    print("\n" + "=" * 65)
    print("RESULTS SAVED")
    print("=" * 65)

    print(f"\nOutput directory: {output_dir}")

    print("  - ridge_regression_coefficients.csv")
    print("  - ridge_regression_predictions.csv")
    print("  - ridge_cross_validation.csv")
    print("  - ridge_regression_summary.txt")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    args = parse_arguments()

    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)

    # Load dataset.
    train_df, test_df = load_data(data_path)

    # Separate predictors and target.
    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    # Cross-validation groups.
    groups = train_df[GROUP_COLUMN]

    # Train and tune Ridge regression.
    search, best_model = train_model(
        X_train,
        y_train,
        groups,
    )

    # Evaluate model.
    (
        train_predictions,
        test_predictions,
        train_metrics,
        test_metrics,
    ) = evaluate_model(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Save results.
    save_results(
        best_model,
        search,
        test_df,
        test_predictions,
        train_metrics,
        test_metrics,
        output_dir,
    )

    print("\nRidge regression completed successfully.")


if __name__ == "__main__":
    main()
