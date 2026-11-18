
"""
Multiple Linear Regression: College Football PPA Prediction

Run from the project root:
    python src/Multiple_regression.py

Optional:
    python src/Multiple_regression.py path/to/model_df.csv
    python src/Multiple_regression.py --output-dir outputs/my_ols_run

Outputs:
    CSV files, statistical summary, and diagnostic PNG graphs.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_df.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "multiple_regression"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

TARGET = "ppa"
SPLIT_COLUMN = "split"

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


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run multiple linear regression to predict PPA."
    )

    parser.add_argument(
        "data",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Dataset path.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for results and graphs.",
    )

    return parser.parse_args()


def resolve_path(path):

    path = Path(path)

    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()

    return path.resolve()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(data_path):

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Expected data/processed/model_df.csv."
        )

    df = pd.read_csv(data_path)

    required = FEATURES + [TARGET, SPLIT_COLUMN]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
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
            "Both 'train' and 'test' rows must exist in split."
        )

    print("\n" + "=" * 65)
    print("COLLEGE FOOTBALL MULTIPLE LINEAR REGRESSION")
    print("=" * 65)

    print(f"\nDataset: {data_path}")
    print(f"Total observations: {len(df):,}")
    print(f"Training observations: {len(train_df):,}")
    print(f"Testing observations:  {len(test_df):,}")
    print(f"Target: {TARGET}")

    return train_df, test_df


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def prepare_features(train_df, test_df):

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    # Numeric and binary features.
    for column in NUMERIC_FEATURES + BINARY_FEATURES:

        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

        fill_value = X_train[column].median()

        if pd.isna(fill_value):
            fill_value = 0

        X_train[column] = X_train[column].fillna(
            fill_value
        )

        X_test[column] = X_test[column].fillna(
            fill_value
        )

    # Categorical features.
    for column in CATEGORICAL_FEATURES:

        X_train[column] = X_train[column].astype("string")
        X_test[column] = X_test[column].astype("string")

        modes = X_train[column].mode(dropna=True)

        fill_value = (
            modes.iloc[0] if not modes.empty else "Unknown"
        )

        X_train[column] = (
            X_train[column]
            .fillna(fill_value)
            .astype(str)
        )

        X_test[column] = (
            X_test[column]
            .fillna(fill_value)
            .astype(str)
        )

    # One-hot encoding.
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

    # Align test columns with training columns.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Remove constant predictors.
    constant_columns = [
        column for column in X_train.columns
        if X_train[column].nunique() <= 1
    ]

    if constant_columns:

        print("\nRemoving constant predictors:")

        for column in constant_columns:
            print(f"  - {column}")

        X_train = X_train.drop(
            columns=constant_columns
        )

        X_test = X_test.drop(
            columns=constant_columns
        )

    print(
        f"\nRegression predictors: {len(X_train.columns)}"
    )

    return X_train, X_test


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

def train_model(X_train, y_train):

    X_train = sm.add_constant(
        X_train,
        has_constant="add",
    )

    matrix_rank = np.linalg.matrix_rank(
        X_train.to_numpy()
    )

    if matrix_rank < X_train.shape[1]:

        warnings.warn(
            "The design matrix is rank-deficient. "
            "Some coefficients may not be uniquely determined.",
            RuntimeWarning,
        )

        print(
            f"WARNING: Matrix rank {matrix_rank}; "
            f"design matrix columns {X_train.shape[1]}."
        )

    model = sm.OLS(
        y_train,
        X_train,
    ).fit(cov_type="HC3")

    return model, X_train


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    X_test = sm.add_constant(
        X_test,
        has_constant="add",
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    metrics = {
        "train": {
            "R_squared": r2_score(
                y_train,
                train_predictions,
            ),
            "Adjusted_R_squared": model.rsquared_adj,
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_train,
                    train_predictions,
                )
            ),
            "MAE": mean_absolute_error(
                y_train,
                train_predictions,
            ),
        },
        "test": {
            "R_squared": r2_score(
                y_test,
                test_predictions,
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test,
                    test_predictions,
                )
            ),
            "MAE": mean_absolute_error(
                y_test,
                test_predictions,
            ),
        },
    }

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE")
    print("=" * 65)

    for split_name, split_metrics in metrics.items():

        print(f"\n{split_name.upper()} SET")

        for name, value in split_metrics.items():
            print(f"{name:<22}: {value:.4f}")

    return test_predictions, metrics


# ---------------------------------------------------------
# SAVE TABLES
# ---------------------------------------------------------

def save_tables(
    model,
    test_df,
    y_test,
    test_predictions,
    output_dir,
):

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
        "p_value"
    )

    id_columns = [
        column for column in [
            "id",
            "gameId",
            "offense",
            "defense",
        ]
        if column in test_df.columns
    ]

    predictions = test_df[id_columns].copy()

    predictions["actual_ppa"] = np.asarray(y_test)
    predictions["predicted_ppa"] = np.asarray(
        test_predictions
    )

    predictions["residual"] = (
        predictions["actual_ppa"]
        - predictions["predicted_ppa"]
    )


    with open(
        output_dir / "ppa_regression_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(model.summary().as_text())

    return coefficients, predictions


# ---------------------------------------------------------
# GRAPH GENERATION
# ---------------------------------------------------------

def make_diagnostic_plots(
    y_test,
    test_predictions,
    coefficients,
    output_dir,
):

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(test_predictions, dtype=float)

    residuals = actual - predicted

    # -----------------------------------------------------
    # GRAPH 1: ACTUAL VS PREDICTED
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        actual,
        predicted,
        alpha=0.65,
        edgecolors="none",
    )

    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    ax.set(
        title="OLS: Actual vs. Predicted PPA (Test Set)",
        xlabel="Actual PPA",
        ylabel="Predicted PPA",
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "actual_vs_predicted.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 2: RESIDUALS VS PREDICTED
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        predicted,
        residuals,
        alpha=0.65,
        edgecolors="none",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=2,
    )

    ax.set(
        title="OLS: Residuals vs. Predicted PPA",
        xlabel="Predicted PPA",
        ylabel="Residual (Actual - Predicted)",
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "residuals_vs_predicted.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 3: RESIDUAL DISTRIBUTION
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        residuals,
        bins=30,
        edgecolor="black",
        alpha=0.8,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=2,
    )

    ax.set(
        title="OLS: Test Residual Distribution",
        xlabel="Residual (Actual - Predicted)",
        ylabel="Number of Plays",
    )

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "residual_distribution.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 4: COEFFICIENT EFFECTS
    # -----------------------------------------------------

    significant = coefficients[
        (coefficients["feature"] != "const")
        & (coefficients["p_value"] < 0.05)
    ].copy()

    if significant.empty:

        significant = coefficients[
            coefficients["feature"] != "const"
        ].head(12).copy()

        chart_title = (
            "OLS: Largest Coefficients "
            "(No p < 0.05 Predictors)"
        )

    else:

        significant = significant.head(15)

        chart_title = (
            "OLS: Statistically Significant "
            "Coefficients (p < 0.05)"
        )

    if not significant.empty:

        significant = significant.sort_values(
            "coefficient"
        )

        fig, ax = plt.subplots(
            figsize=(
                9,
                max(5, 0.42 * len(significant)),
            )
        )

        errors = np.vstack([
            significant["coefficient"]
            - significant["ci_lower"],

            significant["ci_upper"]
            - significant["coefficient"],
        ])

        ax.errorbar(
            significant["coefficient"],
            significant["feature"],
            xerr=errors,
            fmt="o",
            capsize=3,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1.5,
        )

        ax.set(
            title=chart_title,
            xlabel="Coefficient (95% confidence interval)",
            ylabel="Predictor",
        )

        ax.grid(axis="x", alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            graph_dir / "coefficient_effects.png",
            dpi=200,
        )

        plt.close(fig)

    print(f"\nGraphs saved in: {graph_dir}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    args = parse_arguments()

    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load data.
    train_df, test_df = load_data(data_path)

    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    # Prepare predictors.
    X_train, X_test = prepare_features(
        train_df,
        test_df,
    )

    # Train model.
    model, X_train = train_model(
        X_train,
        y_train,
    )

    print("\n" + "=" * 65)
    print("OLS REGRESSION SUMMARY")
    print("=" * 65)

    print(model.summary())

    # Evaluate model.
    test_predictions, metrics = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Save CSV and summary files.
    coefficients, _ = save_tables(
        model,
        test_df,
        y_test,
        test_predictions,
        output_dir,
    )

    # Generate graphs.
    make_diagnostic_plots(
        y_test,
        test_predictions,
        coefficients,
        output_dir,
    )

    print("\nRESULTS SAVED")
    print(f"Output directory: {output_dir}")
    print("  - ppa_regression_summary.txt")
    print("  - graphs/")

    print("\nMultiple linear regression completed successfully.")


if __name__ == "__main__":
    main()
