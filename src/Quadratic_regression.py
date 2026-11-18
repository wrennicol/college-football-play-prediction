
"""
Quadratic Regression: College Football PPA Prediction

Run from the project root:
    python src/Quadratic_regression.py

Optional:
    python src/Quadratic_regression.py path/to/model_df.csv
    python src/Quadratic_regression.py --output-dir outputs/quadratic_regression

The model expands numeric predictors into squared terms and pairwise
interactions, while retaining binary and categorical predictors.

Outputs:
    - quadratic_regression_coefficients.csv
    - quadratic_regression_predictions.csv
    - quadratic_regression_summary.txt
    - graphs/actual_vs_predicted.png
    - graphs/residuals_vs_predicted.png
    - graphs/residual_distribution.png
    - graphs/top_coefficients.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_df.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "quadratic_regression"
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
# ARGUMENTS AND PATHS
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run quadratic regression to predict PPA."
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
        help="Directory for model outputs and graphs.",
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

    required = FEATURES + [
        TARGET,
        SPLIT_COLUMN,
    ]

    missing = [
        column
        for column in required
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

    df = df.dropna(
        subset=[TARGET]
    ).copy()

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

    print("\n" + "=" * 68)
    print("COLLEGE FOOTBALL QUADRATIC REGRESSION")
    print("=" * 68)

    print(f"\nDataset: {data_path}")
    print(f"Total observations: {len(df):,}")
    print(f"Training observations: {len(train_df):,}")
    print(f"Testing observations:  {len(test_df):,}")
    print(f"Target: {TARGET}")

    return train_df, test_df


# ---------------------------------------------------------
# BUILD QUADRATIC REGRESSION PIPELINE
# ---------------------------------------------------------

def build_pipeline():

    """
    Create quadratic terms for numeric predictors:

    - Original numeric terms
    - Squared terms
    - Pairwise numeric interactions

    Binary and categorical predictors remain as main effects.
    """

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
            (
                "quadratic",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
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
                "numeric_quadratic",
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
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regression",
                LinearRegression(),
            ),
        ]
    )


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def train_model(X_train, y_train):

    model = build_pipeline()

    model.fit(
        X_train,
        y_train,
    )

    transformed = (
        model.named_steps["preprocessor"]
        .transform(X_train)
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    rank = np.linalg.matrix_rank(transformed)
    n_features = transformed.shape[1]

    print("\n" + "=" * 68)
    print("QUADRATIC MODEL DETAILS")
    print("=" * 68)

    print(f"Expanded predictor count: {n_features}")
    print(f"Design matrix rank: {rank}")

    if rank < n_features:
        print(
            "WARNING: The expanded design matrix is rank-deficient. "
            "Some coefficients may not be uniquely identifiable."
        )

    return model


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "R_squared": r2_score(
            y_true,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        ),
        "MAE": mean_absolute_error(
            y_true,
            predictions,
        ),
    }


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

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

    print("\n" + "=" * 68)
    print("MODEL PERFORMANCE")
    print("=" * 68)

    for label, metrics in [
        ("TRAINING", train_metrics),
        ("TESTING", test_metrics),
    ]:

        print(f"\n{label} SET")

        for name, value in metrics.items():
            print(f"{name:<15}: {value:.4f}")

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
    regression = model.named_steps["regression"]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": regression.coef_,
    })

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    return coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

def save_outputs(
    model,
    test_df,
    y_test,
    test_predictions,
    train_metrics,
    test_metrics,
    output_dir,
):

    coefficients = get_coefficients(model)


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

    predictions["actual_ppa"] = np.asarray(y_test)

    predictions["predicted_ppa"] = np.asarray(
        test_predictions
    )

    predictions["residual"] = (
        predictions["actual_ppa"]
        - predictions["predicted_ppa"]
    )


    regression = model.named_steps["regression"]

    with open(
        output_dir / "quadratic_regression_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "COLLEGE FOOTBALL QUADRATIC REGRESSION\n"
        )

        file.write("=" * 68 + "\n\n")

        file.write(f"Target: {TARGET}\n")

        file.write(
            "Numeric predictors expanded to degree-2 polynomial terms.\n"
        )

        file.write(
            "Binary and categorical predictors included as main effects.\n"
        )

        file.write(
            f"Number of expanded predictors: {len(regression.coef_)}\n\n"
        )

        for label, metrics in [
            ("TRAINING", train_metrics),
            ("TESTING", test_metrics),
        ]:

            file.write(f"{label} PERFORMANCE\n")
            file.write("-" * 68 + "\n")

            for name, value in metrics.items():
                file.write(
                    f"{name}: {value:.4f}\n"
                )

            file.write("\n")

        file.write("COEFFICIENTS\n")
        file.write("-" * 68 + "\n")

        file.write(
            coefficients.to_string(index=False)
        )

    return coefficients, predictions


# ---------------------------------------------------------
# GRAPH GENERATION
# ---------------------------------------------------------

def generate_graphs(
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

    actual = np.asarray(
        y_test,
        dtype=float,
    )

    predicted = np.asarray(
        test_predictions,
        dtype=float,
    )

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

    low = min(
        actual.min(),
        predicted.min(),
    )

    high = max(
        actual.max(),
        predicted.max(),
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    ax.set_title(
        "Quadratic Regression: Actual vs. Predicted PPA"
    )

    ax.set_xlabel("Actual PPA")
    ax.set_ylabel("Predicted PPA")

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

    ax.set_title(
        "Quadratic Regression: Residuals vs. Predicted PPA"
    )

    ax.set_xlabel("Predicted PPA")

    ax.set_ylabel(
        "Residual (Actual - Predicted)"
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

    ax.set_title(
        "Quadratic Regression: Residual Distribution"
    )

    ax.set_xlabel(
        "Residual (Actual - Predicted)"
    )

    ax.set_ylabel("Number of Plays")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "residual_distribution.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 4: TOP COEFFICIENTS
    # -----------------------------------------------------

    top = coefficients.head(20).sort_values(
        "coefficient"
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(6, 0.35 * len(top)),
        )
    )

    ax.barh(
        top["feature"],
        top["coefficient"],
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_title(
        "Quadratic Regression: 20 Largest Coefficients"
    )

    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Expanded predictor")

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        graph_dir / "top_coefficients.png",
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

    # Separate predictors and target.
    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    # Train model.
    model = train_model(
        X_train,
        y_train,
    )

    # Evaluate model.
    (
        train_predictions,
        test_predictions,
        train_metrics,
        test_metrics,
    ) = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Save results.
    coefficients, _ = save_outputs(
        model,
        test_df,
        y_test,
        test_predictions,
        train_metrics,
        test_metrics,
        output_dir,
    )

    # Generate graphs.
    generate_graphs(
        y_test,
        test_predictions,
        coefficients,
        output_dir,
    )

    print("\nRESULTS SAVED")

    print(f"Output directory: {output_dir}")

    print("  - quadratic_regression_summary.txt")
    print("  - graphs/")

    print(
        "\nQuadratic regression completed successfully."
    )


if __name__ == "__main__":
    main()
