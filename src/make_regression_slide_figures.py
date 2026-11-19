"""
Builds the small, single-panel regression figures used in the term-project slides.

Run from the project root:
    python src/make_regression_slide_figures.py

Refits the tuned XGBoost and neural network models using the settings chosen in
notebooks/06_regression_xgboost.ipynb and notebooks/07_regression_neural_network.ipynb,
so the numbers match those notebooks. Saves PNGs to reports/figures/slides/.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# same colors as the regression notebooks
COLORS = {"Null Model": "tab:gray", "XGBoost": "tab:green", "Neural Network": "tab:purple"}

df = pd.read_csv(ROOT / "data" / "processed" / "model_df.csv").dropna(subset=["ppa"])
info = json.load(open(ROOT / "data" / "processed" / "regression_feature_lists_after_eda.json"))["regression"]
features, numeric, categorical, binary = info["features"], info["numeric"], info["categorical"], info["binary"]
train, test = df[df.split == "train"], df[df.split == "test"]
Xtr, ytr, gtr = train[features], train["ppa"], train["gameId"]
Xte, yte = test[features], test["ppa"]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# ---------------------------------------------------------------- models
null = DummyRegressor(strategy="mean").fit(Xtr, ytr)

xgb_params = {"subsample": 0.7, "reg_lambda": 50, "reg_alpha": 2, "n_estimators": 200, "min_child_weight": 20,
              "max_depth": 4, "learning_rate": 0.02, "colsample_bytree": 0.5}
xgb = XGBRegressor(objective="reg:pseudohubererror", base_score=float(ytr.median()), random_state=SEED,
                   n_jobs=1, **xgb_params)

binary_missing = ["prev_call_1", "prev_call_2", "prev_call_3"]
binary_ok = [c for c in binary if c not in binary_missing]
pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("bin_m", SimpleImputer(strategy="most_frequent", add_indicator=True), binary_missing),
    ("bin", "passthrough", binary_ok),
])
nn = TransformedTargetRegressor(
    regressor=Pipeline([("pre", pre), ("reg", MLPRegressor(hidden_layer_sizes=(16,), activation="relu", alpha=10.0,
                                                           max_iter=200, solver="adam", learning_rate_init=0.001,
                                                           early_stopping=False, n_iter_no_change=1000,
                                                           random_state=SEED))]),
    transformer=StandardScaler())

models = {"XGBoost": xgb, "Neural Network": nn}
for m in models.values():
    m.fit(Xtr, ytr)

# grouped CV RMSE (unseen games), same folds as the notebooks
cv = GroupKFold(n_splits=5)
from sklearn.base import clone

cv_rmse, null_cv = {}, []
for name, m in models.items():
    scores = []
    for tr, va in cv.split(Xtr, ytr, gtr):
        mm = clone(m).fit(Xtr.iloc[tr], ytr.iloc[tr])
        scores.append(rmse(ytr.iloc[va], mm.predict(Xtr.iloc[va])))
    cv_rmse[name] = float(np.mean(scores))
null_cv = float(np.mean([rmse(ytr.iloc[va], np.full(len(va), ytr.iloc[tr].mean()))
                         for tr, va in cv.split(Xtr, ytr, gtr)]))

summary = {}
for name, m in models.items():
    summary[name] = {"train_rmse": rmse(ytr, m.predict(Xtr)), "cv_rmse": cv_rmse[name],
                     "test_rmse": rmse(yte, m.predict(Xte)), "test_r2": float(r2_score(yte, m.predict(Xte)))}
summary["Null Model"] = {"train_rmse": rmse(ytr, null.predict(Xtr)), "cv_rmse": null_cv,
                         "test_rmse": rmse(yte, null.predict(Xte)), "test_r2": float(r2_score(yte, null.predict(Xte)))}
print(json.dumps(summary, indent=2))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- actual vs predicted
for name, tag in [("XGBoost", "gb"), ("Neural Network", "nn")]:
    pred = models[name].predict(Xte)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(pred, yte, alpha=0.4, s=14, color=COLORS[name])
    lim = [yte.min(), yte.max()]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Predicted PPA"); ax.set_ylabel("Actual PPA")
    ax.set_title(f"{name}: Actual vs. Predicted PPA (Test Set)", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"{tag}_actual_vs_predicted.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- does it overfit? (train / CV / test RMSE vs null)
for name, tag in [("XGBoost", "gb"), ("Neural Network", "nn")]:
    labels = ["Train", "Cross-validation\n(unseen games)", "Test\n(unseen games)"]
    keys = ["train_rmse", "cv_rmse", "test_rmse"]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b1 = ax.bar(x - w / 2, [summary["Null Model"][k] for k in keys], w, color=COLORS["Null Model"], label="Null Model")
    b2 = ax.bar(x + w / 2, [summary[name][k] for k in keys], w, color=COLORS[name], label=name)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(1.15, 1.40); ax.set_ylabel("RMSE (lower is better)")
    ax.set_title(f"{name} vs. Null Model: RMSE", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / f"{tag}_rmse_vs_null.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- permutation importance, both models side by side
perm = {}
for name, m in models.items():
    r = permutation_importance(m, Xte, yte, scoring="neg_root_mean_squared_error", n_repeats=30,
                               random_state=SEED, n_jobs=1)
    perm[name] = pd.Series(r.importances_mean, index=features)
perm_df = pd.DataFrame(perm)
perm_df.round(5).to_csv(ROOT / "reports" / "tables" / "regression_permutation_importance_xgb_vs_nn.csv")

top = perm_df.mean(axis=1).sort_values(ascending=False).head(8).index[::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(top)); h = 0.38
ax.barh(y - h / 2, perm_df.loc[top, "XGBoost"], h, color=COLORS["XGBoost"], label="XGBoost")
ax.barh(y + h / 2, perm_df.loc[top, "Neural Network"], h, color=COLORS["Neural Network"], label="Neural Network")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(top)
ax.set_xlabel("Increase in test RMSE when the feature is shuffled")
ax.set_title("Permutation importance (test games)", fontsize=11)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "regression_importance_xgb_vs_nn.png", dpi=200); plt.close(fig)

print("saved to", OUT)
