import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

from service.predict import FEATURES, build_features

FEATURES_PATH = Path("features/all_features.csv")
MODELS_DIR = Path("models")
METRICS_PATH = Path("reports/metrics.json")
BASELINE = "baseline_const"
CHAMPION_ALIAS = "champion"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

ESTIMATORS = {
    "lin_reg": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def data_version():
    with open("data.dvc", encoding="utf-8") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def load_dataset():
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["rooms", "area", "floor", "total_floors", "price"])
    X = np.array(
        [
            build_features(row.area, row.rooms, row.floor, row.total_floors)
            for row in df.itertuples()
        ],
        dtype=float,
    )
    y = df["price"].to_numpy(dtype=float)
    return X, y


def make_estimator(name, model_params, random_state):
    if name == BASELINE:
        return DummyRegressor(strategy="median")
    kwargs = dict(model_params or {})
    if name == "random_forest":
        kwargs["random_state"] = random_state
    return ESTIMATORS[name](**kwargs)


def evaluate(name, model_params, data, params):
    X_train, X_test, y_train, y_test = data
    estimator = make_estimator(name, model_params, params["random_state"])
    cv = KFold(
        n_splits=params["cv_folds"], shuffle=True, random_state=params["random_state"]
    )
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        },
        return_train_score=True,
    )
    estimator.fit(X_train, y_train)
    holdout_pred = estimator.predict(X_test)
    metrics = {
        "train_mae": float(-scores["train_mae"].mean()),
        "train_mape": float(-scores["train_mape"].mean()),
        "val_mae": float(-scores["test_mae"].mean()),
        "val_mae_std": float(scores["test_mae"].std()),
        "val_mape": float(-scores["test_mape"].mean()),
        "holdout_mae": float(mean_absolute_error(y_test, holdout_pred)),
        "holdout_mape": float(mean_absolute_percentage_error(y_test, holdout_pred)),
    }
    return estimator, metrics


def main():
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(params["experiment"])

    X, y = load_dataset()
    data = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )
    X_train, X_test = data[0], data[1]

    tags = {"git_commit": git_commit(), "data_version": data_version()}
    candidates = {BASELINE: {}, **params["models"]}
    MODELS_DIR.mkdir(exist_ok=True)

    results = {}
    versions = {}
    for name, model_params in candidates.items():
        with mlflow.start_run(run_name=name):
            estimator, metrics = evaluate(name, model_params, data, params)
            mlflow.set_tags({**tags, "model_name": name})
            mlflow.log_params(
                {
                    "model": name,
                    "test_size": params["test_size"],
                    "cv_folds": params["cv_folds"],
                    "random_state": params["random_state"],
                    "n_train": len(X_train),
                    "n_holdout": len(X_test),
                    "features": ",".join(FEATURES),
                    **(model_params or {}),
                }
            )
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(
                estimator,
                name="model",
                signature=infer_signature(X_train, estimator.predict(X_train)),
                input_example=X_train[:2],
                skops_trusted_types=TRUSTED_TYPES,
                registered_model_name=(
                    None if name == BASELINE else params["registered_model"]
                ),
            )
            versions[name] = info.registered_model_version
            joblib.dump(estimator, MODELS_DIR / f"{name}.pkl")
            results[name] = metrics
            print(
                f"{name}: val_mae={metrics['val_mae']:,.0f} "
                f"holdout_mae={metrics['holdout_mae']:,.0f} "
                f"holdout_mape={metrics['holdout_mape']:.3f}"
            )

    trained = [name for name in candidates if name != BASELINE]
    best = min(trained, key=lambda name: results[name]["val_mae"])
    beats_baseline = (
        results[best]["holdout_mae"] < results[BASELINE]["holdout_mae"]
    )
    if beats_baseline:
        MlflowClient().set_registered_model_alias(
            params["registered_model"], CHAMPION_ALIAS, versions[best]
        )
        print(f"champion: {best}, версия {versions[best]}")
    else:
        print(f"{best} не лучше baseline, алиас {CHAMPION_ALIAS} не обновлён")

    METRICS_PATH.parent.mkdir(exist_ok=True)
    report = {"best_model": best, "beats_baseline": bool(beats_baseline), **results}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
