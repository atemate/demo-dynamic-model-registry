import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import sklearn.datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

log = logging.getLogger()


def get_or_create_mlflow_experiment_id(exp_name: str, use_legacy_api=False) -> str:
    """MLflow helper to get or create an experiment by name."""
    log.info(f"Getting experiment ID by name: '{exp_name}'")
    if use_legacy_api:
        log.info(f"Using legacy API: DagsHub does not support search_experiments()")
        # Legacy API: DagsHub does not support search_experiments()
        exps = [e for e in mlflow.list_experiments() if e.name == exp_name]
    else:
        exps = mlflow.search_experiments(filter_string=f"name='{exp_name}'")

    log.info(f"Found: {len(exps)} experiments: {exps}")
    exp_id = exps[0].experiment_id if exps else mlflow.create_experiment(exp_name)
    log.info(f"Experiment ID: {exp_id}")
    return exp_id


def dump_mlflow_info(
    output_mlflow_json_file: Path, experiment_name: str, run=None, run_name: str = None
):
    experiment_id = None
    if run is None:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None:
            experiment_id = exp.experiment_id
        run_id = None
    else:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

    # Log mlflow run info
    info = {
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_name": run_name,
        "run_id": run_id,
    }

    log.info(f"MLflow info: {info}")
    if output_mlflow_json_file:
        log.info(f"Writing MLflow run info to {output_mlflow_json_file}")
        output_mlflow_json_file.parent.mkdir(parents=True, exist_ok=True)
        output_mlflow_json_file.write_text(json.dumps(info, indent=4))


def load_data(train_size: float = 0.8, seed: int = 42):
    wine = sklearn.datasets.load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=seed
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, train_size=0.5, random_state=seed
    )
    log.info(f"X_train.shape: {X_train.shape}")
    log.info(f"X_test.shape: {X_test.shape}")
    log.info(f"X_val.shape: {X_val.shape}")

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}
