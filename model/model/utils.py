import json
import logging
import math
from pathlib import Path

import mlflow

log = logging.getLogger()


def get_or_create_mlflow_experiment_id(exp_name: str, use_legacy_api=True) -> str:
    """MLflow helper to get or create an experiment by name."""
    if use_legacy_api:
        # Legacy API: DagsHub does not support search_experiments()
        exps = [e for e in mlflow.list_experiments() if e.name == exp_name]
    else:
        exps = mlflow.search_experiments(filter_string=f"name='{exp_name}'")

    return exps[0].experiment_id if exps else mlflow.create_experiment(exp_name)


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


def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return float(f"{y:.3f}")


def apply_fun(f, x, times=1):
    for _ in range(times):
        x = f(x)
    return x
