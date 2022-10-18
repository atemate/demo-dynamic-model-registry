import math

import mlflow


def _get_or_create_mlflow_experiment_id(exp_name: str, use_legacy_api=True) -> str:
    """MLflow helper to get or create an experiment by name."""
    if use_legacy_api:
        # Legacy API: DagsHub does not support search_experiments()
        exps = [e for e in mlflow.list_experiments() if e.name == exp_name]
    else:
        exps = mlflow.search_experiments(filter_string=f"name='{exp_name}'")

    return exps[0].experiment_id if exps else mlflow.create_experiment(exp_name)


def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return float(f"{y:.3f}")


def apply_fun(f, x, times=1):
    for _ in range(times):
        x = f(x)
    return x
