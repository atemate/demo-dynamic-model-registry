import argparse
import json
import logging
import math
from pathlib import Path

import mlflow

logging.basicConfig
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


MODEL_NAME = "my-model"


def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return float(f"{y:.3f}")


class MockModel:
    def __init__(self, learning_rate: float, max_depth: int, n_estimators: int):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        # Log parameters
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("max_depth", self.max_depth)
        mlflow.log_param("n_estimators", self.n_estimators)

    def fit(self, *args, **kwargs):
        # assign random values to metrics
        roc_auc = sigmoid(
            sigmoid(self.learning_rate * self.max_depth * self.n_estimators)
        )
        precision = sigmoid(roc_auc)
        accuracy = sigmoid(precision)
        f1 = sigmoid(accuracy)

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

    def predict(self, *args, **kwargs):
        pass


def _get_or_create_mlflow_experiment_id(exp_name: str, use_legacy_api=False) -> str:
    if use_legacy_api:
        exps = [e for e in mlflow.list_experiments() if e.name == exp_name]
    else:
        exps = mlflow.search_experiments(filter_string=f"name='{exp_name}'")

    if exps:
        exp_id = exps[0].experiment_id
    else:
        exp_id = mlflow.create_experiment(exp_name)

    return exp_id


def train(
    *,
    experiment_name: str,
    run_name: str = None,
    model_params: dict,
    output_mlflow_json_file: Path = None,
):
    experiment_id = _get_or_create_mlflow_experiment_id(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # train the model ...
        log.info(f"Training model with params: {model_params}")
        model = MockModel(**model_params)
        model.fit()

        # Log model
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        # Log mlflow run info
        info = {
            "mlflow": {
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_name": experiment_name,
                "experiment_id": run.info.experiment_id,
                "run_name": run_name,
                "run_id": run.info.run_id,
            }
        }
        log.info(f"MLflow info: {info}")
        if output_mlflow_json_file:
            log.info(f"Writing MLflow run info to {output_mlflow_json_file}")
            output_mlflow_json_file.parent.mkdir(parents=True, exist_ok=True)
            output_mlflow_json_file.write_text(json.dumps(info, indent=4))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-r", "--run_name")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--output_mlflow_json_file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    log.info(f"Arguments: {args}")

    model_params = dict(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
    )
    train(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_params=model_params,
        output_mlflow_json_file=args.output_mlflow_json_file,
    )
