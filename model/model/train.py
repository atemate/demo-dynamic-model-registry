import argparse
import logging
from pathlib import Path

import mlflow

from .utils import dump_mlflow_info, get_or_create_mlflow_experiment_id, sigmoid

log = logging.getLogger()


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
        # return mock value
        mock_value = self.learning_rate * self.max_depth * self.n_estimators
        return sigmoid(1 - sigmoid(mock_value))


def train(
    *,
    experiment_name: str,
    run_name: str = None,
    model_name: str,
    model_params: dict,
    output_mlflow_json_file: Path = None,
):
    experiment_id = get_or_create_mlflow_experiment_id(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # train the model ...
        log.info(f"Training model with params: {model_params}")
        model = MockModel(**model_params)
        model.fit()

        # Log model
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=model_name
        )

        # Log mlflow run info
        dump_mlflow_info(output_mlflow_json_file, experiment_name, run, run_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-r", "--run_name")
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--output_mlflow_json_file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.INFO)

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
        model_name=args.model_name,
        model_params=model_params,
        output_mlflow_json_file=args.output_mlflow_json_file,
    )
