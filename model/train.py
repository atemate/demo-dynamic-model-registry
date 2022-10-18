import argparse
import tempfile
import joblib
import mlflow
import math
import logging

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

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


def _mlflow_search_experiments(exp_name: str, use_legacy_api=False) -> str:
    if use_legacy_api:
        return [e for e in mlflow.list_experiments() if e.name == exp_name]
    else:
        return mlflow.search_experiments(filter_string=f"name='{exp_name}'")


def _get_or_create_mlflow_experiment_id(exp_name: str) -> str:
    exps = _mlflow_search_experiments(exp_name, use_legacy_api=True)
    if exps:
        exp_id = exps[0].experiment_id
    else:
        exp_id = mlflow.create_experiment(exp_name)
    return exp_id


def train( experiment_name: str, run_name: str, model_params: dict):
    experiment_id = _get_or_create_mlflow_experiment_id(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # train the model ...
        log.info(f"Training model with params: {model_params}")
        model = MockModel(**model_params)

        # Log parameters
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)

        # Train mock model, assign random values to metrics:
        model.fit()
        roc_auc = sigmoid(sigmoid(model.learning_rate * model.max_depth * model.n_estimators))
        precision = sigmoid(roc_auc)
        accuracy = sigmoid( precision)
        f1 = sigmoid(accuracy)
        log.info(f"roc_auc={roc_auc}")
        log.info(f"precision={precision}")
        log.info(f"accuracy={accuracy}")
        log.info(f"f1={f1}")

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Log artifacts
        model_path = f"{tempfile.mkdtemp()}/model.joblib"
        log.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Log model
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-r", "--run_name")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    log.info(f"Arguments: {args}")

    model_params = dict(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
    )
    train(experiment_name=args.experiment_name,  run_name=args.run_name, model_params=model_params)
