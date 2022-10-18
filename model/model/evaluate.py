import argparse
import logging

import mlflow

from .utils import _get_or_create_mlflow_experiment_id, apply_fun, sigmoid

logging.basicConfig
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def evaluate(
    *,
    experiment_name: str,
    model_name: str,
    model_version: int = None,
):
    experiment_id = _get_or_create_mlflow_experiment_id(experiment_name)
    client = mlflow.MlflowClient()

    log.info(f"Loading model '{model_name}' version '{model_version}'")
    model = client.get_model_version(model_name, model_version)

    log.info(f"Loading champion model verisons for '{model_name}'")
    champions = client.get_latest_versions(model_name, stages=["Production", "Staging"])

    models = list({m.version: m for m in [model] + champions}.values())
    log.info(f"Loaded {len(models)} models: {models}")

    for model in models:
        log.info(f"Evaluating model '{model_name}' version '{model.version}'")
        run_name = f"eval-{model_name}-v{model.version}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            # log model names
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", model.version)
            mlflow.log_param("model_stage", model.current_stage)

            log.info(f"Model run: {model.run_id}")
            run = mlflow.get_run(model.run_id)

            # Logging model parameters
            for name, value in run.data.params.items():
                mlflow.log_param(f"param.{name}", value)

            # Logging model training metrics
            for name, value in run.data.metrics.items():
                mlflow.log_param(f"train_metric.{name}", value)

            # # Evaluating sklearn model
            sk_model = mlflow.sklearn.load_model(model.source)
            predictions = sk_model.predict(...)
            print(predictions)

            # Calculating metrics
            metrics = {
                "roc_auc": apply_fun(sigmoid, predictions, 1),
                "precision": apply_fun(sigmoid, predictions, 2),
                "accuracy": apply_fun(sigmoid, predictions, 3),
                "f1": apply_fun(sigmoid, predictions, 4),
            }
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(f"metric.{name}", value)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-v", "--model_version", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    log.info(f"Arguments: {args}")

    evaluate(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        model_version=args.model_version,
    )
