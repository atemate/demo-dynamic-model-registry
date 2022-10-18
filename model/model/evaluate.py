import argparse
import logging
from pathlib import Path

import mlflow

from .utils import (
    apply_fun,
    dump_mlflow_info,
    get_or_create_mlflow_experiment_id,
    sigmoid,
)

log = logging.getLogger()


def evaluate(
    *,
    experiment_name: str,
    run_id: str = None,
    model_name: str = None,
    model_version: int = None,
    output_mlflow_json_file: Path = None,
):
    experiment_id = get_or_create_mlflow_experiment_id(experiment_name)
    client = mlflow.MlflowClient()

    if run_id:
        log.info(f"Loading model from run '{run_id}'")
        run = client.get_run(run_id)
        model = mlflow.MlflowClient().search_model_versions(f"run_id='{run_id}'")[0]
    else:
        assert model_name and model_version, (model_name, model_version)
        log.info(f"Loading model '{model_name}' version '{model_version}'")
        model = client.get_model_version(model_name, model_version)
    model_name = model.name
    log.info(f"Loaded model '{model_name}': {model}")

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

    # Log mlflow run info
    dump_mlflow_info(output_mlflow_json_file, experiment_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-m", "--model_name")
    parser.add_argument("-v", "--model_version", type=int)  # alternative to --run_id
    parser.add_argument("-r", "--run_id")  # alternative to --model_version
    parser.add_argument("--output_mlflow_json_file", type=Path)
    return parser.parse_args()


def main(args=None):
    args = args or get_args()
    args = get_args()
    log.info(f"Arguments: {args}")

    run_id = args.run_id
    model_name, model_version = args.model_name, args.model_version
    if not run_id and not (model_name or model_version):
        raise ValueError(
            "Either run_id or (model_name and model_version) must be provided"
        )

    evaluate(
        experiment_name=args.experiment_name,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        output_mlflow_json_file=args.output_mlflow_json_file,
    )


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.INFO)
    main()
