import argparse
import json
import logging
import math
from pathlib import Path

import mlflow

logging.basicConfig
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def evaluate(
    *,
    experiment_name: str,
    run_name: str = None,
    models: list,
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
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": experiment_name,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "run_id": run.info.run_id,
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
