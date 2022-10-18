import argparse
import json
import logging
from pathlib import Path
from .utils import _get_or_create_mlflow_experiment_id
import mlflow

from .constants import MODEL_NAME

logging.basicConfig
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def evaluate(
    *,
    experiment_name: str,
    run_name: str = None,
    model_tags: list,
):
    experiment_id = _get_or_create_mlflow_experiment_id(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        log.info(f"Loading models by tags: {model_tags}")

        client = mlflow.MlflowClient()
        models = [client.get_model_version(MODEL_NAME, t) for t in model_tags]


        

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
    parser.add_argument("-tt", "--model_tags", type=int, required=True)
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
