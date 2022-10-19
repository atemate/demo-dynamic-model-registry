import argparse
import logging
from pathlib import Path

import mlflow

from .utils import (
    dump_mlflow_info,
    eval_metrics,
    get_or_create_mlflow_experiment_id,
    load_data,
)

log = logging.getLogger()


def evaluate(
    *,
    experiment_name: str,
    run_id: str = None,
    model_name: str = None,
    model_version: int = None,
    compare_champions: bool = True,
    output_mlflow_json_file: Path = None,
):
    experiment_id = get_or_create_mlflow_experiment_id(
        experiment_name, use_legacy_api=True
    )
    client = mlflow.MlflowClient()

    if run_id:
        log.info(f"Loading model from run '{run_id}'")
        run = client.get_run(run_id)
        model = mlflow.MlflowClient().search_model_versions(f"run_id='{run_id}'")[0]
    else:
        assert model_name and model_version, (model_name, model_version)
        log.info(f"Loading model '{model_name}' version '{model_version}'")
        model = client.get_model_version(model_name, model_version)
        run = client.get_run(model.run_id)

    params = run.data.params
    model_name = model.name
    log.info(f"Loaded model '{model_name}': {model}, params {params}")

    if compare_champions:
        log.info(f"Loading champion model verisons for '{model_name}'")
        champions = client.get_latest_versions(
            model_name, stages=["Production", "Staging"]
        )
        models = list({m.version: m for m in [model] + champions}.values())
        log.info(f"Loaded {len(models)} models: {models}")
    else:
        models = [model]

    seed = int(params.get("seed", params.get("random_state", 42)))
    # load data for the model being evaluated. Careful with data leakage!
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data(
        train_size=float(params.get("data_train_size", 0.8)),
        seed=seed,
    )

    for model in models:

        log.info(f"Evaluating model '{model_name}' version '{model.version}'")
        run_name = f"eval-{model_name}-v{model.version}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            log.info(f"Experiment started: {experiment_id}: {run_name}")

            # log model names
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_version", model.version)
            mlflow.log_param("model_stage", model.current_stage)

            log.info(f"Model run: {model.run_id}")
            run = mlflow.get_run(model.run_id)

            # Logging model parameters
            for name, value in run.data.params.items():
                mlflow.log_param(name, value)

            # Evaluating sklearn model
            net = mlflow.sklearn.load_model(model.source)
            predicted_qualities = net.predict(X_val)
            metrics = eval_metrics(y_val, predicted_qualities)

            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

    # Log mlflow run info
    dump_mlflow_info(output_mlflow_json_file, experiment_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-m", "--model_name")
    parser.add_argument("-v", "--model_version", type=int)  # alternative to --run_id
    parser.add_argument("-r", "--run_id")  # alternative to --model_version
    parser.add_argument("--compare_champions", type=bool, default=True)
    parser.add_argument("--output_mlflow_json_file", type=Path)
    return parser.parse_args()


def main(args=None):
    args = args or get_args()
    args = get_args()
    log.info(f"Arguments: {args}")

    run_id = args.run_id
    model_name, model_version = args.model_name, args.model_version
    if run_id and (model_name or model_version):
        raise ValueError(
            f"Provide either run_id or (model_name and model_version), not both"
        )
    if not run_id and not model_name and not model_version:
        raise ValueError(
            "Either run_id or (model_name and model_version) must be provided"
        )

    evaluate(
        experiment_name=args.experiment_name,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        compare_champions=args.compare_champions,
        output_mlflow_json_file=args.output_mlflow_json_file,
    )


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.INFO)
    main()
