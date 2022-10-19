import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import ElasticNet

from .utils import (
    dump_mlflow_info,
    eval_metrics,
    get_or_create_mlflow_experiment_id,
    load_data,
)

log = logging.getLogger()


def train(
    *,
    experiment_name: str,
    run_name: str = None,
    model_name: str,
    alpha: float,
    l1_ratio: float,
    data_train_size: float,
    seed: int = 42,
    output_mlflow_json_file: Path = None,
):
    # mlflow.sklearn.autolog()

    experiment_id = get_or_create_mlflow_experiment_id(
        experiment_name, use_legacy_api=True
    )

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data(
        train_size=data_train_size, seed=seed
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:

        net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        net.fit(X_train, y_train)

        predicted_qualities = net.predict(X_test)
        metrics = eval_metrics(y_test, predicted_qualities)

        log.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        log.info("  RMSE: %s" % metrics["rmse"])
        log.info("  MAE: %s" % metrics["mae"])
        log.info("  R2: %s" % metrics["r2"])

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data_train_size", data_train_size)
        mlflow.log_param("seed", seed)

        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("mae", metrics["mae"])

        mlflow.sklearn.log_model(net, "model", registered_model_name=model_name)

        # Log mlflow run info
        dump_mlflow_info(output_mlflow_json_file, experiment_name, run, run_name)


def get_args():
    parser = argparse.ArgumentParser()
    # mlflow params:
    parser.add_argument("-e", "--experiment_name", required=True)
    parser.add_argument("-r", "--run_name")
    parser.add_argument("-m", "--model_name", required=True)
    # model params:
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    # data params:
    parser.add_argument("--data_train_size", type=float, default=0.75)
    # system params:
    parser.add_argument("--seed", default=42)
    parser.add_argument("--output_mlflow_json_file", type=Path)
    return parser.parse_args()


def main(args=None):
    args = args or get_args()
    log.info(f"Arguments: {args}")
    np.random.seed(args.seed)

    train(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_name=args.model_name,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        data_train_size=args.data_train_size,
        seed=args.seed,
        output_mlflow_json_file=args.output_mlflow_json_file,
    )


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.INFO)
    main()
