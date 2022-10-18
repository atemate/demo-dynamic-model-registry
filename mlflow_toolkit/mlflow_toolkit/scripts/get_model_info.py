import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_id",
        type=str,
        help="Run ID to get associated model from",
    )

    parser.add_argument("-f", "--format", choices=["json", "md"], default="json")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    return parser.parse_args()


def main(args=None):
    # Process arguments
    args = args or get_args()
    run_id = args.run_id
    format, output = args.format, args.output

    if output.is_file():
        raise ValueError(f"Output file '{output}' already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare objects for experiments and pipelines
    log.info(f"Loading model info on run id: '{run_id}'")
    run = mlflow.get_run(run_id)
    model = mlflow.MlflowClient().search_model_versions(f"run_id='{run_id}'")[0]

    info = {
        "name": model.name,
        "version": model.version,
        "status": model.status,
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "artifact_uri": run.info.artifact_uri,
        **{f"metric.{k}": v for k, v in run.data.metrics.items()},
        **{f"param.{k}": v for k, v in run.data.params.items()},
    }

    if format == "json":
        s = json.dumps(info, indent=4)
    elif format == "md":
        df = pd.DataFrame.from_dict(info, orient="index")
        s = df.to_markdown()

    log.info(f"Writing to '{output}'")
    output.write_text(s)


if __name__ == "__main__":
    logging.basicConfig()
    log.setLevel(logging.INFO)
    main()
