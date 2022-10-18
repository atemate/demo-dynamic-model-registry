import argparse
import logging
from pathlib import Path

import mlflow

logging.basicConfig
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ee",
        "--experiment_names",
        type=str,
        nargs="+",
        help="Experiments to put to the dataframe",
    )

    parser.add_argument(
        "-f", "--format", choices=["json", "prettyjson", "md"], default="prettyjson"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    return parser.parse_args()


def main(args=None):
    # Process arguments
    args = args or get_args()
    experiment_names = args.experiment_names
    format, output = args.format, args.output

    if output.is_file():
        raise ValueError(f"Output file '{output}' already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare objects for experiments and pipelines
    log.info(f"Loading runs info on experiments: {experiment_names}")
    df = mlflow.search_runs(experiment_names=experiment_names, output_format="pandas")
    df = df.drop(columns=[c for c in df.columns if c.startswith("tags.")])

    if format == "json":
        s = df.to_json(orient="records")
    elif format == "prettyjson":
        s = df.to_json(orient="records", indent=4)
    elif format == "md":
        s = df.to_markdown()

    log.info(f"Writing to '{output}'")
    output.write_text(s)


if __name__ == "__main__":
    main()
