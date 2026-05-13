import json
import argparse
from trainer import train
import os
from datetime import datetime


def report_done(results_path):
    queue_file = os.path.expanduser("~/signals/exp_queue")
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(queue_file), exist_ok=True)
    with open(queue_file, "a") as f:
        f.write(f"{exp_id} {results_path}\n")
    print(f"[完成] {exp_id} -> {results_path}")


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)
    report_done(os.path.abspath("./logs/") + "/")


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorithms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/finetune.json",
        help="Json file of settings.",
    )

    return parser


if __name__ == "__main__":
    main()
