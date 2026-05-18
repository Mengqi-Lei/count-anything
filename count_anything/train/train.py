"""Public CountAnything training entrypoint.

This module delegates launcher mechanics to the internal SAM3 training
implementation while exposing a CountAnything-facing CLI.
"""

import os
from argparse import ArgumentParser
from typing import Sequence

from hydra import initialize_config_dir, initialize_config_module

from sam3.train.train import main as _sam3_train_main
from sam3.train.utils.train_utils import register_omegaconf_resolvers


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train or evaluate CountAnything using a Hydra config."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help=(
            "Path to a config file, e.g. config/count_anything_train_cloc.yaml, "
            "config/count_anything_val_cloc.yaml, or config/count_anything_test_cloc.yaml."
        ),
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="Whether to launch on a cluster, 0: run locally, 1: run on a cluster.",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="Number of GPUs per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    return parser


def cli_main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    if os.path.isfile(args.config):
        config_path = os.path.abspath(args.config)
        initialize_config_dir(config_dir=os.path.dirname(config_path), version_base="1.2")
        args.config = os.path.splitext(os.path.basename(config_path))[0]
    else:
        # Fallback for package-style config names.
        initialize_config_module("sam3.train", version_base="1.2")

    register_omegaconf_resolvers()
    _sam3_train_main(args)


if __name__ == "__main__":
    cli_main()
