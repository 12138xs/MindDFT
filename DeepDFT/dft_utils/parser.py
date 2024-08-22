import os
import sys
import argparse
import json

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/qm9", help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )

    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Enable equivariant message passing model (PaiNN)"
    )

    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, disable periodic boundary conditions (force to False) in atoms data",
    )

    parser.add_argument(
        "--force_pbc",
        action="store_true",
        help="If flag is given, force periodic boundary conditions to True in atoms data",
    )

    return parser.parse_args(arg_list)

def save_cmdargs(output_dir, file_name)->None:
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("\n".join(sys.argv[1:]))

def save_parsed_cmdargs(output_dir, file_name, args)->None:
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(vars(args), f)



