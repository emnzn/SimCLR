import os
import yaml
from typing import Dict, Union, Any

def get_args(args_path: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    args_path: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    return args


def get_encoder_args(run_dir: str) -> Dict[str, Union[float, str]]:
    """
    Reads the arguments passed to a given run, 
    and returns only arguments relevant to the encoder.
    """

    args = get_args(run_dir)
    encoder_args = ["backbone", "hidden_dim", "projection_dim"]
    encoder_args = {k: v for k, v in args.items() if k in encoder_args}
    
    return encoder_args


def get_pretrain_lr(args: Dict[str, Any]):
    if args["lr_scaling_method"] == "square-root":
        lr = 0.075 * (args["batch_size"] ** 0.5)

    elif args["lr_scaling_method"] == "linear":
        lr = 0.3 * (args["batch_size"] / 256)

    return lr


def save_args(
    args: Dict[str, Union[float, str]], 
    dest_dir: str
    ):

    """
    Saves the arguments as a yaml file in a given destination directory.
    """

    path = os.path.join(dest_dir, "run-config.yaml")
    with open(path, "w") as f:
        yaml.dump(args, f)