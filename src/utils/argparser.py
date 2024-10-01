import os
import yaml
from typing import Dict, Union

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