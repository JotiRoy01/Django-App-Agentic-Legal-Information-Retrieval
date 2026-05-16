import yaml
import os, sys
from pathlib import Path
from agentic.exception import Agentic_Exception
from agentic.logger.logging import setup_logging, get_logger
setup_logging()
logger = get_logger()


def read_yaml() -> dict:
    """
    read a yaml file and the container as a dictionaries
    """
    root = Path(__file__).resolve().parents[3]
    config_path = root / "config" / "config.yaml"
    try :
        with open(config_path, "rb") as yaml_file :
            config_info = yaml.safe_load(yaml_file) 
            return config_info
    except Exception as e :
        raise Agentic_Exception(e, sys) from e 


def create_dictinaries(path_to_directories:list, verbose = True) :
    """
    create list of directories
    agrs:
    path_to directories[list]: list of path of directories
    ignore if directories is already created.
    """
    for path in path_to_directories :
        os.makedirs(path, exist_ok=True)
        if verbose :
            logger.info(f"created directory at : {path}")
