import logging
import logging.config
import yaml
from pathlib import Path
import os


def setup_logging():

    root = Path(__file__).resolve().parents[3]

    config_path = root / "config" / "logging.yaml"

    os.makedirs(root / "logs", exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)


def get_logger(name="agentic"):
    return logging.getLogger(name)