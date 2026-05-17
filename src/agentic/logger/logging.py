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

    
    # 2. Get the actual filename from the config
    # This ensures we create the directory for WHATEVER path is in the YAML
    log_file_path = config.get('handlers', {}).get('file', {}).get('filename')
    if log_file_path:
        log_dir = Path(log_file_path).parent
        os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(config)


def get_logger(name="agentic"):
    return logging.getLogger(name)