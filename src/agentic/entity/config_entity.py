from agentic.exception import Agentic_Exception
from agentic.logger.logging import setup_logging, get_logger
from dataclasses import dataclass
from agentic.utils.config_yaml import read_yaml


@dataclass
class ConfigEntity:
    """
    Configuration entity that retrieves all information from the YAML file.
    """
    data_dir: dict

    @classmethod
    def load_config(cls):
        """
        Load configuration from the YAML file.
        
        Returns:
            ConfigEntity: Instance with loaded configuration.
        """
        yaml_info = read_yaml()
        return cls(data_dir=yaml_info.get('data_dir', {}))


if __name__ == "__main__":
    config = ConfigEntity.load_config()
    print("Loaded configuration:")
    print(f"Data directories: {config.data_dir}")
