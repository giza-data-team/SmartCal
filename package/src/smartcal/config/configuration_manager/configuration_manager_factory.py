from abc import ABC, abstractmethod
from decouple import Config, RepositoryEnv
import os
from pathlib import Path

from smartcal.config.enums.config_reader_enum import ConfigReaderEnum


class ConfigReader(ABC):
    @abstractmethod
    def get_config(self) -> Config:
        pass

    @abstractmethod
    def set_config(self, key: str, value: str):
        pass


class EnvConfigReader(ConfigReader):
    def __init__(self):
        __base_dir = Path(__file__).resolve().parent.parent.parent
        __file_name = f'config-dev.env'
        self.full_path = os.path.join(__base_dir, 'config', 'resources', 'conf', __file_name)

    def get_config(self) -> Config:
        return Config(repository=RepositoryEnv(self.full_path))

    def set_config(self, key: str, value: str):
        # Read all lines from the .env file
        if os.path.exists(self.full_path):
            with open(self.full_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        # Modify the content, updating the key if it exists, or appending if not
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}"):
                lines[i] = f"{key} = '{value}'\n"
                updated = True
                break

        if not updated:
            raise KeyError(f"The configuration key '{key}' was not found in the .env file.")

        # Write the entire updated content back to the .env file, replacing existing content
        with open(self.full_path, "w") as f:
            f.writelines(lines)


class ConfigManagerFactory:
    def __new__(cls, config_reader: ConfigReader.__name__ = ConfigReaderEnum.env.value):
        _factory_localizer = {subclass.__name__: subclass for subclass in ConfigReader.__subclasses__()}
        subclass = _factory_localizer.get(config_reader)
        return subclass()
