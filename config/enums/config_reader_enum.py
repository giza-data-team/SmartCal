from enum import Enum


class ConfigReaderEnum(Enum):
    env = "EnvConfigReader"

    @staticmethod
    def get_value(name):
        for choice in ConfigReaderEnum:
            if choice.name == name:
                return choice.value
        raise ValueError(f"{name} is not a valid ConfigReaderEnum")
