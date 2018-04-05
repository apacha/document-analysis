from typing import List

from models.ResNet50PretrainedConfiguration import ResNet50PretrainedConfiguration
from models.TrainingConfiguration import TrainingConfiguration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  width: int,
                                  height: int) -> TrainingConfiguration:

        configurations = ConfigurationFactory.get_all_configurations(width, height)

        for i in range(len(configurations)):
            if configurations[i].name() == name:
                return configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(width, height) -> List[TrainingConfiguration]:
        configurations = [ResNet50PretrainedConfiguration(width, height)]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
