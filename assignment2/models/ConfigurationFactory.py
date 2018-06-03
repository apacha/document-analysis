from typing import List

from models.Simple2Configuration import Simple2Configuration
from models.SimpleConfiguration import SimpleConfiguration
from models.TrainingConfiguration import TrainingConfiguration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  height: int,
                                  width: int,
                                  alphabet_length: int,
                                  absolute_maximum_string_length: int) -> TrainingConfiguration:

        all_configurations = ConfigurationFactory.get_all_configurations(height, width, alphabet_length,
                                                                         absolute_maximum_string_length)

        for i in range(len(all_configurations)):
            if all_configurations[i].name() == name:
                return all_configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(height: int, width: int, alphabet_length: int, absolute_maximum_string_length: int) -> \
    List[TrainingConfiguration]:
        all_configurations = [
            SimpleConfiguration(height, width, alphabet_length, absolute_maximum_string_length),
            Simple2Configuration(height, width, alphabet_length, absolute_maximum_string_length),
                              ]
        return all_configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1, 1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
