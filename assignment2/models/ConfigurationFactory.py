from typing import List

from models.Simple2Configuration import Simple2Configuration
from models.SimpleConfiguration import SimpleConfiguration
from models.TrainingConfiguration import TrainingConfiguration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  width: int,
                                  height: int,
                                  alphabet_length: int,
                                  maximum_number_of_characters_in_longest_text_line: int) -> TrainingConfiguration:

        all_configurations = ConfigurationFactory.get_all_configurations(width, height, alphabet_length,
                                                                         maximum_number_of_characters_in_longest_text_line)

        for i in range(len(all_configurations)):
            if all_configurations[i].name() == name:
                return all_configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(width: int, height: int, alphabet_length: int,
                               maximum_number_of_characters_in_longest_text_line: int) -> List[TrainingConfiguration]:
        all_configurations = [
            SimpleConfiguration(width, height, alphabet_length, maximum_number_of_characters_in_longest_text_line),
            Simple2Configuration(width, height, alphabet_length, maximum_number_of_characters_in_longest_text_line),
        ]
        return all_configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1, 1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
