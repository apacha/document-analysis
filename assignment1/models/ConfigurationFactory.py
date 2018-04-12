from typing import List

from models.DenseNetPretrainedConfiguration import DenseNetPretrainedConfiguration
from models.InceptionResNetV2PretrainedConfiguration import InceptionResNetV2PretrainedConfiguration
from models.ResNet50PretrainedConfiguration import ResNet50PretrainedConfiguration
from models.ResNet50_2_PretrainedConfiguration import ResNet50_2_PretrainedConfiguration
from models.TrainingConfiguration import TrainingConfiguration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  width: int,
                                  height: int) -> TrainingConfiguration:

        all_configurations = ConfigurationFactory.get_all_configurations(width, height)

        for i in range(len(all_configurations)):
            if all_configurations[i].name() == name:
                return all_configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(width, height) -> List[TrainingConfiguration]:
        all_configurations = [ResNet50PretrainedConfiguration(width, height),
                              ResNet50_2_PretrainedConfiguration(width, height),
                              DenseNetPretrainedConfiguration(width, height),
                              InceptionResNetV2PretrainedConfiguration(width, height),
                              ]
        return all_configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
