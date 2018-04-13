from typing import List

from models.DenseNetConfiguration import DenseNetConfiguration
from models.InceptionResNetV2Configuration import InceptionResNetV2Configuration
from models.InceptionResNetV2GapConfiguration import InceptionResNetV2GapConfiguration
from models.ResNet50Configuration import ResNet50Configuration
from models.ResNet50LargeBackendConfiguration import ResNet50LargeBackendConfiguration
from models.ResNet50GapConfiguration import ResNet50GapConfiguration
from models.TrainingConfiguration import TrainingConfiguration
from models.XceptionGapConfiguration import Xception_GAP_PretrainedConfiguration


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
        all_configurations = [ResNet50Configuration(width, height),
                              ResNet50LargeBackendConfiguration(width, height),
                              ResNet50GapConfiguration(width, height),
                              Xception_GAP_PretrainedConfiguration(width, height),
                              DenseNetConfiguration(width, height),
                              InceptionResNetV2Configuration(width, height),
                              InceptionResNetV2GapConfiguration(width, height),
                              ]
        return all_configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
