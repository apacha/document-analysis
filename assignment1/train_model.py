from glob import glob
import os

from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import numpy as np

from directory_iterator_with_target import DirectoryIteratorWithTarget, load_mapping
from models.ConfigurationFactory import ConfigurationFactory

if __name__ == "__main__":
    filename_to_target_mapping = load_mapping()
    batch_size = 16
    training_data_generator = DirectoryIteratorWithTarget("data/images", ImageDataGenerator(), filename_to_target_mapping,
                                                          target_size=(224, 400), batch_size=batch_size)

    training_configuration = ConfigurationFactory.get_configuration_by_name("res_net_50", 400, 224)
    training_steps_per_epoch = np.math.ceil(training_data_generator.samples / training_data_generator.batch_size)

    model = training_configuration.classifier()
    model.summary()

    print("Training on dataset...")
    history = model.fit_generator(
        generator=training_data_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_configuration.number_of_epochs,
        workers=4
    )


