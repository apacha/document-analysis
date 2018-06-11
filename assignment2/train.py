import datetime
from typing import Dict

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import dataset_loader
from models.ConfigurationFactory import ConfigurationFactory


def train_model_for_ocr(dataset_directory: str, configuration_name: str, image_width: int, image_height: int, alphabet_length: int,
                        maximum_number_of_characters_in_longest_text_line: int,
                        text_line_image_to_text_mapping: Dict[str, str]) -> str:
    configuration = ConfigurationFactory.get_configuration_by_name(configuration_name, image_width, image_height,
                                                                   alphabet_length,
                                                                   maximum_number_of_characters_in_longest_text_line)
    print(configuration.summary())
    model = configuration.model()
    training_inputs, training_outputs = dataset_loader.load_dataset_split_into_memory(dataset_directory, "training",
                                                                                      text_line_image_to_text_mapping,
                                                                                      image_width, image_height,
                                                                                      maximum_number_of_characters_in_longest_text_line)
    start_of_training = datetime.date.today()
    model_output_name = "{0}_{1}_{2}x{3}.h5".format(start_of_training, configuration.name(), image_width, image_height)
    model_checkpoint = ModelCheckpoint(model_output_name, verbose=1, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor="val_loss", patience=configuration.number_of_epochs_before_early_stopping,
                                   verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                patience=configuration.number_of_epochs_before_reducing_learning_rate,
                                                verbose=1,
                                                factor=configuration.learning_rate_reduction_factor,
                                                min_lr=configuration.minimum_learning_rate)
    callbacks = [model_checkpoint, early_stopping, learning_rate_reduction]
    model.fit(x=training_inputs,
              y=training_outputs,
              batch_size=configuration.training_minibatch_size,
              epochs=configuration.number_of_epochs,
              validation_split=0.25,
              callbacks=callbacks)

    return model_output_name