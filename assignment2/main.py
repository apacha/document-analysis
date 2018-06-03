import argparse
import datetime

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import annotation_loader
import dataset_loader
import dataset_splitter
import image_to_lines_converter
import ocr_downloader
from models.ConfigurationFactory import ConfigurationFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_known_args()

    dataset_directory = flags.dataset_directory

    ocr_downloader.download_i_am_printed_database(dataset_directory)
    dataset_splitter.split_dataset_into_training_and_test_sets(dataset_directory)
    maximal_line_height = image_to_lines_converter.split_images_into_text_lines(dataset_directory)
    text_line_image_to_text_mapping = annotation_loader.load_mapping(dataset_directory)
    annotation_loader.remove_lines_without_matching_annotation(dataset_directory, text_line_image_to_text_mapping)

    image_width, image_height = 1900, 64
    absolute_max_string_length = 146
    alphabet_length = 77
    configuration = ConfigurationFactory.get_configuration_by_name("simple", image_width, image_height, alphabet_length,
                                                                   absolute_max_string_length)
    print(configuration.summary())
    model = configuration.model()

    training_inputs, training_outputs = dataset_loader.load_dataset(dataset_directory, "training",
                                                                    text_line_image_to_text_mapping,
                                                                    image_width, image_height,
                                                                    absolute_max_string_length)

    start_of_training = datetime.date.today()
    model_description = "{0}_{1}_{2}x{3}".format(start_of_training, configuration.name(), image_width, image_height)
    best_model_path = model_description + ".h5"

    model_checkpoint = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, monitor='val_loss')
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
              validation_split=0.2,
              callbacks=callbacks)

    # TODO: dictionary correction
    # To compute edit distance between two words, use this method
    # edit_dist = editdistance.eval("hallo", "hello")
