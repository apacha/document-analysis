import argparse
import datetime
import os
from typing import Tuple, Dict

import editdistance
from keras.callbacks import ModelCheckpoint

import annotation_loader
import dataset_splitter
import image_to_lines_converter
import lines_to_window_converter
import ocr_downloader
from annotation_loader import load_alphabet_from_samples
from models.ConfigurationFactory import ConfigurationFactory
import numpy as np
import keras.backend as K

import tensorflow as tf


def text_to_labels(text, alphabet):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


def load_dataset(dataset_directory, dataset_split, text_line_image_to_text_mapping, image_width,
                 image_height_with_padding,
                 absolute_max_string_length) -> Tuple[Dict, Dict]:
    """ Loads all training images into a big numpy-array """

    files = os.listdir(os.path.join(dataset_directory, "lines-" + dataset_split))

    size = len(files)

    y_data = np.ones([size, absolute_max_string_length]) * -1
    y_lenghts = [0] * size
    pool_size = 2
    downsampling_factor = pool_size ** 2
    input_length = np.ones([size, 1]) * (image_width // downsampling_factor - 2)

    alphabet = load_alphabet_from_samples(text_line_image_to_text_mapping)
    images = []
    image_strings = []
    for i, file in enumerate(files):
        line_image = lines_to_window_converter.read_grayscale_image_and_add_padding(
            os.path.join(dataset_directory, "lines-" + dataset_split, file), image_height_with_padding)
        images.append(line_image)
        sentence = text_line_image_to_text_mapping[file]
        image_strings.append(sentence)
        y_data[i, 0:len(sentence)] = text_to_labels(sentence, alphabet)
        y_lenghts[i] = len(sentence)

    y_lenghts = np.expand_dims(np.array(y_lenghts), 1)

    x_data = np.stack(images)

    inputs = {'the_input': x_data,
              'the_labels': y_data,
              'input_length': input_length,
              'label_length': y_lenghts,
              }

    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

    return inputs, outputs


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

    training_inputs, training_outputs = load_dataset(dataset_directory, "training", text_line_image_to_text_mapping,
                                                     image_width, image_height, absolute_max_string_length)

    start_of_training = datetime.date.today()
    model_description = "{0}_{1}_{2}x{3}".format(start_of_training, configuration.name(), image_width, image_height)
    best_model_path = model_description + ".h5"

    model_checkpoint = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, monitor='val_loss')
    callbacks = [model_checkpoint]

    model.fit(x=training_inputs,
              y=training_outputs,
              batch_size=configuration.training_minibatch_size,
              epochs=configuration.number_of_epochs,
              validation_split=0.2,
              callbacks=callbacks)

    # TODO: dictionary correction
    # To compute edit distance between two words, use this method
    # edit_dist = editdistance.eval("hallo", "hello")
