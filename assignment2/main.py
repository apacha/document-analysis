import argparse
import os
from typing import Tuple

import editdistance

import annotation_loader
import dataset_splitter
import image_to_lines_converter
import lines_to_window_converter
import ocr_downloader
from annotation_loader import load_alphabet_from_samples
from models.ConfigurationFactory import ConfigurationFactory
import numpy as np
import keras.backend as K

def text_to_labels(text, alphabet):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

def load_dataset(dataset_directory, text_line_image_to_text_mapping, padded_line_height) -> Tuple[
    np.ndarray, np.ndarray]:
    "Loads all training images into a big numpy-array"
    files = os.listdir(os.path.join(dataset_directory, "lines-training"))

    absolute_max_string_len = 146
    y_train = np.ones([len(files), absolute_max_string_len], dtype=np.int8) * -1
    y_lenghts = [0] * len(files)

    alphabet = load_alphabet_from_samples(text_line_image_to_text_mapping)
    images = []
    image_strings = []
    for i, file in enumerate(files):
        line_image = lines_to_window_converter.read_grayscale_image_and_add_padding(
            os.path.join(dataset_directory, "lines-training", file), padded_line_height)
        # flipped_line_image = line_image [:, :, ::-1].copy()
        images.append(line_image)
        sentence = text_line_image_to_text_mapping[file]
        image_strings.append(sentence)
        y_train[i, 0:len(sentence)] = text_to_labels(sentence, alphabet)
        y_lenghts[i] = len(sentence)

    y_lenghts = np.expand_dims(np.array(y_lenghts), 1)

    x_train = np.stack(images).astype(np.uint8)

    return x_train, y_train


def get_batch(self, index, size, train):
    # width and height are backwards from typical Keras convention
    # because width is the time dimension when it gets fed into the RNN
    if K.image_data_format() == 'channels_first':
        X_data = np.ones([size, 1, self.img_w, self.img_h])
    else:
        X_data = np.ones([size, self.img_w, self.img_h, 1])

    labels = np.ones([size, self.absolute_max_string_len])
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []
    for i in range(size):
        # Mix in some blank inputs.  This seems to be important for
        # achieving translational invariance
        if train and i > size - 4:
            if K.image_data_format() == 'channels_first':
                X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
            else:
                X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
            labels[i, 0] = self.blank_label
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = 1
            source_str.append('')
        else:
            if K.image_data_format() == 'channels_first':
                X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
            else:
                X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
            labels[i, :] = self.Y_data[index + i]
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = self.Y_len[index + i]
            source_str.append(self.X_text[index + i])
    inputs = {'the_input': X_data,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              'source_str': source_str  # used for visualization only
              }
    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
    return (inputs, outputs)

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
    # ocr_downloader.download_i_am_printed_database(dataset_directory)
    #
    # dataset_splitter.split_dataset_into_training_validation_and_test_sets(dataset_directory)
    #
    # maximal_line_height = image_to_lines_converter.split_images_into_text_lines(dataset_directory)

    text_line_image_to_text_mapping = annotation_loader.load_mapping(dataset_directory)
    annotation_loader.remove_lines_without_matching_annotation(dataset_directory, text_line_image_to_text_mapping)

    # To compute edit distance between two words, use this method
    # edit_dist = editdistance.eval("hallo", "hello")

    configuration = ConfigurationFactory.get_configuration_by_name("simple", 1900, 64, 77, 146)
    print(configuration.summary())
    model = configuration.model()

    x_train, y_train = load_dataset(dataset_directory, text_line_image_to_text_mapping, 64)

    model.fit(x_train, y_train, configuration.training_minibatch_size, configuration.number_of_epochs)

    # dictionary correction
