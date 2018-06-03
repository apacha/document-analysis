import os
from typing import Tuple, Dict

import numpy as np

import lines_to_window_converter
from annotation_loader import load_alphabet_from_samples


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
              'ground_truth': image_strings
              }

    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

    return inputs, outputs
