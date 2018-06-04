import argparse
import itertools

import numpy as np
from keras import models

import annotation_loader
import dataset_loader


def predict(dataset_directory: str, model_path: str, image_height: int, image_width: int,
            absolute_max_string_length=146):
    mapping = annotation_loader.load_mapping(dataset_directory)
    alphabet = annotation_loader.load_alphabet_from_samples(mapping)

    print("Loading model...")
    m = models.load_model(model_path, custom_objects={'<lambda>': lambda y_true, y_predict: y_predict})

    inputs, _ = dataset_loader.load_dataset(dataset_directory, "test", mapping, image_width, image_height,
                                            absolute_max_string_length)
    prediction = m.predict(inputs)

    softmax_activations = prediction[1]

    # Best path decoding
    for index, sample_activations in enumerate(softmax_activations):
        out_best = list(np.argmax(sample_activations[2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        decoded = ""
        for c in out_best:
            if c == len(alphabet):
                continue  # CTC Blank
            else:
                decoded += (alphabet[c])

        ground_truth = inputs['ground_truth'][index]
        print("Decoded: {0}\nGT:      {1}\n".format(decoded, ground_truth))
        # TODO: Perform dictionary correction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_path", type=str, default="2018-06-03_simple_1900x64_val_loss_3.24.h5",
                        help="The trained model")
    parser.add_argument("--height", default=64, type=int, help="Height of the input-images for the network in pixel")
    parser.add_argument("--width", default=1900, type=int, help="Width of the input-images for the network in pixel")
    parser.add_argument("--absolute_max_string_length", default=146, type=int, help="Max length of strings")

    flags, unparsed = parser.parse_known_args()

    predict(flags.dataset_directory, flags.model_path, flags.height, flags.width, flags.absolute_max_string_length)
