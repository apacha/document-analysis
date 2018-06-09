import argparse
import itertools

import numpy as np
from keras import models

import annotation_loader
import dataset_loader
import editdistance

from spellchecker import SpellChecker


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

    # init evaluation parameter
    word_error_count = 0
    word_quantity = 0
    spell_checker = SpellChecker()

    without_spell_cor_edit_dist = 0

    with_spell_cor_ground_truth = ""
    with_spell_cor_decoded = ""
    with_spell_cor_edit_dist = 0

    total_without_spell_cor_edit_dist = 0
    total_with_spell_cor_edit_dist = 0

    # Best path decoding
    print("Spelling correction...")
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
        edit_dist = editdistance.eval(decoded, ground_truth)
        without_spell_cor_edit_dist += edit_dist

        print("Decoded: {0}\nGT:      {1}\ndist:  {2}\n".format(decoded, ground_truth, edit_dist))

        if ground_truth.endswith("-"):
            with_spell_cor_ground_truth += ground_truth[:len(ground_truth) - 1]
            if decoded.endswith("-"):
                with_spell_cor_decoded += decoded[:len(decoded) - 1]
            else:
                with_spell_cor_decoded += decoded
        else:
            with_spell_cor_ground_truth += ground_truth + " "
            with_spell_cor_decoded += decoded + " "

        if with_spell_cor_ground_truth.endswith(". "):
            word_list = spell_checker.words(with_spell_cor_decoded)
            misspelled = spell_checker.unknown(word_list)

            # ignore numbers and 'quot' words
            misspelled_without_numbers = [x for x in misspelled if
                                          not ((x.isdigit() or x[0] == '-' and x[1:].isdigit()) or 'quot' in x)]

            for word in misspelled_without_numbers:
                new_word = spell_checker.correction(word)
                with_spell_cor_decoded = with_spell_cor_decoded.replace(word, new_word)  # lower case
                with_spell_cor_decoded = with_spell_cor_decoded.replace(word[0].upper() + word[1:],
                                                                        new_word[0].upper() + new_word[
                                                                                              1:])  # upper case

                with_spell_cor_edit_dist = editdistance.eval(with_spell_cor_decoded, with_spell_cor_ground_truth)

            print("with SpellCor - Decoded: {0}\nwith SpellCor - GT:      {1}\ndist:  {2} - {3}\n".format(
                with_spell_cor_decoded,
                with_spell_cor_ground_truth,
                with_spell_cor_edit_dist,
                without_spell_cor_edit_dist))

            # count how many words are not correctly detected from ground truth
            words_ground_truth = with_spell_cor_ground_truth.split()
            words_decoded = with_spell_cor_decoded.split()

            word_error_count += len(set(words_ground_truth) - set(words_decoded).intersection(words_ground_truth))
            word_quantity += len(words_ground_truth)

            total_without_spell_cor_edit_dist += without_spell_cor_edit_dist
            total_with_spell_cor_edit_dist += with_spell_cor_edit_dist

            with_spell_cor_ground_truth = ""
            with_spell_cor_decoded = ""
            with_spell_cor_edit_dist = 0
            without_spell_cor_edit_dist = 0

    print("Error rate: words {0}/{1} - {2}%".format(word_error_count, word_quantity, word_error_count / word_quantity))
    print("Error rate: dist without spell correction {0}/{1} - {2}%".format(total_without_spell_cor_edit_dist,
                                                                            word_quantity,
                                                                            total_without_spell_cor_edit_dist / word_quantity))
    print("Error rate: dist with spell correction {0}/{1} - {2}%".format(total_with_spell_cor_edit_dist, word_quantity,
                                                                         total_with_spell_cor_edit_dist / word_quantity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_path", type=str, default="2018-06-04_simple_1900x64.h5",
                        help="The trained model")
    parser.add_argument("--height", default=64, type=int, help="Height of the input-images for the network in pixel")
    parser.add_argument("--width", default=1900, type=int, help="Width of the input-images for the network in pixel")
    parser.add_argument("--absolute_max_string_length", default=146, type=int, help="Max length of strings")

    flags, unparsed = parser.parse_known_args()

    predict(flags.dataset_directory, flags.model_path, flags.height, flags.width, flags.absolute_max_string_length)
