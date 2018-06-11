import argparse
import itertools

import numpy as np
from keras import models

import annotation_loader
import dataset_loader
import editdistance


def predict_and_evaluate(dataset_directory: str, model_path: str,
                         maximum_number_of_characters_in_longest_text_line=146):
    mapping = annotation_loader.load_mapping(dataset_directory)
    alphabet = annotation_loader.load_alphabet_from_samples(mapping)

    print("Loading model...")
    # We are forced to provide a implementation for the lambda-operation, that was not serialized
    m = models.load_model(model_path, custom_objects={'<lambda>': lambda y_true, y_predict: y_predict})
    (_, image_width, image_height, _) = m.layers[0].input.shape
    image_height, image_width = int(image_height), int(image_width)

    inputs, _ = dataset_loader.load_dataset_split_into_memory(dataset_directory, "test", mapping, image_width,
                                                              image_height,
                                                              maximum_number_of_characters_in_longest_text_line)
    prediction = m.predict(inputs)

    softmax_activations = prediction[1]

    word_quantity = 0

    # create dictionary for spelling correction
    dictionary = set()
    for line in inputs['ground_truth']:
        ground_truth_line = line.split(" ")
        word_quantity += len(ground_truth_line)
        ground_truth_line = [
            x.replace('.', '') and x.replace('&quot;', '') and x.replace(',', '') and x.replace(';', '') for x in
            ground_truth_line]
        dictionary = set().union(set(ground_truth_line), dictionary)

    dictionary.add('&quot')

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
        edit_dist_without_spell_correction = editdistance.eval(decoded, ground_truth)

        # proof if word exist
        # test_correct_elements = set(w for w in words if w in dictionary)
        decoded_words = decoded.split(" ")
        decoded_words = [x.replace('.', '') and x.replace('&quot;', '') and x.replace(',', '') and x.replace(';', '')
                         for x in decoded_words]
        text_incorrect_elements = set(w for w in decoded_words if w not in dictionary)

        minDistWord = ""
        decoded_new = decoded

        # if word does not exist - calculate shortest distance
        for w1 in text_incorrect_elements:
            min = 1000
            for w2 in dictionary:
                dist = editdistance.eval(w1, w2)
                if min > dist:
                    min = dist
                    minDistWord = w2

            decoded_new = decoded_new.replace(w1, minDistWord)

        edit_dist_with_spell_correction = editdistance.eval(decoded_new, ground_truth)
        total_with_spell_cor_edit_dist += edit_dist_with_spell_correction
        total_without_spell_cor_edit_dist += edit_dist_without_spell_correction

        print("Decoded:  {0}\nDecoded-C:{1}\nGT:       {2}\ndist:  {3} {4}\n".format(decoded, decoded_new, ground_truth,
                                                                                     edit_dist_without_spell_correction,
                                                                                     edit_dist_with_spell_correction))

    print("Error rate: dist without spell correction {0}/{1} - {2}%".format(total_without_spell_cor_edit_dist,
                                                                            word_quantity,
                                                                            total_without_spell_cor_edit_dist / word_quantity))
    print("Error rate: dist with spell correction {0}/{1} - {2}%".format(total_with_spell_cor_edit_dist,
                                                                         word_quantity,
                                                                         total_with_spell_cor_edit_dist / word_quantity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_path", type=str, default="2018-06-11_simple_1900x64.h5",
                        help="The trained model")
    parser.add_argument("--maximum_number_of_characters_in_longest_text_line", default=146, type=int,
                        help="Max length of strings")

    flags, unparsed = parser.parse_known_args()

    predict_and_evaluate(flags.dataset_directory, flags.model_path,
                         flags.maximum_number_of_characters_in_longest_text_line)
