import math
import os
import random
import shutil

from typing import Tuple, List


def _copy_image_and_annotation(source_directory: str, destination_directory: str, pair: Tuple[str, str]):
    image_source_path = os.path.join(source_directory, pair[0])
    image_target_path = os.path.join(destination_directory, pair[0])
    annotation_source_path = os.path.join(source_directory, pair[1])
    annotation_target_path = os.path.join(destination_directory, pair[1])

    os.makedirs(os.path.dirname(image_target_path), exist_ok=True)
    shutil.copy(image_source_path, image_target_path)
    shutil.copy(annotation_source_path, annotation_target_path)


def _get_randomized_sample_pairs(i_am_printed_directory: str, seed_for_reproducible_split: int = 0) -> List[
    Tuple[str, str]]:
    files = os.listdir(i_am_printed_directory)
    number_of_samples = int(len(files) / 2)

    random_indices = list(range(0, number_of_samples))
    random.seed(seed_for_reproducible_split)
    random.shuffle(random_indices)

    shuffled_sample_pairs = []

    for i in range(0, len(random_indices)):
        shuffled_sample_pairs.append((files[random_indices[i] * 2], files[random_indices[i] * 2 + 1]))

    return shuffled_sample_pairs


def split_dataset_into_training_and_test_sets(dataset_directory: str = "data"):
    return split_dataset_into_training_validation_and_test_sets(dataset_directory, 0.8, 0.2)


def split_dataset_into_training_validation_and_test_sets(dataset_directory: str = "data", training_precentage=0.6,
                                                         test_percentage=0.2):
    i_am_printed_directory = os.path.join(dataset_directory, "I AM printed")
    training_directory = os.path.join(dataset_directory, "training")
    validation_directory = os.path.join(dataset_directory, "validation")
    test_directory = os.path.join(dataset_directory, "test")

    if os.path.exists(training_directory):
        shutil.rmtree(training_directory)
    os.makedirs(training_directory)
    if os.path.exists(validation_directory):
        shutil.rmtree(validation_directory)
    os.makedirs(validation_directory)
    if os.path.exists(test_directory):
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    # remove wrong annotated image
    try:
        os.remove(os.path.join(i_am_printed_directory, "a04-006.png"))
        os.remove(os.path.join(i_am_printed_directory, "a04-006.xml"))
    except OSError:
        pass

    randomized_files = _get_randomized_sample_pairs(i_am_printed_directory)

    test_pairs = randomized_files[0:math.ceil(len(randomized_files) * test_percentage)]
    training_pairs = randomized_files[
                     len(test_pairs):len(test_pairs) + math.ceil(len(randomized_files) * training_precentage)]
    validation_pairs = randomized_files[len(training_pairs) + len(test_pairs):len(randomized_files)]

    print("Split a total of {0} samples into three sets with {1} training, {2} validation and {3} test samples.".format(
        len(randomized_files), len(training_pairs), len(validation_pairs), len(test_pairs)
    ))

    for i in range(0, len(test_pairs)):
        _copy_image_and_annotation(i_am_printed_directory, test_directory, test_pairs[i])

    for i in range(0, len(training_pairs)):
        _copy_image_and_annotation(i_am_printed_directory, training_directory, training_pairs[i])

    for i in range(0, len(validation_pairs)):
        _copy_image_and_annotation(i_am_printed_directory, validation_directory, validation_pairs[i])


if __name__ == "__main__":
    split_dataset_into_training_and_test_sets("data")
